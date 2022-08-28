# coding=utf-8

from collections import OrderedDict

import torch
import sys

sys.path.append('../')
import model.clip as clip
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import BertTokenizer, BertModel
import util
from loss import *
from bigfile import BigFile
from generic_utils import Progbar

def _initialize_weights(m):
    """Initialize module weights
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm1d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def to_device_and_float16(x: torch.Tensor):
    x = x.to(device)
    # if float16:
    #     x = x.half()
    return x

class TxtEncoder(nn.Module):
    def __init__(self, opt):
        super(TxtEncoder, self).__init__()

    def forward(self, caption_feat_dict, task3=False):
        output = {}
        output['text_features'] = caption_feat_dict['caption']

        return output

class VisEncoder(nn.Module):
    def __init__(self, opt):
        super(VisEncoder, self).__init__()

    def forward(self, feat, task3=False):
        return feat


class CLIPEncoder(nn.Module):
    """
    CLIP encoder.
    transform text and image into features.
    """

    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.Clip_name = opt.text_encoding['CLIP_encoding']['name']
        self.frozen = opt.clip_opt['frozen']
        self.dim = opt.clip_opt['size']
        self.tokenizer = clip.tokenize
        self.tokenizer_withmask=clip.tokenizewithmask
        self.simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.ClipModel, self.preprocess,crossdict = clip.load(self.Clip_name, device=device, jit=False,cross=False)

        self.hidden_size=self.ClipModel.hiddensize
        self.vocab_size=self.ClipModel.vocab_size
    def forward(self, caption_feat_dict, vis_origin_frame_tuple=None, task3=False,
                frame_agg_method='mean'):
        """

        :param caption_feat_dict:
        :param vis_origin_frame_tuple: ([sample_frame, 3, 224, 224], ...)
        :param task3:
        :return: (batch_size, dim)
        """
        output = {}
        # For text encoding
        if caption_feat_dict is not None:
            if 'CLIP_encoding' in caption_feat_dict and self.frozen:
                text_features = caption_feat_dict['CLIP_encoding']
            else:
                text = caption_feat_dict['clipcaption']
                #text,_ ,_= self.tokenizer(caption_feat_dict['caption'])
                text=to_device_and_float16(text)
                if self.opt.clip_opt['frozen']:
                    with torch.no_grad():
                        text_features = self.ClipModel.encode_text(text)
                else:
                    text_features = self.ClipModel.encode_text(text)
            output['text_features'] = text_features

        # For visual encoding
        if vis_origin_frame_tuple is not None:
            batch_size = len(vis_origin_frame_tuple)
            origin_frames = to_device_and_float16(torch.cat(vis_origin_frame_tuple, dim=0))

            if self.frozen:
                with torch.no_grad():
                    frame_features,frames_hiddenstate = self.ClipModel.encode_image(origin_frames)
            else:
                frame_features,frames_hiddenstate = self.ClipModel.encode_image(origin_frames)
            frame_features = frame_features.reshape((batch_size, -1, self.dim))
            frames_hiddenstate = frames_hiddenstate.reshape((batch_size, -1, self.dim))
            if frame_agg_method == 'mean':
                visual_features = torch.mean(frame_features, dim=1)
            else:
                raise Exception("frame_agg_method is not applied.")

            output['visual_features'] = visual_features
            output['visual_hiddenstate'] = frames_hiddenstate

        return output




class T2vmodel(nn.Module):
    """
    Base T2vModel
        """

    def _init_vis_net(self, opt):
        self.vis_net = VisEncoder(opt)

    def _init_txt_net(self, opt):
        self.txt_net = TxtEncoder(opt)

    def __init__(self, opt):
        super().__init__()
        self.scaler = GradScaler()
        if opt is None:
            return
        self._init_vis_net(opt)
        self._init_txt_net(opt)

        self.opt = opt
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = False

        self.criterion = MarginRankingLoss(margin=opt.margin,
                                           measure=opt.measure,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction,
                                           device=device)

        self.params = list(self.parameters())  # 所有 params

        # 设置学习率
        params_special = []
        params_usual = []
        for name, parm in list(self.named_parameters()):
            if ('BertModel' in name) or ('csn_model' in name) or ('ClipModel' in name):
                params_special.append(parm)
            else:
                params_usual.append(parm)
        params = [{'params': params_usual},
                  {'params': params_special, 'lr': opt.lr / 20}]

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                              torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                         patience=2)]

        self.iters = 0

    def get_txt2vis_matrix(self, txt_embs, vis_embs, measure='cosine'):
        if len(vis_embs.shape) == len(txt_embs.shape) == 2:
            txt2vis_sim = self.compute_sim(txt_embs, vis_embs, measure, device)

        elif len(vis_embs.shape) == len(txt_embs.shape) == 3:
            for j, each in enumerate(range(vis_embs.size(1))):
                txt2vis_sim_temp = self.compute_sim(txt_embs[:, each, :], vis_embs[:, each, :], measure,
                                                    device).unsqueeze(0)
                txt2vis_sims = txt2vis_sim_temp if j == 0 else torch.cat(
                    (txt2vis_sims, txt2vis_sim_temp), dim=0)

            txt2vis_sim = torch.mean(txt2vis_sims, dim=0)

        return txt2vis_sim

    def compute_loss(self, vis_embs, txt_embs, vis_embs_multi_labels, txt_embs_multi_labels, labels_embs):
        """Compute the loss given pairs of image and caption embeddings
        """
        if len(vis_embs.shape) == len(txt_embs.shape) == 2:
            triplet_loss = self.criterion(txt_embs, vis_embs)
            multi_label_loss_vis = 0
            multi_label_loss_txt = 0
            multi_label_triplet_loss = 0
            loss = triplet_loss + multi_label_loss_vis + multi_label_loss_txt + multi_label_triplet_loss
            loss_items = {
                'triplet_loss': triplet_loss
            }
        elif len(vis_embs.shape) == len(txt_embs.shape) == 3:
            triplet_loss_multi_head = 0
            for each in range(vis_embs.size(1)):
                triplet_loss_multi_head += self.criterion(txt_embs[:, each, :], vis_embs[:, each, :])
            loss = triplet_loss_multi_head
            loss_items = {
                'triplet_loss': triplet_loss_multi_head
            }
        else:
            raise Exception("vis_embs dims are not equal to txt_embs dims")
        return loss, loss_items


    @staticmethod
    def compute_sim(query_embs, retro_embs, measure='cosine', device=torch.device('cuda')):
        query_embs = query_embs.to(device)
        retro_embs = retro_embs.to(device)
        if measure == 'cosine':
            return cosine_sim(query_embs, retro_embs)
        elif measure == 'hist':
            return hist_sim(query_embs, retro_embs)
        elif measure == 'euclidean':
            raise Exception('Not implemented')
        else:
            raise Exception('%s is invalid' % measure)

    @property
    def learning_rate(self):
        """Return learning rate"""
        lr_list = []
        for param_group in self.optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list

    def lr_step(self, val_value):
        """
        降低学习率
        :param val_value:
        :return:
        """
        self.lr_schedulers[0].step()
        self.lr_schedulers[1].step(val_value)


    def cal_foward(self, train_data,epoch=None):
        (vis_input, caption_feat_dict, labels_input,
         vis_frame_feat_dict_input,
         vis_origin_frame_tuple) = (
            train_data['vis_feats'], train_data['captions'],
            train_data['captions_task2'], train_data['vis_frame_feat_dict'],
            train_data['vis_origin_frame_tuple']
        )
        if vis_frame_feat_dict_input == {}:
            vis_frame_feat_dict_input = None
        # compute the embeddings
        txt_embs = self.txt_net(caption_feat_dict)
        vis_embs = self.vis_net(vis_input, txt_emb=txt_embs,
                                vis_frame_feat_dict_input=vis_frame_feat_dict_input)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, loss_items = self.compute_loss(vis_embs, txt_embs, 0, 0, 0)
        # print("triplet_loss and multi_label_loss_vis", loss_items, end='\r')

        return loss, loss_items

    def forward(self, train_data, epoch=None):
        """One training step given vis_feats and captions.
        """

        self.iters += 1

        if float16:
            # 前向过程(model + loss)开启 autocast
            with autocast():
                loss, loss_items = self.cal_foward(train_data,epoch)

            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                clip_grad_norm_(self.params, self.grad_clip)

            # scaler.step() unscale之前放大后的梯度，但是scale太多可能出现inf或NaN
            # 故其会判断是否出现了inf/NaN
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 如果检测到出现了inf或者NaN，就跳过这次梯度更新，同时动态调整scaler的大小
            self.scaler.step(self.optimizer)
            # 查看是否要更新scaler,这个要注意不能丢
            self.scaler.update()
        else:
            loss, loss_items = self.cal_foward(train_data)
            # compute gradient and do SGD step
            loss.backward()
            if self.grad_clip > 0:
                clip_grad_norm_(self.params, self.grad_clip)
            self.optimizer.step()

        return loss_items

    @util.timer
    def predict_multi(self, txt_loader, vis_loader, measure, record_emb=False):
        if vis_loader.dataset.length > 5e4:
            return self.predict_batch(txt_loader, vis_loader, measure, record_emb)
        self.eval()

        txt_ids = []
        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.vis_net(vis_input, vis_frame_feat_dict_input=vis_frame_feat_dict,vis_origin_frame_tuple=vis_origin_frame_tuple).cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            labels=[]
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids,label) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.txt_net(caption_feat_dict)
                labels.extend(label)
                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break
                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, self.vis_ids,labels


class W2VVPP_FrozenClip(T2vmodel):
    """
    use extracted clip feature for  prediction
        """

    class SimpleVisMudule(nn.Module):
        def __init__(self):
            """
            简单返回
            """
            super().__init__()

        def forward(self, vis_input: dict, txt_emb=None, vis_frame_feat_dict_input=None):
            vis_feature = torch.cat(list(vis_input.values()), dim=1)  # batch_size, vis_feature_concat

            return vis_feature

    class SimpleTxtMudule(nn.Module):

        def __init__(self, opt):
            super().__init__()
            self.encoder = CLIPEncoder(opt)  # return a dict

        def forward(self, caption_feat_dict):
            features = self.encoder(caption_feat_dict)['text_features']
            return features

    def _init_txt_net(self, opt):
        self.txt_net = W2VVPP_FrozenClip.SimpleTxtMudule(opt)

    def _init_vis_net(self, opt):
        self.vis_net = W2VVPP_FrozenClip.SimpleVisMudule()

    def __init__(self, opt):
        super().__init__(opt)



class Clip(T2vmodel):
    """
        use clip as txt/vis encoder
        """

    def __init__(self, opt):
        super().__init__(None)
        self.clip_model = CLIPEncoder(opt)

        self.opt = opt



    def predictneg_adhoc(self, txt_loader, vis_loader, measure, record_emb=False, neg_method="sub"):
        self.eval()
        txt_ids = []

        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []
        labels = []
        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    # if j>0:
                    #     break
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.clip_model(
                        caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
                    )['visual_features'].cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            negscores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            negmasks = []
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids, neginfo, negmask, videoids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue
                labels.extend(videoids)
                negidx = [[txt_idxs[k]] for k in range(len(negmask)) if negmask[k] > 0]
                negmasks.extend(negmask)
                text = to_device_and_float16(caption_feat_dict['clipcaption'])
                txt_embs = self.clip_model.ClipModel.encode_text(text)
                if len(neginfo) > 0:
                    neginfo = to_device_and_float16(neginfo)
                    negtxt_embs = self.clip_model.ClipModel.encode_text(neginfo)
                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break
                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if len(neginfo) > 0:
                        negscore = self.get_txt2vis_matrix(negtxt_embs, vis_embs, measure=measure).float()
                        negscore = negscore.clamp(min=0)

                    if i != len(txt_loader) - 1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    if len(neginfo) > 0:
                        negscores[negidx, idxs] = negscore.cpu()
                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)
            tempnegmask = torch.Tensor(negmasks).unsqueeze(1).repeat(1, len(self.vis_ids))
            scores = (scores + 1) / 2
            negscores = (negscores + 1) / 2
            if neg_method == "sub":
                scores = scores - negscores
            elif neg_method == "mul":
                scores = scores * (1 - negscores)
        return scores.detach().numpy(), txt_ids, self.vis_ids, labels

    @util.timer
    def predict_multi(self,txt_loader, vis_loader, measure, record_emb=False):
        """
        :param txt_loader:
        :param vis_loader:
        :param measure:
        :param record_emb: record the video_all_embs and accelerate the prediction.
        :return:
        """
        self.eval()

        txt_ids = []

        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.clip_model(
                        caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
                    )['visual_features'].cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            labels=[]
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids,video_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                #txt_embs = self.clip_model(caption_feat_dict)['text_features']
                text = to_device_and_float16(caption_feat_dict['clipcaption'])
                txt_embs = self.clip_model.ClipModel.encode_text(text)
                labels.extend(video_ids)
                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break

                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)
        print(len(labels))
        return scores.detach().numpy(), txt_ids, self.vis_ids,labels






from model.negclip import End2EndClip_withNeg


def get_model(name, device_, config):
    global device
    global float16
    device = device_
    float16 = config.float16
    NAME_TO_MODELS = {
        'Clip': Clip, #clip
        'End2EndClip_withNeg': End2EndClip_withNeg, #clip-bnl
    }
    assert name in NAME_TO_MODELS, '%s not supported.' % name

    model_ = NAME_TO_MODELS[name](config)
    model_ = model_.float().to(device_)
    return model_

# if __name__ == '__main__':
#     global device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = get_model('w2vvpp', device)