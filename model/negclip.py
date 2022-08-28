import sys

import loss

sys.path.append('../')
import torch.nn.init
from model.model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class End2EndClip_withNeg(T2vmodel):
    """
    w2v端到端 clip model

        输入视频帧信息和原始文本，使用 clip 模型计算匹配。
        """

    def __init__(self, opt):
        super().__init__(None)

        self.clip_model = CLIPEncoder(opt)
        self.opt = opt
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = False
        self.criterion = MarginRankingLossWithScore(margin=opt.margin,
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
                  {'params': params_special, 'lr': opt.lr / 100}]

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                              torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                         patience=2)]

        self.iters = 0
        self.criterion_task3 = Margin2Loss(neg_weight=opt.task3_neg_weight,bottommargin=opt.task3_bottommargin,uppermargin=opt.task3_uppermargin,
                                           bottommargin_t2t=opt.task3_bottommargin_t2t,uppermargin_t2t=opt.task3_uppermargin_t2t,
                                          measure=opt.measure,
                                          cost_style=opt.cost_style, device=device)
        self.softmax = nn.Softmax(dim=1)
        self.celoss =nn.CrossEntropyLoss(ignore_index=-1)
        self.log_softmax = nn.LogSoftmax(dim = 1)
        self.neg_t2t_weight=opt.neg_t2t_weight
        if hasattr(opt,"neg_t2v_weight"):
            self.neg_t2v_weight = opt.neg_t2v_weight
        self.neg_visgrad=opt.neg_visgrad
        self.no_norm = opt.no_norm

    def predict_multi(self,txt_loader, vis_loader, measure, record_emb=False,debug=False):
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

                    ( idxs, batch_vis_ids, vis_origin_frame_tuple
                     ) = (
                        output_dict['idxs'],
                        output_dict['vis_ids'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.vis_ids.extend(batch_vis_ids)

                    if debug and j > 1:
                        continue
                    vis_embs = self.clip_model(
                        caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
                    )['visual_features'].cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)
                    self.video_idxs_list.append(idxs)


            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            labels=[]
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids,video_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue
                txt_ids.extend(batch_txt_ids)
                labels.extend(video_ids)
                if debug and i > 1:
                    continue

                #txt_embs = self.clip_model(caption_feat_dict)['text_features']
                text = to_device_and_float16(caption_feat_dict['clipcaption'])
                txt_embs = self.clip_model.ClipModel.encode_text(text)

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


        return scores.detach().numpy(), txt_ids, self.vis_ids,labels






    def compute_loss(self, vis_embs, txt_embs, false_txt_embs,mask_task3 ):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss,loss_item=self.compute_triplet_loss( vis_embs, txt_embs, false_txt_embs, mask_task3)

        return loss, loss_item

    def compute_triplet_loss(self, vis_embs, txt_embs,false_txt_embs=None,mask_task3=None):
        """Compute the loss given pairs of image and caption embeddings
        """

        score=cosine_sim(txt_embs,vis_embs )
        loss = self.criterion(score)
        loss_items = {
            'triplet_loss': loss}
        if false_txt_embs is not None :

            task3_index = np.where(mask_task3 > -1)[0]
            mask_task3 = mask_task3[task3_index]
            negmask = torch.Tensor(mask_task3).to(device)
            if self.neg_visgrad:
                vis_embs2=vis_embs[task3_index,:]
            else:
                vis_embs2 = vis_embs[task3_index, :]
                vis_embs2 = vis_embs2.detach()
            los_t2t,los_t2v = self.criterion_task3(txt_embs[task3_index,:],vis_embs2 , false_txt_embs[:, :]
                                                      )
            if not self.no_norm:
                los_t2t=los_t2t/len(mask_task3)*txt_embs.shape[0]
                los_t2v=los_t2v/len(mask_task3)*txt_embs.shape[0]
            loss_items['los_t2t'] = los_t2t
            loss_items['los_t2v']=los_t2v
            loss += los_t2t * self.neg_t2t_weight
            loss += los_t2v * self.neg_t2v_weight

        return loss, loss_items



    def cal_foward(self, train_data,epoch):
        ( caption_feat_dict,

         vis_origin_frame_tuple,falsecaption,captions_neg_flag) = (
            train_data['captions'],
            train_data['vis_origin_frame_tuple']
           ,train_data['falsecaption'],train_data['captions_task3_mask']
        )
        # compute the embeddings

        output = self.clip_model(
            caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
        )
        vis_embs=output["visual_features"]
        origin_txt_feat = None
        #有反例的句子
        text = to_device_and_float16(caption_feat_dict['clipcaption'])
        txt_embs= self.clip_model.ClipModel.encode_text(text)
            # 错误query
        falsecaptionid = to_device_and_float16(falsecaption["clipcaption"])
        false_embs = self.clip_model.ClipModel.encode_text(falsecaptionid)

        self.optimizer.zero_grad()

        loss, loss_items = self.compute_loss(vis_embs, txt_embs, false_embs,  captions_neg_flag)
               # measure accuracy and record loss

        # print("triplet_loss and multi_label_loss_vis", loss_items, end='\r')

        return loss, loss_items
