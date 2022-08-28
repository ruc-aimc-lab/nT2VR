# -*-coding:utf-8 -*-
# --------------------------------------------------------
# Pytorch W2VV++
# Written by Xirong Li & Chaoxi Xu
# Modified by Jie Wang & Fan Hu
# --------------------------------------------------------


import sys
import time
import json
import shutil
import importlib
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
import torch.distributed as dist
import util
import evaluation
import data_providerneg as data
from common import *
from bigfile import BigFile
from generic_utils import Progbar
from model.model import get_model
from do_trainer import parse_args
from collections import OrderedDict
from pathlib import Path
PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))
def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.config()


def prepare_config(opt, checkToSkip=True,train=True ):
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.backends.cudnn.deterministic = True
    rootpath = opt.rootpath
    trainCollection = opt.trainCollection

    valCollection = opt.valCollection
    if "task3_caption" in opt:
        task3_caption_suffix = opt.task3_caption
    else:
        task3_caption_suffix='no_task3_caption'
    if opt.val_set == 'no':
        val_set = ''
    else:
        val_set = opt.val_set

    # cuda number
    global device
    if torch.cuda.is_available() and opt.device != "cpu":
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(opt.random_seed)
    else:
        device = torch.device('cpu')

    config = load_config('configs.%s' % opt.config_name)  # 模型参数文件

    model_name = config.model_name

    # set the config parm you adjust
    if opt.parm_adjust_config != 'None':
        config.adjust_parm(opt.parm_adjust_config)
    model_path = os.path.join(rootpath, trainCollection, 'w2vvpp_train', valCollection, val_set, opt.config_name,
                              opt.model_prefix)
    if checkToSkip :
        if util.checkToSkip(os.path.join(model_path, 'model_best.pth.tar'), opt.overwrite):
            sys.exit(0)
        if not  os.path.exists(model_path):
            util.makedirs(model_path)

        print(json.dumps(vars(opt), indent=2))

    global writer
    writer = SummaryWriter(log_dir=model_path, flush_secs=5)
    config.cross=False
    if not train:
        config.float16 = True
        return {'config': config}
    collections = {'train': trainCollection, 'val': valCollection}  # 数据集
    capfiles = {'train': '%s.caption.txt', 'val': os.path.join(val_set, '%s.caption.txt')}

    if train==False:
        collections={}
        capfiles_negationset=None

    # 标注文件
    cap_file_paths = {x: os.path.join(rootpath, collections[x], 'TextData', capfiles[x] % collections[x]) for x in
                      collections}
    # ***************************萌萌哒*****************************
    # 视频 Feature 文件
    vis_feat_files = {x: None for x in collections}
    if len(config.vid_feats) > 0:
        vis_feat_files = {collection: {y: BigFile(os.path.join(rootpath, collections[collection], 'FeatureData', y))
                                       for y in config.vid_feats} for collection in collections}
        # config.vis_fc_layers = list(map(int, config.vis_fc_layers.split('-')))
        config.vis_fc_layers[0] = {}
        for each in vis_feat_files['train'].keys():
            config.vis_fc_layers[0][each] = vis_feat_files['train'][each].ndims
        if config.vis_feat_add_concat:
            feat_dim_sum = np.sum(list(config.vis_fc_layers[0].values()))
            config.vis_fc_layers[0]['vis_feat_add_concat'] = feat_dim_sum

    # 视频 muti_feat 文件 （Faster-rnn 特征）
    vis_muti_feat_dicts = {x: None for x in collections}
      # 视频帧特征文件
    vis_frame_feat_dicts = {x: None for x in collections}
    if config.frame_feat_with_vid_feats:
        vis_frame_feat_dicts = {
            collection: {y: BigFile(os.path.join(rootpath, collections[collection], 'FrameFeatureData', y))
                         for y in config.vid_frame_feats} for collection in collections}
        for each in vis_frame_feat_dicts['train'].keys():  # 增加相关维度信息
            config.vis_fc_layers[0][each] = vis_frame_feat_dicts['train'][each].ndims
    # 视频帧文件
    if config.frame_loader:
        frame_id_path_file = {'train': os.path.join(rootpath, trainCollection, 'id.imagepath.txt'),
                              'val': os.path.join(rootpath, valCollection, 'id.imagepath.txt')
                              }
    else:
        frame_id_path_file = {'train': None,
                              'val': None
                              }

    # ***************************萌萌哒*****************************


    origin_vis_feat_files={x: None for x in collections}
    if task3_caption_suffix == 'no_task3_caption':
        config.task3 = False
        cap_file_paths_task3 = {x: None for x in collections}
    else:
        config.task3=True


        capfiles_task3 = {'train': '%s.caption.%s.txt' % ('%s', task3_caption_suffix),
                          'val': os.path.join(val_set, '%s.caption.%s.txt' % ('%s', task3_caption_suffix))}
        cap_file_paths_task3 = {
        x: os.path.join(rootpath, collections[x], 'TextData', capfiles_task3[x] % collections[x])
        for x in collections}

    startepoch=0
    model = get_model(model_name, device, config)

    if __name__ != '__main__' and torch.cuda.device_count() > 1 and opt.device!='cpu' :
        print("GPU 大于1")
        # 1） 初始化
        torch.distributed.init_process_group(backend='nccl')
        # 2） 配置每个进程的gpu
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        config.local_rank=local_rank
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model = model.module
    else:
        model = model.to(device)

    model_params = sum(p.numel() for p in model.parameters())
    print('params: %.2fM' % (model_params / 1000000.0))

    prepared_configs = {'vis_feat_files': vis_feat_files,
                        'origin_vis_feat_files': origin_vis_feat_files,
                        'vis_muti_feat_dicts': vis_muti_feat_dicts,
                        'vis_frame_feat_dicts': vis_frame_feat_dicts,
                        'frame_id_path_file': frame_id_path_file,
                        'cap_file_paths': cap_file_paths,
                        'cap_file_paths_task2': None,
                        'cap_file_paths_task3': cap_file_paths_task3,
                        'opt': opt,
                        'val_set': val_set,
                        'config': config,
                        'collections': collections,
                        'model_path': model_path,
                        'device': device,
                        'task3_caption_suffix': task3_caption_suffix,
                        'model': model,
                        "startepoch":startepoch
                        }
    return prepared_configs





def main(opt):
    prepared_configs = prepare_config(opt)
    vis_feat_files = prepared_configs['vis_feat_files']
    origin_vis_feat_files=prepared_configs['origin_vis_feat_files']
    vis_frame_feat_dicts = prepared_configs['vis_frame_feat_dicts']
    frame_id_path_file = prepared_configs['frame_id_path_file']
    vis_muti_feat_dicts = prepared_configs['vis_muti_feat_dicts']
    cap_file_paths = prepared_configs['cap_file_paths']
    cap_file_paths_task2 = prepared_configs['cap_file_paths_task2']
    cap_file_paths_task3= prepared_configs['cap_file_paths_task3']
    opt = prepared_configs['opt']
    config = prepared_configs['config']
    collections = prepared_configs['collections']
    model_path = prepared_configs['model_path']
    model = prepared_configs['model']
    device = prepared_configs['device']
    val_set = prepared_configs['val_set']
    vis_ids = list(
        map(str.strip, open(os.path.join(opt.rootpath, opt.trainCollection, 'VideoSets', opt.trainCollection + '.txt'))))
    params_parirloader = {x: {'vis_feat_files': vis_feat_files[x], 'capfile': cap_file_paths_task3[x],
                              'vis_frame_feat_dicts': vis_frame_feat_dicts[x],
                              'max_frame': config.max_frame, 'vis_ids': vis_ids,
                              'sample_type': config.frame_sample_type_train,
                              'vis_muti_feat_dicts': vis_muti_feat_dicts[x],
                              'frame_id_path_file': frame_id_path_file[x], 'pin_memory': False,
                              'batch_size': opt.batch_size, 'num_workers': opt.workers,
                              'config': config,
                              'collection': x,
                              'shuffle': (x == 'train'), 'task3': config.task3,'sampler':None,
       'clip_vocab_size':config.clip_opt['vocab_size'],"max_txtlength":config.max_txtlength
                              } for x in ['train']}

    if __name__ != '__main__' and torch.cuda.device_count() > 1:
        params_parirloader["train"]['sampler'] = 'NotNone'
        params_parirloader["train"]['shuffle'] = False
    data_loaders = {x: data.pair_provider_withneg(params_parirloader[x])
                for x in  ['train']}


    vis_ids = list(map(str.strip, open(os.path.join(opt.rootpath, opt.valCollection, 'VideoSets', opt.valCollection + '.txt'))))
    vis_loader_val = data.vis_provider({'vis_feat_files': vis_feat_files['val'], 'vis_ids': vis_ids, 'pin_memory': False,
                                    'vis_frame_feat_dicts': vis_frame_feat_dicts['val'],
                                    'max_frame': config.max_frame,
                                        'sample_type': config.frame_sample_type_test,
                                        'frame_id_path_file': frame_id_path_file['val'],
                                    'batch_size': int(opt.batch_size * 2),
                                        'config': config,
                                        'num_workers': opt.workers})
    capfile = os.path.join(opt.rootpath, opt.valCollection, 'TextData', val_set, opt.valCollection+'.caption.txt')
    txt_loader_val = data.txt_provider({'capfile': capfile, 'pin_memory': False, 'config': config,
                                    'batch_size': opt.batch_size, 'task3': config.task3,
                                    'capfile_task2': False, "max_txtlength": config.max_txtlength})

    txt_loader_concept_val=None
    # Train the Model
    best_perf = 0
    no_impr_counter = 0
    val_perf_hist_fout = open(os.path.join(model_path, 'val_perf_hist.txt'), 'w')
    for epoch in range(opt.num_epochs):
        logger.info(json.dumps(vars(opt), indent=2))
        print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, model.learning_rate))
        print('-' * 10)

        writer.add_scalar('train/learning_rate', model.learning_rate[0], epoch)

        if epoch > 0 and hasattr(model, 'change_raw_global_emb_weight'):
            model.change_raw_global_emb_weight()
        # train for one epoch
        train(model, data_loaders['train'], epoch)
        if config.lr_warmup:
            if model.task3_neg_retrival_weight<config.task3_upper_lr:
                model.task3_neg_retrival_weight+=0.0005
        # additional training data
        if 'train2' in data_loaders:
            train(model, data_loaders['train2'], epoch)
        # evaluate on validation set

        cur_perf = validate(model ,txt_loader_val, vis_loader_val, epoch, measure=config.measure, metric=opt.metric,
                            config=config)

        model.lr_step(val_value=cur_perf)

        print(' * Current perf: {}\n * Best perf: {}\n'.format(cur_perf, best_perf))
        val_perf_hist_fout.write('epoch_%d:\nText2Video(%s): %f\n' % (epoch, opt.metric, cur_perf))
        val_perf_hist_fout.flush()

        # remember best performance and save checkpoint
        is_best = cur_perf > best_perf
        best_perf = max(cur_perf, best_perf)
        save_checkpoint({'epoch': epoch + 1, 'model': model.state_dict(), 'best_perf': best_perf,
                         'config': config, 'opt': opt,'optimizer':model.optimizer.state_dict()}, is_best, logdir=model_path, only_best=False,
                        filename='checkpoint_epoch_%s.pth.tar' % epoch)

        if is_best:
            no_impr_counter = 0

        no_impr_counter += 1
        if no_impr_counter > 2 or epoch == opt.num_epochs-1:

            save_checkpoint({'epoch': epoch + 1, 'model': model.state_dict(), 'best_perf': best_perf,
                             'config': config, 'opt': opt,'optimizer':model.optimizer.state_dict()}, is_best=False, logdir=model_path, only_best=True,
                            filename='checkpoint_epoch_%s.pth.tar' % epoch)

            print('Early stopping happended or stopped.\n')
            print(json.dumps(vars(opt), indent=2))
            break
        # 测试状态下早停
        if __name__ == '__main__' and epoch > 11:
            break

    val_perf_hist_fout.close()
    message = 'best performance on validation:\n Text to video({}): {}'.format(opt.metric, best_perf)
    print(message)
    with open(os.path.join(model_path, 'val_perf.txt'), 'w') as fout:
        fout.write(message)
    # if torch.cuda.device_count() > 1:
    #     dist.destroy_process_group()



def train(model, train_loader, epoch):
    # average meters to record the training statistics
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()

    # switch to train mode
    model.train()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()

    for i, train_data in enumerate(train_loader):
        if __name__ == '__main__':
            pass
            if i > 5:
                break
                # sys.exit(0)
        data_time.update(time.time() - end)

        loss_items = model(train_data,epoch)

        values = [('batch_time', batch_time.val)]
        # print(loss_items)
        # print(torch.cuda.is_available())
        for key in loss_items.keys():
            if isinstance(loss_items[key], torch.Tensor):
                loss_items[key] = round(loss_items[key].item(), 4)
            values.append((key, loss_items[key]))
        progbar.add(len(list(train_data["vis_ids"])), values=values)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        writer.add_scalar('train/Loss', sum(list(loss_items.values())), model.iters)
        for key in loss_items.keys():
            writer.add_scalar('train/'+key, loss_items[key], model.iters)
    print()


def validate(model, txt_loader, vis_loader, epoch, measure='cosine', metric='mir', config=None):
    # compute the encoding for all the validation videos and captions
    # vis_embs: 200*2048,  txt_embs: 200*2048, vis_ids: 200, txt_ids: 200

    if __name__ == '__main__':
        debug=True
    else:
        debug=False
    txt2vis_sim, txt_ids, vis_ids,labels = model.predict_multi(txt_loader, vis_loader, config.measure,debug=debug)
    inds = np.argsort(txt2vis_sim, axis=1)
    label_matrix = np.zeros(inds.shape)  #

    for index in range(inds.shape[0]):
        ind = inds[index][::-1]
        # print(txt_ids[index])
        gt_index = np.in1d(np.array(vis_ids)[ind], labels[index])
        label_matrix[index][gt_index] = 1

    (r1, r5, r10, medr, meanr, mir, mAP,_,_,r5s,r10s,ranks) = evaluation.eval(label_matrix)
    write_metric(r1, r5, r10, medr, meanr, mir, mAP, epoch)


    return locals().get(metric, mir)


def write_metric(r1, r5, r10, medr, meanr, mir, mAP, epoch, mode="task1"):
    sum_recall = r1 + r5 + r10
    print(" * Text to video:")
    print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
    print(" * medr, meanr, mir: {}".format([round(medr, 3), round(meanr, 3), round(mir, 3)]))
    print(" * mAP: {}".format(round(mAP, 3)))
    print(" * " + '-' * 10)
    writer.add_scalar(mode + 'val/r1', r1, epoch)
    writer.add_scalar(mode + 'val/r5', r5, epoch)
    writer.add_scalar(mode + 'val/r10', r10, epoch)
    writer.add_scalar(mode + 'val/medr', medr, epoch)
    writer.add_scalar(mode + 'val/meanr', meanr, epoch)
    writer.add_scalar(mode + 'val/mir', mir, epoch)
    writer.add_scalar(mode + 'val/mAP', mAP, epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', only_best=False, logdir=''):
    """

    :param state:
    :param is_best: 比以前的好，就保存下来
    :param filename:
    :param only_best: 当结束训练时，only_best=True, 删除 checkpoint.pth.tar 文件，把 model_temp_best.pth.tar 文件 复制成 model_best.pth.tar
    :param logdir:
    :return:
    """
    resfile = os.path.join(logdir, filename)
    torch.save(state, resfile)
    if is_best:
        shutil.copyfile(resfile, os.path.join(logdir, 'model_temp_best.pth.tar'))


    if only_best:
        if os.path.exists(os.path.join(logdir, 'model_temp_best.pth.tar')):
            shutil.copyfile(os.path.join(logdir, 'model_temp_best.pth.tar'), os.path.join(logdir, 'model_best.pth.tar'))
            os.remove(os.path.join(logdir, 'model_temp_best.pth.tar'))




if __name__ == '__main__':

    if len(sys.argv) == 1:
        print()

        sys.argv = "trainer.py --device 1 msrvtt10ktrain msrvtt10kval " \
                   "--rootpath /home/wzy/VisualSearch --batch_size 32 " \
                   "--train_strategy usual " \
                   "--config_name CLIP.CLIPEnd2EndNegnomask " \
                   "--parm_adjust_config 1_0.001_0.1_0.3_100_0.1_0.6_0.001 " \
                   "--val_set no " \
                   "--save_mean_last 0 " \
                   "--pretrained_file_path None " \
                   "--model_prefix runs_9_ --overwrite 1 " \
                    "--task3_caption mask".split(' ')

    opt = parse_args()  # 这里opt是输入的值，config 才是参数文件中读取的

    main(opt)