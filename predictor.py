# coding=utf-8
import os
import sys
import time
import json
import argparse
import pickle
import random
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import util
import evaluation
import data_providerneg as data
import trainer
from common import *
from trainer import get_model, load_config
from bigfile import BigFile


def parse_args():
    parser = argparse.ArgumentParser('W2VVPP predictor')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('testCollection', type=str,
                        help='test collection')
    parser.add_argument('model_path', type=str,
                        help='Path to load the model.')
    parser.add_argument('sim_name', type=str,
                        help='sub-folder where computed similarities are saved')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt',
                        help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18 and tv19.avs.txt for TRECVID19.')
    parser.add_argument('--predict_result_file', type=str, default='result_log/result_test.txt',
                        help='if dataset=msrvtt10k, print the result to txt_file')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of a predicting mini-batch')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of data loader workers.')
    parser.add_argument("--device", default=0, type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument('--adjust_weight_predict', type=bool, default=False,
                        help='whether adjust the weight')
    parser.add_argument('--task3_caption', type=str, default='no_task3_caption',
                        help='the suffix of task3 caption.(It looks like "caption.false ") Default is false.')
    parser.add_argument("--task2_caption", default="no", type=str, help='the suffix of task2 caption.(It looks like "caption.nouns vocab_nouns") Default is nouns')
    parser.add_argument("--config_name", default="no", type=str,
                        help='config')
    parser.add_argument("--adhoc", default=False, type=bool,
                        help='adhoc')
    parser.add_argument("--sim_path", default=None, type=str,
                        help='whether tp load a similarity matrix')
    args = parser.parse_args()
    return args

def txt2video_write_to_file(pred_result_file, inds, vis_ids, txt_ids, t2i_matrix, metrics, labels, pkl_saved_file=None,
                            txt_loader=None, Threshold=1000):
    start = time.time()
    with open(pred_result_file, 'w') as fout:
        fout.write('query_id'+ ' ' + ' '.join([vis_id for vis_id in vis_ids]) + '\n')
        for index in range(t2i_matrix.shape[0]):
            fout.write(txt_ids[index] + ' ' + ' '.join(['%s' % sim for sim in t2i_matrix[index]]) + '\n')
    print('writing result into file time: %.3f seconds\n' % (time.time() - start))





def vec_write_to_file(output_dir, txt_embs, vis_embs,txt_ids,vis_ids ):

    pred_result_filetxt = os.path.join(output_dir, 'txtvec.pkl')
    pred_result_filevis = os.path.join(output_dir, 'visvec.pkl')
    print("write to",pred_result_filetxt)

    shot_dicttxt = {}  # 写到字典，方便做 demo
    shot_dictvis={}

    for index in range(txt_embs.shape[0]):
        idx=txt_ids[index]
        shot_dicttxt[idx]=txt_embs[index,:]

    with open(pred_result_filetxt, 'wb') as f_shot_dict:
        print(pred_result_filetxt)
        pickle.dump(shot_dicttxt, f_shot_dict)
    for index in range(vis_embs.shape[0]):
        idx = vis_ids[index]
        shot_dictvis[idx]= vis_embs[index, :]
    with open(pred_result_filevis, 'wb') as f_shot_dict2:
        pickle.dump(shot_dictvis, f_shot_dict2)


def write_to_predict_result_file(
        predict_result_file, model_path, checkpoint,
        result_tuple,testCollection,epoch, name_str="Text to video"
                                 ):
    """

    :param predict_result_file:
    :param model_path:
    :param checkpoint:
    :param result_tuple: [(r1, r5, r10, medr, meanr, mir, mAP), ...]
    :return:
    """
    result_file_dir = os.path.dirname(predict_result_file)
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir)
    print(predict_result_file)
    with open(predict_result_file, 'a') as f:

        (r1, r5, r10, medr, meanr, mir, mAP) = result_tuple
        tempStr = " * %s:\n" % name_str
        tempStr += " * r_1_5_10: {}\n".format([round(r1, 3), round(r5, 3), round(r10, 3)])
        tempStr += " * medr, meanr, mir: {}\n".format([round(medr, 3), round(meanr, 3), round(mir, 3)])
        tempStr += " * mAP: {}\n".format(round(mAP, 3))
        tempStr += " * " + '-' * 10
        print(tempStr)

        f.write(str(time.asctime(time.localtime(time.time()))) + '\t')
        for each in [model_path,testCollection, round(r1, 3), round(r5, 3), round(r10, 3),
                     round(medr, 3), round(meanr, 3), round(mir, 3), round(mAP, 3)]:
            f.write(str(each))
            f.write('\t')
        f.write(str(epoch))
        f.write('\t')
        if checkpoint!='None':
            f.write(checkpoint['opt'].parm_adjust_config.replace('_', '\t'))
        f.write('\n')
    pass

def prepare_config(opt, checkToSkip=True ):
    import torch
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.multiprocessing.set_sharing_strategy('file_system')  # 多线程

    if '~' in opt.rootpath:
        opt.rootpath = opt.rootpath.replace('~', os.path.expanduser('~'))
    rootpath = opt.rootpath

    testCollection = opt.testCollection

    task2_caption_suffix = opt.task2_caption  # 提取的标签的文件后缀
    if "task3_caption" in opt:
        task3_caption_suffix = opt.task3_caption
    else:
        task3_caption_suffix='no_task3_caption'
    if opt.model_path != 'None':
        if opt.val_set == 'no':
            val_set = ''
        else:
            val_set = opt.val_set
        trainCollection = opt.trainCollection
    # cuda number
    global device
    if torch.cuda.is_available() and opt.device != "cpu":
        device = torch.device('cuda')

    else:
        device = torch.device('cpu')
    if opt.device!="cpu":
        print(opt.device)
        torch.cuda.set_device(int(opt.device))
        #os.environ['CUDA_VISIBLE_DEVICES'] =
    # set the config parm you adjust
    config = load_config('configs.%s' % opt.config_name)  # 模型参数文件
    if hasattr(opt, 'parm_adjust_config')  and opt.parm_adjust_config != 'None':
        config.adjust_parm(opt.parm_adjust_config)
    model_path = opt.model_path
    print(json.dumps(vars(opt), indent=2))

    model_name = config.model_name

    global writer
    writer = SummaryWriter(log_dir=opt.result_file_dir, flush_secs=5)

    collections = {'test': testCollection}  # 数据集


    # ***************************萌萌哒*****************************
    # 视频 Feature 文件
    vis_feat_files = {x: None for x in collections}
    if len(config.vid_feats) > 0:
        vis_feat_files = {collection: {y: BigFile(os.path.join(rootpath, collections[collection], 'FeatureData', y))
                                       for y in config.vid_feats} for collection in collections}
        # config.vis_fc_layers = list(map(int, config.vis_fc_layers.split('-')))
        config.vis_fc_layers[0] = {}
        for each in vis_feat_files['test'].keys():
            config.vis_fc_layers[0][each] = vis_feat_files['test'][each].ndims
        if config.vis_feat_add_concat:
            feat_dim_sum = np.sum(list(config.vis_fc_layers[0].values()))
            config.vis_fc_layers[0]['vis_feat_add_concat'] = feat_dim_sum

    # 视频 muti_feat 文件 （Faster-rnn 特征）
    vis_muti_feat_dicts = {x: None for x in collections}
    if config.SGRAF:
        vis_muti_feat_paths = {x: os.path.join(rootpath, collections[x], 'VideoMultiLabelFeat', config.muti_feat) for x
                               in
                               collections}
        if os.path.realpath(vis_muti_feat_paths['train']) == os.path.realpath(vis_muti_feat_paths['val']):
            vis_muti_feat_dicts['train'] = vis_muti_feat_dicts['val'] = np.load(vis_muti_feat_paths['train'],
                                                                                allow_pickle=True).item()
        else:
            vis_muti_feat_dicts['train'] = np.load(vis_muti_feat_paths['train'], allow_pickle=True).item()
            vis_muti_feat_dicts['val'] = np.load(vis_muti_feat_paths['val'], allow_pickle=True).item()

    # 视频帧特征文件
    vis_frame_feat_dicts = {x: None for x in collections}
    if hasattr(config,'frame_feat_input') and config.frame_feat_input:
        vis_frame_feat_dicts = {
            collection: {y: BigFile(os.path.join(rootpath, collections[collection], 'FrameFeatureData', y))
                         for y in config.vid_frame_feats} for collection in collections}
        for each in vis_frame_feat_dicts['test'].keys():  # 增加相关维度信息
            config.vis_fc_layers[0][each] = vis_frame_feat_dicts['test'][each].ndims


        # 视频转换参数
        config.vis_fc_layers_task2 = list(map(int, config.vis_fc_layers_task2.split('-')))
        config.vis_fc_layers_task2[0] = config.vis_fc_layers[0]
        config.vis_fc_layers_task2[1] = config.t2v_bow_task2.ndims
    if task3_caption_suffix == 'no_task3_caption':
        config.task3 = False
    else:
        config.task3=True

    prepared_configs = {
        'model_name':model_name,
                        'config': config
                        }
    return prepared_configs

def eval_matrix(filepath,labels,pkl_saved_file):
    lines=open(filepath).read().strip().split("\n")
    vis_ids=lines[0].split(" ")[1:]
    txt_ids=[]
    t2i_matrix=np.zeros((len(lines)-1,len(vis_ids)))
    for num,line in enumerate(lines[1:]):
        txt_id,sims=line.split(" ",1)
        sims=list(map(float,sims.split(" ")))
        txt_ids.append(txt_id)
        t2i_matrix[num,:]=sims
    inds = np.argsort(t2i_matrix, axis=1)
    # caption2index 里面是 ('video001#1', caption, 1, [video001, ...])，这样的 caption 到 gt 检索结果的形式，最后是前10个结果。

    label_matrix = np.zeros(inds.shape)  #
    for index in range(inds.shape[0]):
        ind = inds[index][::-1]

        gt_index = np.in1d(np.array(vis_ids)[ind], labels[index])
        # gt_index = np.where(np.in1d(np.array(vis_ids)[ind] ,labels[index]))[0]
        label_matrix[index][gt_index] = 1
    # caption2index = sorted(caption2index, key=lambda kv: kv[2], reverse=True)  # 倒序排列
    (r1, r5, r10, medr, meanr, mir, mAP, aps, r1s, r5s, r10s, ranks) = evaluation.eval(label_matrix)
    #for compute delta on negated
    shot_dict = {}
    TopK=10
    if pkl_saved_file is not None:
        for index in range(inds.shape[0]):
            ind = inds[index][::-1][0:TopK]
            shot_dict[txt_ids[index]] = {}
            shot_dict[txt_ids[index]]['labels'] = labels[index]
            shot_dict[txt_ids[index]]['rank_list'] = [vis_ids[i] for i in ind]
            shot_dict[txt_ids[index]]['sim_value'] = [t2i_matrix[index][i] for i in ind]
            shot_dict[txt_ids[index]]['mAP'] = aps[index]
            shot_dict[txt_ids[index]]['r1'] = r1s[index]
            shot_dict[txt_ids[index]]['r5'] = r5s[index]
            shot_dict[txt_ids[index]]['r10'] = r10s[index]
            shot_dict[txt_ids[index]]['ranks'] = ranks[index]
        with open(pkl_saved_file, 'wb') as f_shot_dict:
            pickle.dump(shot_dict, f_shot_dict)
        print("save to", pkl_saved_file)
    return r1, r5, r10, medr, meanr, mir, mAP

def get_predict_file_from_sim(opt, checkpoint):
    rootpath = opt.rootpath
    testCollection = opt.testCollection
    for query_set in opt.query_sets.split(','):
        output_dir = os.path.join(opt.rootpath, testCollection, 'SimilarityIndex', query_set, opt.config_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
        gt={}
        labels=[]
        pred_result_file = os.path.join(opt.sim_path, opt.config_name + "_" + query_set)
        #pred_result_file = os.path.join(output_dir, opt.config_name + "_" + query_set)
        lines = open(pred_result_file).read().strip().split("\n")
        txt_ids = []
        if opt.adhoc:
            with open(capfile, 'r') as reader:
                capfilelines = reader.readlines()
                for line in capfilelines:
                    capinfo = eval(line)
                    cap_id = capinfo["cap_id"]
                    gt[cap_id] = capinfo["video_ids"]
        for num, line in enumerate(lines[1:]):
            txt_id, sims = line.split(" ", 1)
            txt_ids.append(txt_id)
            if opt.adhoc:
                labels.append(gt[txt_id])
            else:
                labels.append([txt_id.split('#')[0]])
        if not opt.adhoc:
            pkl_saved_file = os.path.join(output_dir, 't2v_eval.pkl')
        else:
            pkl_saved_file = None
        r1, r5, r10, medr, meanr, mir, mAP=eval_matrix(pred_result_file,labels,pkl_saved_file)
        result_file_dir = os.path.dirname(opt.predict_result_file)
        result_file_name = os.path.basename(opt.predict_result_file)
        write_to_predict_result_file(
            os.path.join(result_file_dir, 'TextToVideo', result_file_name), opt.model_path, checkpoint,
            (r1, r5, r10, medr, meanr, mir, mAP), query_set, 0
        )


def get_predict_file(opt, checkpoint):
    rootpath = opt.rootpath
    testCollection = opt.testCollection
    # cuda number
    device = torch.device("cuda:{}".format(opt.device)
                          if (torch.cuda.is_available() and opt.device != "cpu") else "cpu")

    resume_file = os.path.join(opt.model_path)

    # Load checkpoint
    if checkpoint != 'None':
        epoch = checkpoint['epoch']
        best_perf = checkpoint['best_perf']
        config = checkpoint['config']
        model_name = checkpoint['config'].model_name
    else:

        config = load_config('configs.%s' % opt.config_name)  # 模型参数文件
        model_name = config.model_name
        epoch = 0
        best_perf = 0

    if opt.task3_caption == "no_task3_caption":
        task3 = False
    else:
        task3 = True
    vis_feat_files = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData', y))
                                   for y in config.vid_feats}
    config.vis_fc_layers[0] = {}
    for each in vis_feat_files.keys():
        config.vis_fc_layers[0][each] = vis_feat_files[each].ndims
    if config.vis_feat_add_concat:
        feat_dim_sum = np.sum(list(config.vis_fc_layers[0].values()))
        config.vis_fc_layers[0]['vis_feat_add_concat'] = feat_dim_sum
    # Construct the model
    model = get_model(model_name, device, config)
    model = model.to(device)
    # print(model)
    # calculate the number of parameters
    try:
        vis_net_params = sum(p.numel() for p in model.vis_net.parameters())
        txt_net_params = sum(p.numel() for p in model.txt_net.parameters())
        print('    VisNet params: %.2fM' % (vis_net_params / 1000000.0))
        print('    TxtNet params: %.2fM' % (txt_net_params / 1000000.0))
        print('    Total params: %.2fM' %
              ((vis_net_params + txt_net_params) / 1000000.0))
    except:
        pass

    if checkpoint != 'None':
        model.load_state_dict(checkpoint['model'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {}, best_perf {})"
              .format(resume_file, epoch, best_perf))

    vis_feat_files = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData', y))
                                   for y in config.vid_feats}
    # 视频帧特征文件
    vis_frame_feat_dicts = None
    if config.frame_feat_with_vid_feats:
        vis_frame_feat_dicts = {y: BigFile(os.path.join(rootpath, testCollection, 'FrameFeatureData', y))
                                             for y in config.vid_frame_feats}
    vis_ids = list(map(str.strip, open(os.path.join(rootpath, testCollection, 'VideoSets', testCollection + '.txt'))))
    # 视频帧文件
    if  hasattr(config, 'frame_loader') and  config.frame_loader:
        frame_id_path_file = os.path.join(rootpath, testCollection, 'id.imagepath.txt')
    else:
        frame_id_path_file = None
        config.frame_loader=False
    config.sample_frame=config.test_sample_frame
    vis_loader = data.vis_provider({'vis_feat_files': vis_feat_files, 'vis_ids': vis_ids, 'pin_memory': False,
                                    'vis_frame_feat_dicts': vis_frame_feat_dicts,
                                    'max_frame': config.max_frame,
                                    'sample_type': config.frame_sample_type_test,
                                    'config': config,"origin_vis_feat_files":None,
                                    'frame_id_path_file': frame_id_path_file,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers})

    for query_set in opt.query_sets.split(','):
        if resume_file != "None":
            output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', query_set, opt.sim_name)
        else:
            output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', query_set, opt.config_name)
        pred_result_file = os.path.join(output_dir,  opt.config_name+"_"+query_set)
        pkl_saved_file = os.path.join(output_dir, 't2v_eval.pkl')
        if util.checkToSkip(pred_result_file, opt.overwrite):
            continue
        if not os.path.exists(output_dir):
            util.makedirs(output_dir)


        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
        textcollection=query_set.split(".")[0]
        # load text data
        if not opt.adhoc:
            txt_loader = data.txt_provider({'capfile': capfile, 'pin_memory': False, 'config': config,
                                            'batch_size': opt.batch_size, 'num_workers': opt.num_workers, 'capfile_task2': False, "max_txtlength": 77})
        else:
            txt_loader = data.adhoctxt_provider({'capfile': capfile, 'pin_memory': False, 'config': config,
                                                 'batch_size': opt.batch_size, 'num_workers': opt.num_workers,
                                                 'capfile_task2': False, "max_txtlength": 77, 'neginfo': False})

        result_file_dir = os.path.dirname(opt.predict_result_file)
        result_file_name = os.path.basename(opt.predict_result_file)
        #
        t2i_matrix, txt_ids, vis_ids, labels = model.predict_multi(txt_loader, vis_loader, measure=config.measure)

        inds = np.argsort(t2i_matrix, axis=1)
        aps, r1s, r5s, r10s, ranks = None, None, None, None, None


        txt2video_write_to_file(pred_result_file, inds, vis_ids, txt_ids, t2i_matrix, (aps, r1s, r5s, r10s, ranks),
                                    labels, txt_loader=txt_loader,
                                    pkl_saved_file=None, Threshold=1000)

        r1, r5, r10, medr, meanr, mir, mAP=eval_matrix(pred_result_file,labels,pkl_saved_file)
        write_to_predict_result_file(
            os.path.join(result_file_dir, 'TextToVideo', result_file_name), opt.model_path, checkpoint,
            (r1, r5, r10, medr, meanr, mir, mAP), query_set, epoch
        )







def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))
    # Load checkpoint
    #resume_file="/data4/wzy/VisualSearch/msvdtrain/w2vvpp_train/msrvtt1kAval/w2vvpp_msrvtt1ka/model_best.pth.tar"
    # set the config parm you adjust

    # if checkpoint['opt'].parm_adjust_config != 'None':
    #     checkpoint['config'].adjust_parm(checkpoint['opt'].parm_adjust_config)
    if '~' in opt.rootpath:
        opt.rootpath = opt.rootpath.replace('~', os.path.expanduser('~'))
    opt.device=opt.device[0]
    if opt.model_path != 'None':
        resume_file = os.path.join(opt.model_path)
        # resume_file="/data4/wzy/VisualSearch/msvdtrain/w2vvpp_train/msrvtt1kAval/w2vvpp_msrvtt1ka/model_best.pth.tar"

        if '~' in resume_file:
            resume_file = resume_file.replace('~', os.path.expanduser('~'))
            opt.model_path = resume_file
        if not os.path.exists(resume_file):
            logging.info(resume_file + '\n not exists.')
            sys.exit(0)
        print(resume_file)

        checkpoint = torch.load(resume_file, map_location='cpu')
        # set the config parm you adjust

        # if checkpoint['opt'].parm_adjust_config != 'None':
        #     checkpoint['config'].adjust_parm(checkpoint['opt'].parm_adjust_config)
        checkpoint['opt'].device = opt.device
        checkpoint['opt'].model_path = opt.model_path
        checkpoint['opt'].adhoc = opt.adhoc
        result_file_dir = os.path.dirname( opt.model_path)
        checkpoint['opt'].result_file_dir = result_file_dir
        checkpoint['opt'].rootpath = opt.rootpath
        checkpoint['opt'].testCollection = opt.testCollection
        print(checkpoint['config'].sample_frame)
        config = prepare_config(checkpoint['opt'], False)['config']
        checkpoint['config'] = config

    else:
        checkpoint = 'None'

    get_predict_file(opt, checkpoint)
    #get_multi_predict_file(opt, checkpoint)
    #get_vector(opt, checkpoint)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv = "predictor.py --device 2 msrvtt10ktest " \
                   "/data/wzy/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/CLIP.CLIPEnd2EndNegnomask/runs_7_1_0.001_0.1_0.6_100_0.1_0.3_seed_2/model_best.pth.tar sim " \
                   "--rootpath /home/wzy/VisualSearch --batch_size 32 " \
                   "--query_sets msrvtt10ktest.caption.falseset.txt " \
                   "--overwrite 1 --task3_caption mask".split(' ')

        # sys.argv = "predictor.py --device 2 msrvtt10ktest " \
        #            "/data/wzy/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/CLIP.CLIPEnd2End_adjust/runs_1_0_8_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 256 " \
        #            "--query_sets msrvtt10ktest.caption.falseset.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')

        #simple_query.txt,msrvtt10ktest.caption.txt,msrvtt10ktest.caption.negation.txt
        # sys.argv = "predictor.py --device 0 msrvtt1kAtest " \
        #            "/data/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPEnd2End_adjust/runs_1_0_8_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 128 " \
        #            "--query_sets msrvtt1kAtest.caption.falseset.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        sys.argv = "predictor.py --device 3 msrvtt10ktest " \
                   "/data1/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPEnd2EndNegnomask/runs_7_1_0.001_0.1_0.6_100_0.1_0.3_seed_2/model_best.pth.tar sim " \
                   "--rootpath /home/wzy/VisualSearch --batch_size 256 " \
                   "--query_sets simple_query.txt " \
                   "--overwrite 1 --task3_caption mask".split(' ')
        # sys.argv = "predictor.py --device 1 msrvtt10ktest " \
        #            "/data/wzy/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/CLIP.CLIPEnd2EndNegnomask/runs_7_1_0.001_0.1_0.6_100_0.1_0.3_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 32 " \
        #            "--query_sets simple_query.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        # sys.argv = "predictor.py --device 1 msrvtt1kAtest " \
        #            "/data/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPpre/runclippre/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 32 " \
        #            "--query_sets msrvtt1kAtest.captionsubset.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        # sys.argv = "predictor.py --device 3 vatex_test1k5 " \
        #            "/data/wzy/VisualSearch/vatex_train/w2vvpp_train/vatex_val1k5/CLIP.CLIPEnd2End_adjust/runs_1_0_8_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 128 " \
        #            "--query_sets vatex_test1k5.caption.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        # sys.argv = "predictor.py --device 3 vatex_test1k5 " \
        #            "/data/wzy/VisualSearch/vatex_train/w2vvpp_train/vatex_val1k5/CLIP.CLIPEnd2EndNegnomask/runs_7vatexreal_1_0.001_0.1_0.6_100_0.1_0.3_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 128 " \
        #            "--query_sets vatex_test1k5.caption.falseset.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        # sys.argv = "predictor.py --device 3 msrvtt1kAtest " \
        #            "/data/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPpre/runs_7_1_0.001_0.1_0.6_100_0.1_0.3_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 16 " \
        #            "--query_sets msrvtt1kAtest.caption.falseset.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        # # gcc
        # sys.argv = "predictor.py --device 3 gcc11val " \
        #            "/home/~/hf_code/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/w2vvpp_resnext101_resnet152_subspace_AdjustAttention/runs_w2vvpp_attention3_seed_2/model_best.pth.tar " \
        #            "gcc11train/gcc11train_subset/w2vvpp_resnext101_resnet152_subspace_AdjustTxtEncoder " \
        #            "--rootpath /home/~/hf_code/VisualSearch --batch_size 256 " \
        #            "--query_sets msrvtt10ktest.caption.txt " \
        #            "--overwrite 1".split(' ')

    main()
    # main_adjust_weight()
