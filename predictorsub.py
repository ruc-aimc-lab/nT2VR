# coding=utf-8
import os
import sys
import time
import json
import argparse
import pickle

import numpy as np

import util
import evaluation
import data_providerneg as data
import trainer
from common import *
from trainer import get_model, load_config
from bigfile import BigFile
from predictor import txt2video_write_to_file,eval_matrix,get_predict_file_from_sim,prepare_config,write_to_predict_result_file

def parse_args():
    parser = argparse.ArgumentParser('W2VVPP predictor')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('testCollection', type=str,
                        help='test collection')
    parser.add_argument('model_path', type=str,
                        help='Path to load the model.')
    parser.add_argument('sim_name', type=str,
                        help='sub-folder where computed similarities are saved')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1],
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
    parser.add_argument("--task2_caption", default="no", type=str,
                        help='the suffix of task2 caption.(It looks like "caption.nouns vocab_nouns") Default is nouns')
    parser.add_argument("--sim_path", default=None, type=str,
                        help='whether tp load a similarity matrix')
    parser.add_argument("--config_name", default="no", type=str,
                        help='config')
    parser.add_argument("--adhoc", default=False, type=bool,
                        help='adhoc')
    args = parser.parse_args()
    return args
#
#
# def txt2video_write_to_file(pred_result_file, inds, vis_ids, txt_ids, t2i_matrix, metrics,labels,pkl_saved_file=None,
#                             txt_loader=None, Threshold=10000):
#     if len(vis_ids) >= Threshold:  # 只保存前 1e4 的检索结果
#         TopK = Threshold
#     else:
#         TopK = -1
#     start = time.time()
#     if metrics:
#         aps, r1s, r5s, r10s, ranks=metrics
#     with open(pred_result_file, 'w') as fout:
#         shot_dict = {}  # 写到字典，方便做 demo
#         for index in range(inds.shape[0]):
#             ind = inds[index][::-1][0:TopK]
#
#             fout.write(txt_ids[index] + ' ' + ' '.join([vis_ids[i] + ' %s' % t2i_matrix[index][i]
#                                                         for i in ind]) + '\n')
#             if pkl_saved_file is not None:
#                 shot_dict[txt_ids[index]] = {}
#                 shot_dict[txt_ids[index]]['labels'] =labels[index]
#                 shot_dict[txt_ids[index]]['query'] = \
#                     txt_loader.dataset.get_caption_dict_by_id(txt_ids[index])["caption"]
#                 shot_dict[txt_ids[index]]['rank_list'] = [vis_ids[i] for i in ind]
#                 shot_dict[txt_ids[index]]['sim_value'] = [t2i_matrix[index][i] for i in ind]
#                 if metrics is not None:
#                     shot_dict[txt_ids[index]]['mAP'] = aps[index]
#                     shot_dict[txt_ids[index]]['r1'] = r1s[index]
#                     shot_dict[txt_ids[index]]['r5'] = r5s[index]
#                     shot_dict[txt_ids[index]]['r10'] = r10s[index]
#                     shot_dict[txt_ids[index]]['ranks'] = ranks[index]
#         if pkl_saved_file is not None:
#             with open(pkl_saved_file, 'wb') as f_shot_dict:
#                 pickle.dump(shot_dict, f_shot_dict)
#     print('writing result into file time: %.3f seconds\n' % (time.time() - start))






def get_predict_file_multigt(opt, checkpoint):
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

        config = load_config('configs.%s' % opt.config_name.replace("bool_",""))  # 模型参数文件
        model_name = config.model_name
        epoch = 0
        best_perf = 0

    # if hasattr(config, 't2v_w2v') and hasattr(config.t2v_w2v, 'w2v'):
    #     w2v_feature_file = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m', 'feature.bin')
    #     config.t2v_w2v.w2v.binary_file = w2v_feature_file
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
        if "StrongCLIP" in str(checkpoint['config']):
            try:
                # if 'clip_finetune_8frame_uniform_1103' == checkpoint['config'].text_encoding['CLIP_encoding']['dir_name']:
                if "StrongCLIP" in str(checkpoint['config']):
                    print("load CLIP-FT model")
                    checkpoint1 = torch.load(
                        os.path.join(rootpath, testCollection,
                                     'TextData/clip_finetune_8frame_uniform_1103/model_best.pth.tar'),
                        map_location='cpu')
                    import collections
                    checkpoint1['model'] = collections.OrderedDict(
                        [(k[11:], v) for k, v in checkpoint1['model'].items()])
                    model.txt_net.encoder.CLIP_encoder.load_state_dict(checkpoint1['model'], strict=True)

                    checkpoint['config'].text_encoding['CLIP_encoding']['dir_name'] = ''
            except Exception as e:
                print("load CLIP-FT model failed!!!")
                print(e)

    vis_feat_files = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData', y))
                      for y in config.vid_feats}
    # 视频帧特征文件
    vis_frame_feat_dicts = None
    if config.frame_feat_with_vid_feats:
        vis_frame_feat_dicts = {y: BigFile(os.path.join(rootpath, testCollection, 'FrameFeatureData', y))
                                for y in config.vid_frame_feats}
    vis_ids = list(
            map(str.strip, open(os.path.join(rootpath, testCollection, 'VideoSets', testCollection + '.txt'))))

    # 视频帧文件
    if  hasattr(config, 'frame_loader') and  config.frame_loader:
        frame_id_path_file = os.path.join(rootpath, testCollection, 'id.imagepath.txt')
    else:
        frame_id_path_file = None
        config.frame_loader=False
    if hasattr(config, 'test_sample_frame'):
        config.sample_frame=config.test_sample_frame
    else:
        config.frame_sample_type_test = "uniform"
        config.sample_frame = 8
    print("use", config.sample_frame, 'frame')
    vis_loader = data.vis_provider({'vis_feat_files': vis_feat_files, 'vis_ids': vis_ids, 'pin_memory': False,
                                    'vis_frame_feat_dicts': vis_frame_feat_dicts,"origin_vis_feat_files":None,
                                    'max_frame': config.max_frame,
                                    'sample_type': config.frame_sample_type_test,
                                    'config': config,
                                    'frame_id_path_file': frame_id_path_file,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers})

    for query_set in opt.query_sets.split(','):
        if resume_file != "None":
            output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', query_set, opt.sim_name)
        else:
            output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', query_set, opt.config_name)
        pred_result_file = os.path.join(output_dir,  opt.config_name+"_"+query_set)

        if util.checkToSkip(pred_result_file, opt.overwrite):
            continue
        if not os.path.exists(output_dir):
            util.makedirs(output_dir)

        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
        #neginfo_file=os.path.join(rootpath, testCollection, 'TextData', testCollection+".caption.negationinfo.txt")
        #neginfo_file = os.path.join(rootpath, testCollection, 'TextData', "simple_query.negationinfo.txt")
        textcollection = query_set.split(".")[0]
        # load text data
        txt_loader = data.adhoctxt_provider_withneginfo({'capfile': capfile, 'pin_memory': False, 'config': config,
                                        'batch_size': opt.batch_size, 'num_workers': opt.num_workers,
                                        'capfile_task2': False, "max_txtlength":77, "neginfo":True})
        t2i_matrix, txt_ids, vis_ids, labels = model.predictneg_adhoc(txt_loader, vis_loader, measure=config.measure)


        inds = np.argsort(t2i_matrix, axis=1)
        #
        # if 'simple_query' not in query_set and 'test.txt' not in query_set:
        #
        #     # caption2index 里面是 ('video001#1', caption, 1, [video001, ...])，这样的 caption 到 gt 检索结果的形式，最后是前10个结果。
        #     caption2index = []
        #     label_matrix = np.zeros(inds.shape)  #
        #     for index in range(inds.shape[0]):
        #         ind = inds[index][::-1]
        #         gt_index=np.in1d(np.array(vis_ids)[ind], labels[index])
        #         #gt_index = np.where(np.in1d(np.array(vis_ids)[ind] ,labels[index]))[0]
        #         label_matrix[index][gt_index] = 1
        #         caption2index.append((txt_ids[index], txt_loader.dataset.captions[txt_ids[index]],
        #                               gt_index[0], tuple(np.array(vis_ids)[ind[0:10]])))
        #     # caption2index = sorted(caption2index, key=lambda kv: kv[2], reverse=True)  # 倒序排列
        #     (r1, r5, r10, medr, meanr, mir, mAP, aps,r1s,r5s,r10s,ranks) = evaluation.eval(label_matrix)
        #     sum_recall = r1 + r5 + r10
        #

        #     start = time.time()
        #     # write_concept_to_file(txt_ids,concepts_txt,aps,pkl_saved_file_txt,txt_loader_concept,txt_loader=txt_loader)
        #     # write_concept_to_file(vis_ids, concepts_vis, aps, pkl_saved_file_vis, txt_loader_concept)
        result_file_dir = os.path.dirname(opt.predict_result_file)
        result_file_name = os.path.basename(opt.predict_result_file)
        aps, r1s, r5s, r10s, ranks = None, None, None, None, None
        pkl_saved_file = os.path.join(output_dir, 't2v_eval.pkl')
        txt2video_write_to_file(pred_result_file, inds, vis_ids, txt_ids, t2i_matrix, (aps, r1s, r5s, r10s, ranks),
                                labels, txt_loader=txt_loader,
                                pkl_saved_file=None, Threshold=1000)

        if 'simple_query' not in query_set and 'test.txt' not in query_set and 'tv' not in query_set:
            r1, r5, r10, medr, meanr, mir, mAP = eval_matrix(pred_result_file, labels, pkl_saved_file)
            write_to_predict_result_file(
                os.path.join(result_file_dir, 'TextToVideo', result_file_name), opt.model_path, checkpoint,
                (r1, r5, r10, medr, meanr, mir, mAP), query_set, epoch
            )


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))
    # Load checkpoint
    if '~' in opt.rootpath:
        opt.rootpath = opt.rootpath.replace('~', os.path.expanduser('~'))
    opt.device = opt.device[0]
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
        result_file_dir = os.path.dirname(opt.model_path)
        checkpoint['opt'].result_file_dir = result_file_dir
        checkpoint['opt'].rootpath = opt.rootpath
        checkpoint['opt'].testCollection = opt.testCollection
        config = prepare_config(checkpoint['opt'], False)['config']
        checkpoint['config'] = config
    else:
        checkpoint = 'None'

    # get_predict_file(opt, checkpoint)
    # if config.task2 == True:
    #     get_predict_file_with_task2(opt, checkpoint)
    # elif config.task2 == False:
    #     get_predict_file(opt, checkpoint)
    # else:
    #     raise Exception ('config.task2 must be bool!')
    if  opt.sim_path is None:
        get_predict_file_multigt(opt, checkpoint)
    else:
        get_predict_file_from_sim(opt, checkpoint)
    # get_multi_predict_file(opt, checkpoint)
    # get_vector(opt, checkpoint)


if __name__ == '__main__':
    if len(sys.argv) == 1:

        sys.argv = "predictorsub.py --device 2 msrvtt10ktest " \
                   "/data/wzy/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/CLIP.CLIPEnd2End_adjust/runs_1_0_8_seed_2/model_best.pth.tar sim " \
                   "--rootpath /home/wzy/VisualSearch --batch_size 256 " \
                   "--query_sets msrvtt10ktest.generated.txt " \
                   "--overwrite 1 --task3_caption mask".split(' ')
        sys.argv = "predictorsub.py --device 1 flickr30ktest " \
                   "/data/wzy/VisualSearch/flickr30ktrain/w2vvpp_train/flickr30kval/ICDE.CLIPEnd2End_adjust/runs_1_0_1_0_0_seed_2/model_best.pth.tar sim " \
                   "--rootpath /home/wzy/VisualSearch --batch_size 16 " \
                   "--query_sets flickr30ktest.generated.txt " \
                   "--overwrite 1 --task3_caption mask".split(' ')
        # sys.argv = "predictorsub.py --device 0 msrvtt1kAtest " \
        #            "/data/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPEnd2End_adjust/runs_1_0_8_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 128 " \
        #            "--query_sets msrvtt1kAtest.generatedv3.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        # sys.argv = "predictorsub.py --device 0 vatex_test1k5 " \
        #            "/data/wzy/VisualSearch/vatex_train/w2vvpp_train/vatex_val1k5/CLIP.CLIPEnd2End_adjust/runs_1_0_8_seed_2/model_best.pth.tar sim " \
        #            "--rootpath /home/wzy/VisualSearch --batch_size 128 " \
        #            "--query_sets vatex_test1k5.generatedv3.txt " \
        #            "--overwrite 1 --task3_caption mask".split(' ')
        sys.argv = "predictorsub.py --device 3 msrvtt1kAtest " \
                   "/data/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPEnd2End_adjust/runs_1_0_8_seed_2/model_best.pth.tar sim " \
                   "--rootpath /home/wzy/VisualSearch --batch_size 16 " \
                   "--query_sets msrvtt1kAtest.caption.falseset_withneginfo.txt " \
                   "--overwrite 1 --task3_caption mask".split(' ')
        sys.argv = "predictorsub.py --device 3 msrvtt1kAtest " \
                   "/data/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPpre/runclippre/model_best.pth.tar sim " \
                   "--rootpath /home/wzy/VisualSearch --batch_size 16 " \
                   "--query_sets msrvtt1kAtest.generated.txt " \
                   "--overwrite 1 --task3_caption mask".split(' ')
        # simple_query.txt,msrvtt10ktest.caption.txt,


        # gcc
        # sys.argv = "predictor.py --device 3 gcc11val " \
        #            "/home/~/hf_code/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/w2vvpp_resnext101_resnet152_subspace_AdjustAttention/runs_w2vvpp_attention3_seed_2/model_best.pth.tar " \
        #            "gcc11train/gcc11train_subset/w2vvpp_resnext101_resnet152_subspace_AdjustTxtEncoder " \
        #            "--rootpath /home/~/hf_code/VisualSearch --batch_size 256 " \
        #            "--query_sets msrvtt10ktest.caption.txt " \
        #            "--overwrite 1".split(' ')

    main()
    # main_adjust_weight()
