
import os
import numpy as np
import pickle
import random
import pandas as pd
import time
import argparse
import json
import sys
def parse_args():
    parser = argparse.ArgumentParser('W2VVPP predictor')
    parser.add_argument('testCollection', type=str,
                        help='test collection')
    parser.add_argument('sim_name', type=str,
                        help='sub-folder where computed similarities are saved')
    parser.add_argument('--rootpath', type=str, default='/data1/wzy/',
                        help='path to datasets. (default: %s)'%'/data1/wzy/')
    parser.add_argument("--config_name", default="no", type=str,
                        help='congig')
    parser.add_argument("--model_path", default="no", type=str,
                        help='None')
    parser.add_argument("--predict_result_file", default="no", type=str,
                        help='predict_result_file')
    parser.add_argument('--original_cap', type=str, default='testCollection.captionsubset_neginfo.txt',
                        help='original_cap')
    parser.add_argument("--negated_cap", default="testCollection.caption.falseset_withneginfo.txt", type=str,
                        help='negated_cap')
    args = parser.parse_args()
    return args
def write_to_predict_result_file(
        predict_result_file, config,model_path,
        result_tuple, testCollection, name_str="Text to video"
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

        (r1, r5, r10, medr, mir, mAP) = result_tuple
        tempStr = " * %s:\n" % name_str
        tempStr += " * dr_1_5_10: {}\n".format([round(r1, 3), round(r5, 3), round(r10, 3)])
        tempStr += " * dmedr, dmir: {}\n".format([round(medr, 3), round(mir, 3)])
        tempStr += " * dmAP: {}\n".format(round(mAP, 3))
        tempStr += " * " + '-' * 10
        print(tempStr)

        f.write(str(time.asctime(time.localtime(time.time()))) + '\t')
        f.write(model_path)
        for each in [config, testCollection, round(r1, 3), round(r5, 3), round(r10, 3),
                     round(mir, 3)]:
            f.write(str(each))
            f.write('\t')

        f.write('\n')
    pass

def load_dict(predpath):
    pkl_file = open(predpath, 'rb')
    pred_data = pickle.load(pkl_file)
    pred_data2={}
    for k,v in pred_data.items():
        kk="#".join(k.split("#")[:2])
        pred_data2[kk]=v
    #print((pred_data.keys()))
    return pred_data


def compute_mean_delta(origin_metrics,false_cap_predpath):
    pkl_file = open(false_cap_predpath, 'rb')
    print(false_cap_predpath)
    pred_data = pickle.load(pkl_file)
    aps = np.zeros(len(pred_data))
    r1s = np.zeros(len(pred_data))
    r5s = np.zeros(len(pred_data))
    r10s= np.zeros(len(pred_data))
    ranks= np.zeros(len(pred_data))

    for i, (idx, txt) in enumerate(pred_data.items()):
        try:
            capid = "#".join(idx.split("#")[:-1])
            aps[i] = (origin_metrics[capid]["mAP"] - txt["mAP"])
        except:
            capid = "#".join(idx.split("#"))
            aps[i] = (origin_metrics[capid]["mAP"] - txt["mAP"])
        r1s[i]=origin_metrics[capid]["r1"]-txt["r1"]
        r5s[i]=origin_metrics[capid]["r5"]-txt["r5"]
        r10s[i]=origin_metrics[capid]["r10"]-txt["r10"]
        ranks[i]=1/origin_metrics[capid]["ranks"]-1/txt["ranks"]
    dmAP=aps.mean()
    dr1 = r1s.mean()
    dr5 = r5s.mean()
    dr10=r10s.mean()
    dmeanr=ranks.mean()
    dmedr=np.median(ranks)
    return dr1,dr5, dr10, dmedr, dmeanr, dmAP


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))
    rootpath = opt.rootpath
    model_path=opt.model_path
    testCollection=opt.testCollection
    model_name = os.path.basename(opt.model_path)
    original_cap=opt.original_cap
    negated_cap=opt.negated_cap
    query_set = testCollection + ".caption.falseset.txt"
    # subsetid= set(open(os.path.join(rootpath, testCollection,"TextData/negated_subset.txt")).read().strip().split("\n"))
    false_cap_predpath = os.path.join(rootpath, testCollection, 'SimilarityIndex',
                                      negated_cap, opt.sim_name,'t2v_eval.pkl')
    # false_cap_predpath = os.path.join(rootpath, testCollection, 'SimilarityIndex', testCollection + ".caption.falseset.txt", config,'t2v_res.pkl')

    # false_cap_predpath = os.path.join(rootpath, testCollection, 'SimilarityIndex',
    #                                  testCollection + ".caption.falseset_withneginfo.txt", config, params, 't2v_res.pkl')
    # false_cap_predpath = "/home/wzy/VisualSearch/msrvtt10ktest/SimilarityIndex/msrvtt10ktest.caption.txt/clip4clip/msrvtt10ktest.falseset_msrvtt_retrieval_bs32_seqTransf_crEn.pkl"
    origin_cap_predpath = os.path.join(rootpath, testCollection, 'SimilarityIndex',
                                           original_cap, opt.sim_name,
                                           't2v_eval.pkl')

    # origin_cap_predpath = os.path.join(rootpath, testCollection, 'SimilarityIndex',testCollection + ".caption.txt", config,'t2v_res.pkl')

    # origin_cap_predpath = os.path.join(rootpath, testCollection, 'SimilarityIndex',
    #                                  testCollection + ".captionsubset_neginfo.txt", config, params, 't2v_res.pkl')
    result_file_dir = os.path.dirname(opt.predict_result_file)
    result_file_name = os.path.basename(opt.predict_result_file)
    # origin_cap_predpath="/home/wzy/VisualSearch/msrvtt10ktest/SimilarityIndex/msrvtt10ktest.caption.txt/clip4clip/msrvtt10ktest.msrvtt7k_retrieval_bs32_seqTransf_crEn.pkl"
    origin_metrics = load_dict(origin_cap_predpath)
    # dr1, dr5, dr10, dmedr, dmeanr, dmAP = compute_subset_delta(origin_metrics, false_cap_predpath,subsetid)
    dr1, dr5, dr10, dmedr, dmeanr, dmAP = compute_mean_delta(origin_metrics, false_cap_predpath)
    write_to_predict_result_file(
        os.path.join(result_file_dir, 'TextToVideo', result_file_name), opt.config_name,opt.model_path,
        (dr1, dr5, dr10, dmedr, dmeanr, dmAP), query_set
    )
    os.remove(origin_cap_predpath)
    os.remove(false_cap_predpath)
    # clip2video
    #
    # dirnames = ["msrvtt_data", "vatex_data"]
    # testCollections = ['msrvtt1kAtest', "vatex_test1k5"]
    # for dirname, testCollection in zip(dirnames, testCollections):
    #     query_set = testCollection + ".caption.falseset"
    #     # if testCollection=="vatex_test1k5":
    #     #     params = "runs_7vatexreal_1_0.001_0.1_0.6_100_0.1_0.3_seed_2"
    #     # else:
    #     #     params = "runs_7_1_0.001_0.1_0.6_100_0.1_0.3_seed_2"
    #     result_file_dir = os.path.join("/home/wzy/VisualSearch/CLIP2Video")
    #     false_cap_predpath = os.path.join(rootpath, "CLIP2Video", dirname, testCollection + ".caption.falseset_withneginfo",
    #                                       't2v_res.pkl')
    #     origin_cap_predpath = os.path.join(rootpath, "CLIP2Video", dirname, testCollection + ".captionsubset_neginfo",
    #                                   't2v_res.pkl')
    #     origin_metrics = load_dict(origin_cap_predpath)
    #     dr1, dr5, dr10, dmedr, dmeanr, dmAP = compute_mean_delta(origin_metrics, false_cap_predpath)
    #     write_to_predict_result_file(
    #         os.path.join(result_file_dir,  "result_sotafalseset.txt"), "clip2video","",
    #         (dr1, dr5, dr10, dmedr, dmeanr, dmAP), query_set
    #     )
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv = "predict_compute_delta.py msrvtt1kAtest CLIP.CLIPEnd2EndNegnomask/runs_10_1_0.001_0.1_0.6_100_0.1_0.3_seed_2 " \
                   "--model_path /data1/wzy/VisualSearch/msrvtt1kAtrain/w2vvpp_train/msrvtt1kAval/CLIP.CLIPEnd2EndNegnomask/runs_10_1_0.001_0.1_0.6_100_0.1_0.3_seed_2/checkpoint2.pth.tar " \
                   "--rootpath /data1/wzy/VisualSearch " \
                   "--predict_result_file result_log/result_test.txt " \
                   "--config_name CLIP.CLIPEnd2EndNegnomask".split(' ')
    main()

