# sea_avs_adjustVisTxt
from .. import base_config as BaseConfig
import numpy as np


class config(BaseConfig.config):
    model_name = 'End2EndClip_withNeg'  # 选择的模型，见 model.py
    # visual Attention
    dropout = 0.2
    activation = 'tanh'
    vid_feats = []
    # txt encoder and transform
    text_encoding = {
        'bow_encoding': {'name': 'nobow_nsw'},  # [nobow_nsw, bow_nsw]
        'w2v_encoding': {'name': 'now2v_nsw'},  # [now2v_nsw, w2v_nsw]
        'rnn_encoding': {'name': 'nogru_mean'},  # [gru_mean, bigru_mean, nogru_mean]
        'bert_encoding': {'name': 'noBert',  # [noBert, bert-base-uncased, \bert_name, ...]
                          'dir_name': 'bert-base-uncased'
                          },
        'CLIP_encoding': {'name': 'ViT-B/32',  # [noCLIP, ViT-B/32, \CLIP_name, ...]
                          'dir_name': 'CLIP_ViT-B32'
                          },
        'NetVLAD_encoding': {'name': 'noNetVLAD'},  # [noNetVLAD, NetVLAD]
    }
#

    float16 = True
    # end2end 学习，输入 frame/video 原始文件
    frame_loader = True
    # if text_encoding includes CLIP
    clip_opt = {
        'size': 512, 'transform_batch_norm': True, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': False,'vocab_size':49408
    }

    task3_bottommargin=0.1
    task3_uppermargin=0.6
    max_txtlength = 77
    task3_bottommargin_t2t=None
    task3_uppermargin_t2t=None
    lr_warmup=False
    neg_visgrad=False
    sample_frame = 8# 每个视频均匀选 sample_frame 张
    test_sample_frame=8
    no_norm = False
    def adjust_parm(self, value):
        a = []
        for i, each in enumerate(value.split('_')):
            a.append(eval(each))
        self.task3_neg_weight=a[0]
        self.neg_t2v_weight = a[1]
        self.neg_t2t_weight = a[1]
        if a[2] <2:
            self.task3_bottommargin = a[2]
        else:
            self.task3_bottommargin = None
        if len(a) > 3:
            if a[3]<2:
                self.task3_uppermargin = a[3]
            else:
                self.task3_uppermargin = None
        if len(a) > 5:
            if a[5]<2:
                self.task3_bottommargin_t2t=a[5]
            else:
                self.task3_bottommargin_t2t = None
            if a[6]<2:
                self.task3_uppermargin_t2t = a[6]
            else:
                self.task3_uppermargin_t2t = None
        if len(a) > 7:
            self.neg_t2t_weight=a[7]
