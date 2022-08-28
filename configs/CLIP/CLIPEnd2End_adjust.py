# CLIPEnd2End_adjust
from .. import base_config as BaseConfig
import numpy as np


class config(BaseConfig.config):
    model_name = 'Clip'  # 选择的模型，见 model.py
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
                          },
        'NetVLAD_encoding': {'name': 'noNetVLAD'},  # [noNetVLAD, NetVLAD]
    }
#
    # if text_encoding includes bert
    bert_size = 768
    bert_frozen = True
    bert_do_lower_case = True
    bert_transform_batch_norm = True
    bert_transform_dropout = 0
    bert_transform_activation = 'tanh'
    max_txtlength = 77
    float16 = True
    # end2end 学习，输入 frame/video 原始文件
    frame_loader = True
    # if text_encoding includes CLIP
    clip_opt = {
        'size': 512, 'transform_batch_norm': True, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': False,
    }
    sample_frame = 8 # 每个视频均匀选 sample_frame 张
    test_sample_frame=8
    # For Attention params
    def adjust_parm(self, value):
        a = []
        for i, each in enumerate(value.split('_')):
            a.append(eval(each))