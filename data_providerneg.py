# coding=utf-8
import json

import torch
import torch.utils.data as data
from torchvision.datasets import Kinetics400
import numpy as np
import pickle
import os
from bigfile import BigFile
from textlib import TextTool, Vocabulary, negation_augumentation
from torchvision.transforms import Compose, Resize, CenterCrop, TenCrop, Lambda, ToTensor, Normalize, RandomResizedCrop
import PIL
import model.clip as clip
import random
import regex as re


class DataLoaderX(torch.utils.data.DataLoader):
    pass
    # def __iter__(self):
    #     return BackgroundGenerator(super().__iter__())


def generate_sent_masks(source_lengths,maxframe):
    """ Generate sentence masks for encoder hidden states.
        returns enc_masks (Tensor): Tensor of sentence masks of shape (b, max_seq_length),where max_seq_length = max source length """
    max_seq_length =maxframe
    batch_size = len(source_lengths)
    enc_masks = torch.zeros(batch_size, max_seq_length, dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


# 这些是得到 dataloader 列表的后处理
def collate_vision(data):
    max_frame=12
    idxs, vis_ids, vis_origin_frame_tuple,frame_mask = list(zip(*data))
    # 得到多视频特征字典
    vis_feat_dict = {}

    # 视频帧原始数据
    if vis_origin_frame_tuple[0] != None:
        frame_mask = torch.stack([each for each in frame_mask], 0)

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行

    output_dict = {
        'vis_feat_dict': None,'vis_frame_feat_dict':None,
        'idxs': idxs, 'vis_ids': vis_ids,
        'vis_origin_frame_tuple': vis_origin_frame_tuple,"frame_mask":frame_mask
    }
    return output_dict


def collate_text(data):
    # data.sort(key=lambda x: len(TextTool.tokenize(x[0]['caption'])), reverse=True)
    caption_dict_tuples, idxs, cap_ids,video_ids,negcap,neginfomask = list(zip(*data))

    # 得到多特征 caption 字典
    caption_feat_dict = {}
    caption_feat_dict['clipcaption'] = torch.cat([each["clipcaption"] for each in caption_dict_tuples], 0)
    caption_feat_dict["EOS_pos"] =torch.LongTensor( [each["EOS_pos"][0] for each in caption_dict_tuples])
    for name in caption_dict_tuples[0].keys():
        if name not in ['caption', "textmask", "EOS_pos","clipcaption"]:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    return caption_feat_dict, idxs, cap_ids,video_ids

def collate_adhoc_text(data):
    # data.sort(key=lambda x: len(TextTool.tokenize(x[0]['caption'])), reverse=True)
    caption_dict_tuples, idxs, cap_ids,videoids = list(zip(*data))

    # 得到多特征 caption 字典
    caption_feat_dict = {}

    caption_feat_dict = {}
    caption_feat_dict['clipcaption'] = torch.cat([each["clipcaption"] for each in caption_dict_tuples], 0)
    caption_feat_dict["EOS_pos"] =torch.LongTensor( [each["EOS_pos"][0] for each in caption_dict_tuples])
    for name in caption_dict_tuples[0].keys():
        if name not in ['caption', "textmask", "EOS_pos","clipcaption"]:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    return caption_feat_dict, idxs, cap_ids,videoids


def collate_text_withneginfo(data):
    # data.sort(key=lambda x: len(TextTool.tokenize(x[0]['caption'])), reverse=True)
    caption_dict_tuples, idxs, cap_ids,videoids,negcap_token,neginfomask = list(zip(*data))

    # 得到多特征 caption 字典
    neginfomasks=[]
    negcaps=[]
    caption_feat_dict = {}
    caption_feat_dict['clipcaption'] = torch.cat([each["clipcaption"] for each in caption_dict_tuples], 0)
    caption_feat_dict["EOS_pos"] =torch.LongTensor( [each["EOS_pos"][0] for each in caption_dict_tuples])
    for name in caption_dict_tuples[0].keys():
        if name not in ['caption', "textmask", "EOS_pos","clipcaption"]:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)
    for i,each in enumerate(neginfomask):
         if each:
            negcaps.append(negcap_token[i])
         neginfomasks.append(each)
    if len(negcaps)>0:
        negcaps=torch.cat(negcaps, 0)
    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    return caption_feat_dict, idxs, cap_ids,negcaps,neginfomasks,videoids



def collate_text_withneg(data):
    # data.sort(key=lambda x: len(TextTool.tokenize(x[0]['caption'])), reverse=True)
    caption_dict_tuples, idxs, cap_ids, falsecaption_dict_tuples = list(zip(*data))

    # 得到多特征 caption 字典


    caption_feat_dict = {}
    caption_feat_dict['clipcaption'] = torch.cat([each['clipcaption'] for each in caption_dict_tuples], 0)
    caption_feat_dict["EOS_pos"] = torch.LongTensor([each["EOS_pos"][0] for each in caption_dict_tuples])

    for name in caption_dict_tuples[0].keys():
        if name not in ['caption',  "EOS_pos","clipcaption"]:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    falsecaption, mask_task3,falseEOS_pos =  [], [],[],[]

    for i, caption_dict in enumerate(falsecaption_dict_tuples):
        if caption_dict is not None and caption_dict['postive_mask'] > -1:
            mask_task3.append(caption_dict['postive_mask'])
            falsecaption.append(caption_dict['falsesent'])
            falseEOS_pos.append(caption_dict['EOS_pos'])
    mask_task3 = np.array(mask_task3)
    falsecaption = torch.cat(falsecaption,0)
    falseEOS_pos=torch.LongTensor(falseEOS_pos)
    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    output = {"caption": caption_feat_dict, 'idxs': idxs, 'cap_ids': cap_ids,
              'falsecaption': falsecaption,
             "captions_task3_mask": mask_task3,"falseEOS_pos":falseEOS_pos}
    return output



def collate_pair(data):
    max_frame=12
    data.sort(key=lambda x: len(TextTool.tokenize(x[1]['caption'])), reverse=True)
    vis_feat_tuple, caption_dict_tuples, vis_muti_feat, caption_labels_task2, \
    idxs, vis_ids, cap_ids, vis_frame_feat_tuple, caption_labels_task3, mask_task3,frame_mask = list(zip(*data))

    # 视频特征字典
    vis_feat_dict = {}
    for name in vis_feat_tuple[0].keys():
        vis_feat_dict[name] = torch.stack([each[name] for each in vis_feat_tuple], 0)

    # 视频帧特征字典，由于帧数不统一，使用 0 填充，并且输出 mask_tensor 矩阵
    vis_frame_feat_dict = {}
    if vis_frame_feat_tuple[0] != {}:
        # 得到 source_lengths 列表
        name = list(vis_frame_feat_tuple[0].keys())[0]
        source_lengths = [each[name].shape[0] for each in vis_frame_feat_tuple]
        mask_tensor = generate_sent_masks(source_lengths,max_frame)
        vis_frame_feat_dict['mask_tensor'] = mask_tensor
        batch_size, max_length = mask_tensor.shape

        for name in vis_frame_feat_tuple[0].keys():
            vis_frame_feat_dict[name] = torch.zeros(
                batch_size, max_length, vis_frame_feat_tuple[0][name].shape[-1]
            )
            for index, each in enumerate(vis_frame_feat_tuple):
                vis_frame_feat_dict[name][index][0:source_lengths[index]] = each[name]

    if vis_origin_frame_tuple[0] != {}:
        frame_mask = torch.stack([each for each in frame_mask], 0)
    if vis_muti_feat[0] is not None:
        vis_muti_feat = torch.stack(vis_muti_feat, 0)

    # 文本特征字典


    # 文本特征字典
    # 得到多特征 caption 字典
    caption_feat_dict = {}
    caption_feat_dict['clipcaption'] = torch.cat([each["clipcaption"] for each in caption_dict_tuples],0)
    caption_feat_dict['textmask'] = torch.cat([each["textmask"] for each in caption_dict_tuples], 0)
    caption_feat_dict["EOS_pos"] = torch.LongTensor([each["EOS_pos"][0] for each in caption_dict_tuples])
    for name in caption_dict_tuples[0].keys():
        if name not in[ 'caption',"textmask","EOS_pos","clipcaption"]:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    caption_task2_feat_dict = {}
    if caption_labels_task2[0] is not None:
        for name in caption_labels_task2[0].keys():
            if name == 'caption':
                caption_task2_feat_dict[name] = [each[name] for each in caption_labels_task2]
            else:
                caption_task2_feat_dict[name] = torch.stack([each[name] for each in caption_labels_task2], 0)
    caption_task3_feat_dict = {}

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    output = {'vis_feats': vis_feat_dict, 'vis_muti_feat': vis_muti_feat,
              'vis_frame_feat_dict': vis_frame_feat_dict,
              'vis_origin_frame_tuple': vis_origin_frame_tuple,
              'captions': caption_feat_dict, 'captions_task2': caption_labels_task2,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids,
              'captions_task3': caption_task3_feat_dict, "captions_task3_mask": mask_task3,"frame_mask":frame_mask}
    return output


def collate_pair_frame_list(data):
    """
    输出的 视频帧特征 是一个 list
    :param data:
    :return:
    """
    data.sort(key=lambda x: len(TextTool.tokenize(x[1]['caption'])), reverse=True)
    vis_feat_tuple, caption_dict_tuples, vis_muti_feat, caption_labels_task2, \
    idxs, vis_ids, cap_ids, vis_frame_feat_tuple = list(zip(*data))
    # 视频特征字典
    vis_feat_dict = {}
    for name in vis_feat_tuple[0].keys():
        vis_feat_dict[name] = torch.stack([each[name] for each in vis_feat_tuple], 0)

    # 视频帧特征字典，由于帧数不统一，里面是列表
    vis_frame_feat_dict = {}
    if vis_frame_feat_tuple[0] != {}:
        for name in vis_frame_feat_tuple[0].keys():
            vis_frame_feat_dict[name] = [each[name] for each in vis_frame_feat_tuple]

    if vis_muti_feat[0] is not None:
        vis_muti_feat = torch.stack(vis_muti_feat, 0)

    # 文本特征字典
    caption_feat_dict = {}
    for name in caption_dict_tuples[0].keys():
        if name == 'caption':
            caption_feat_dict[name] = [each[name] for each in caption_dict_tuples]
        else:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    output = {'vis_feats': vis_feat_dict, 'vis_muti_feat': vis_muti_feat,
              'vis_frame_feat_dict': vis_frame_feat_dict,
              'captions': caption_feat_dict, 'captions_task2': caption_labels_task2,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids}
    return output


def collate_pair_subset(data):
    data.sort(key=lambda x: len(TextTool.tokenize(x[1])), reverse=True)
    vis_feats, captions, captions_task2, idxs, vis_ids, cap_ids = list(zip(*data))
    vis_feats = torch.stack(vis_feats, 0)
    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    idxs = np.array(idxs) - np.array(idxs).min()
    output = {'vis_feats': vis_feats, 'captions': captions, 'captions_task2': captions_task2,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids}
    return output


def collate_pair_ircsn(data):
    data.sort(key=lambda x: len(TextTool.tokenize(x[1])), reverse=True)
    vis_feats, captions, captions_task2, idxs, vis_ids, cap_ids = list(zip(*data))
    vis_feats = torch.stack(vis_feats, 0)
    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    output = {'vis_feats': vis_feats, 'captions': captions, 'captions_task2': captions_task2,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids}
    return output


def collate_pair_withneg(data):
    # data.sort(key=lambda x: len(TextTool.tokenize(x[1])), reverse=True)
    caption_dict_tuples, idxs, vis_ids, cap_ids, falsecaption, mask_task3, vis_origin_frame_tuple,frame_mask = list(zip(*data))



    # 文本特征字典
    # 得到多特征 caption 字典
    caption_feat_dict = {}
    caption_feat_dict['clipcaption'] = torch.cat([each["clipcaption"] for each in caption_dict_tuples],0)
    caption_feat_dict["EOS_pos"]= [each["EOS_pos"][0] for each in caption_dict_tuples]
    for name in caption_dict_tuples[0].keys():
        if name not in[ 'caption',"EOS_pos","clipcaption"]:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    mask_task3 = np.array((mask_task3))
    index_task3 = np.where(mask_task3 > -1)[0]
    falsecaption = list(falsecaption)

    falsecaption = [falsecaption[i] for i in index_task3]
    falsecaption_feat_dict={}
    falsecaption_feat_dict['clipcaption'] = torch.cat([each["clipcaption"] for each in falsecaption],0)
    falsecaption_feat_dict["EOS_pos"]= [each["EOS_pos"][0] for each in falsecaption]


    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    output = {  'captions': caption_feat_dict,
              'vis_origin_frame_tuple': vis_origin_frame_tuple,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids,
              'falsecaption': falsecaption_feat_dict, "captions_task3_mask": mask_task3,"vis_mask":frame_mask}
    return output


class ImageDataset(data.Dataset):
    def __init__(self, id_path_file, max_length=20,oversample=False, sample_frame=8, sample_type='uniform'):
        """
        :param id_path_file: similar to "video5027_200  ImageData/video5027/video5027_200.jpg \n ..."
        :param oversample:
        :param sample_type: ['uniform', 'random', ...]
        # 均匀取 sample_frame 帧，随机选 sample_frame 帧.
        """
        oversample_preprocess = Compose([
            Resize(256),
            RandomResizedCrop(224),  # this is a list of PIL Images
            Lambda(lambda crops: torch.stack(
                [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(ToTensor()(crop)) for crop in crops]))
            # returns a 4D tensor
        ])

        preprocess = Compose([
            Resize(256),
            RandomResizedCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Using the mean and std of the ImageNet dataset.
        ])

        self.sample_frame = sample_frame
        self.sample_type = sample_type
        self.max_length=max_length
        collection_path = os.path.dirname(id_path_file)
        data = list(map(str.strip, open(id_path_file).readlines()))
        self.image_ids = [x.split()[0] for x in data]
        self.file_names = [os.path.join(collection_path, x.split()[1]) for x in data]

        # Get the mapping of video_id to image path
        self.video2Image_path = {}
        for each in data:
            image_id, image_path = each.split()[0], os.path.join(collection_path, each.split()[1])

            video_id = "_".join(image_id.split('_')[:-1])
            if video_id not in self.video2Image_path:
                self.video2Image_path[video_id] = []
            self.video2Image_path[video_id].append(image_path)
        # rank the image_paths

        for video_id in self.video2Image_path:
            try:
                self.video2Image_path[video_id].sort(key=lambda x: int(os.path.basename(x).split('.')[0].split("_")[-1]))
            except ValueError:
                self.video2Image_path[video_id].sort(
                    key=lambda x: os.path.basename(x).split('.')[0].split("_")[-1])

        if oversample:
            self.preprocess = oversample_preprocess
        else:
            self.preprocess = preprocess

        _, self.preprocess_clip = clip.load("ViT-B/32", device="cpu")
        self.preprocess_clip_toTensor = Compose([
            # Resize(256),
            # CenterCrop(224),
            Resize(512),
            CenterCrop(512),
            lambda image: image.convert("RGB"),
            ToTensor(),
        ])
        self.meta = {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
        self.preprocess_clip_fromTensor = Compose([
            Resize(224),
            Normalize(self.meta['mean'], self.meta['std']),
    ])


    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_name = self.file_names[index]
        image = PIL.Image.open(file_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.preprocess(image)

        return image_id, image

    def __len__(self):
        return len(self.image_ids)

    def __get_image_from_videoid(self, video_id):
        images = None  # (image_num, 3, 224, 224)
        image_ids = []

        for each in self.video2Image_path[video_id]:
            image = PIL.Image.open(each)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.preprocess(image)
            image = image.unsqueeze(0)
            if images is None:
                images = image
            else:
                images = torch.cat((images, image), dim=0)
            image_ids.append(os.path.basename(each).split('.')[0])

        return image_ids, images


    def get_image_from_videoid_with_clip(self, video_id):
        images = None  # (image_num, 3, 224, 224)
        image_ids = []
        frame_indexs = []  # The index of chosen frames
        # video_id missing

        if video_id not in self.video2Image_path:
            print(video_id)
            print(video_id, "is missing in id.imagepath.txt file")
            image_ids = ["%s_%d" % (video_id, 0) for each in range(0, self.sample_frame)]
            images = torch.ones((self.sample_frame, 3, 224, 224))

            return image_ids, images

        if self.sample_type == 'uniform' or len(self.video2Image_path[video_id]) <= self.sample_frame:
            frame_indexs = np.linspace(0, len(self.video2Image_path[video_id]) - 1,
                                       self.sample_frame, dtype=int)

        elif self.sample_type == 'random':
            frame_indexs = random.sample(list(np.arange(0, len(self.video2Image_path[video_id]))), self.sample_frame)
            frame_indexs.sort()
        else:
            raise Exception("Sample_type is not implemented!")
        for index in frame_indexs:
            each = self.video2Image_path[video_id][index]
            try:
                image = self.preprocess_clip(PIL.Image.open(each)).unsqueeze(0)  # (1, 3, 224, 224)
            except Exception as e:
                print(e)
                image = torch.ones((1, 3, 224, 224))

            if images is None:
                images = image
            else:
                images = torch.cat((images, image), dim=0)
            image_ids.append(os.path.basename(each).split('.')[0])
        enc_masks = torch.zeros( self.max_length)
        enc_masks[ :len(image_ids)] = 1
        return image_ids, images,enc_masks


class VisionDataset(data.Dataset):
    """
    得到视频的 Dataset
    """

    def __init__(self, params):

        # 帧级别格式: frame_name tensors ...
        if 'vis_frame_feat_dicts' in params:
            if params['vis_frame_feat_dicts'] is not None:
                self.max_frame = params['max_frame']  # 最大出现帧数
                self.multi_frame_feat = True
                self.vis_frame_feat_dict = params['vis_frame_feat_dicts']
                self.visual_id2frame_id_dict = self.__get_visual_id2frame_id_dict__(self.vis_frame_feat_dict)

        # self.vis_ids = self.vis_feat_file.names if params.get('vis_ids', None) is None else params['vis_ids']
        self.vis_ids = params.get('vis_ids', None)

        self.length = len(self.vis_ids)

        # 原始帧数据
        self.frame_loader = False
        if 'config' in params:
            if params['config'].frame_loader:
                self.frame_loader = True
                if 'sample_type' in params:
                    sample_type = params['sample_type']
                else:
                    sample_type = 'uniform'
                self.ImageDataset = ImageDataset(
                    params['frame_id_path_file'],
                    sample_frame=params['config'].sample_frame,
                    sample_type=sample_type,max_length=params["max_frame"]
                )

    def __get_visual_id2frame_id_dict__(self, vis_frame_feat_dict):
        visual_id2frame_id_dict = {}
        for each in vis_frame_feat_dict:
            frameid_list = vis_frame_feat_dict[each].names
            visual_id2frame_id_dict[each] = {}
            # 得到 videoid 对应的 frame id
            for frame_id in frameid_list:
                video_id = frame_id.split('_')[0]
                if video_id not in visual_id2frame_id_dict[each]:
                    visual_id2frame_id_dict[each][video_id] = []
                visual_id2frame_id_dict[each][video_id].append(frame_id)

        # rank the frame_id
        for each_name in visual_id2frame_id_dict:
            for each_video_id in visual_id2frame_id_dict[each_name]:
                visual_id2frame_id_dict[each_name][each_video_id].sort(key=lambda x: int(x.split("_")[-1]))
        return visual_id2frame_id_dict

    def __getitem__(self, index):
        vis_id = self.vis_ids[index]

        visual_output = self.get_feat_by_id(vis_id)
        vis_origin_frame_tensor = visual_output['vis_origin_frame_tensor']
        frame_mask = visual_output['frame_mask']
        return  index, vis_id, vis_origin_frame_tensor,frame_mask

    def get_feat_by_id(self, vis_id):
        # 视频原始帧信息
        vis_origin_frame_tensor = None
        frame_mask=None
        if self.frame_loader:
            frame_ids, vis_origin_frame_tensor,frame_mask = self.ImageDataset.get_image_from_videoid_with_clip(vis_id)

        vis_output_dict = {
                           'vis_origin_frame_tensor': vis_origin_frame_tensor,
                           "frame_mask":frame_mask
                           }

        return vis_output_dict

    def __len__(self):
        return self.length


class TextDataset(data.Dataset):
    """
    得到 文字的 Dataset, self.get_caption_by_id(cap_id)可以得到第几个 caption.
    """

    def __init__(self, params, task3=False, capfile_task2=False, capfile_task3=False):
        capfile = params['capfile']
        # 读取预先计算特征
        capfile_task2 = params['capfile_task2']
        self.pre_calculate_feat_files = {}
        # try:
        #     if not capfile_task2 and not task3:
        #         self.pre_calculate_feat_files = self.get_precalculate_file(params['config'],
        #                                                                    os.path.dirname(params['capfile']))
        #     else:
        #         self.pre_calculate_feat_files = {}
        # except Exception as e:
        #     print("读取预先计算特征错误 !", e)
        #    self.pre_calculate_feat_files = {}

        if task3 and 'CLIP_encoding' in self.pre_calculate_feat_files:
            self.pre_calculate_feat_files.pop('CLIP_encoding')
        if capfile_task2:
            cap_ids = list(map(lambda x: x.split("#")[0], open(capfile).readlines()))
            capfile = params['capfile_task2']
        elif capfile_task3:
            capfile = params['capfile_task3']

        self.capfile_task3 = capfile_task3
        self.capfile_task2 = capfile_task2

        self.captions = {}
        self.cap_ids = []
        if capfile_task3:
            # mask 0：negtive 1：positive
            self.mask_task3 = {}
            with open(capfile, 'r') as reader:
                lines = reader.readlines()


                for line in lines:
                    cap_idfull, caption = line.strip().split(None, 1)
                    cap_id, cap_id2 = cap_idfull.split('#')
                    cap_id = cap_id + '#' + cap_id2.split("F")[0]
                    if "p" in cap_idfull:
                        self.mask_task3[cap_id] = 1

                    else:
                        self.mask_task3[cap_id] = 0

                    if cap_id not in self.captions:
                        self.captions[cap_id] = [caption]
                        self.cap_ids.append(cap_id)
                    else:
                        self.captions[cap_id].append(caption)
        else:
            with open(capfile, 'r') as reader:
                for line in reader.readlines():
                    if len(line.strip().split(None, 1)) < 2:
                        cap_id = line.strip().split(None, 1)[0]
                        caption = ''
                    else:
                        cap_id, caption = line.strip().split(None, 1)
                    self.captions[cap_id] = caption
                    self.cap_ids.append(cap_id)
        if capfile_task2:
            self.cap_ids = cap_ids
        self.negcaption={}
        self.length = len(self.cap_ids)
        if "neginfo_file"  in params:
            with open(params["neginfo_file"], 'r') as reader:
                lines = reader.readlines()
                for line in lines:
                    capinfo = json.loads(line)
                    cap_id=capinfo["cap_id"]
                    self.negcaption[cap_id]=capinfo["negcap"]


        self.tokenizer=clip.tokenize
        self.context_length=params["max_txtlength"]



    def get_precalculate_file(self, config, TextPath):
        precalculate_feat_files = {}
        for each_encoding_name in config.text_encoding:
            if 'no' in config.text_encoding[each_encoding_name]['name']:
                continue
            each_encoding_dict = config.text_encoding[each_encoding_name]
            if 'dir_name' in each_encoding_dict:
                precalculate_feat_files[each_encoding_name] = BigFile(
                    os.path.join(TextPath, each_encoding_dict['dir_name']))
                print('load pretrained', each_encoding_dict['dir_name'])

        return precalculate_feat_files

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        caption_dict = self.get_caption_dict_by_id(cap_id)
        if cap_id in self.negcaption:

            negcap=self.negcaption[cap_id]
            negcap_token, _, _ = self.tokenizer(negcap,  context_length=self.context_length)

            neginfomask=1
        else:
            negcap_token=np.zeros((1))
            neginfomask = 0
        return caption_dict, index, cap_id,[cap_id.split("#")[0]],negcap_token,neginfomask

    def get_caption_dict_by_id(self, cap_id):
        caption_dict = {}

        pop_list = []
        for each in self.pre_calculate_feat_files:
            try:
                caption_dict[each] = torch.Tensor(self.pre_calculate_feat_files[each].read_one(cap_id))
            except:
                caption_dict[each] = torch.Tensor(
                    self.pre_calculate_feat_files[each].read_one(cap_id.replace("#enc", "")))
            #    print("{}, 读取预先计算特征错误 !".format(each), e)
            #    pop_list.append(each)
        for each in pop_list:
            self.pre_calculate_feat_files.pop(each)

        caption_dict={}
        caption = self.captions[cap_id]
        caption_dict["caption"]=caption
        caption_dict["clipcaption"],caption_dict["textmask"],caption_dict["EOS_pos"]=self.tokenizer(caption,context_length=self.context_length)

        return caption_dict

    def get_falsecaption_by_id(self, cap_id):
        caption_dict = {}
        if cap_id in self.captions:
            caption = self.captions[cap_id]
            caption = random.choice(caption)
            mask = self.mask_task3[cap_id]
            pop_list = []
            for each in self.pre_calculate_feat_files:
                pop_list.append(each)
            for each in pop_list:
                self.pre_calculate_feat_files.pop(each)

            caption_dict["caption"] = caption
        else:
            mask = -1
            caption_dict["caption"] = None

        return caption_dict, mask

    def __len__(self):
        return self.length


class PairDataset(data.Dataset):
    """
    得到 vis_feat, caption, capfile_task2, index, vis_id, cap_id
    """

    def __init__(self, params):
        """

        :param params: params['vis_muti_feat_dicts']: Faster-rcnn 特征
        """
        self.params = params
        self.visData = VisionDataset(params)

        if params['capfile_task2'] is None:
            self.txtData_task2 = None
        else:
            self.txtData_task2 = TextDataset(params)
        if params['capfile_task3'] is None:
            self.txtData_task3 = None
            self.txtData = TextDataset(params)
        else:
            self.txtData = TextDataset(params, task3=True)
            self.txtData_task3 = TextDataset(params, task3=True, capfile_task3=True)
            self.txtData_augmentation = self.get_negation_augumentation(self.txtData.captions,
                                                                        self.txtData_task3.mask_task3)
        self.cap_ids = self.txtData.cap_ids
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        vis_id = self.get_visId_by_capId(cap_id)

        caption_dict = self.txtData.get_caption_dict_by_id(cap_id)  # cap_id: 'video7768#14'
        vis_output_dict = self.visData.get_feat_by_id(vis_id)
        # 原始视频帧
        vis_origin_frame_tensor = vis_output_dict['vis_origin_frame_tensor']
        frame_mask = vis_output_dict['frame_mask']
        if self.txtData_task3 is None:
            caption_labels_task3 = None
            mask_task3 = None
        else:
            caption_labels_task3, mask_task3 = self.txtData_task3.get_falsecaption_by_id(cap_id)

            if mask_task3 == 1:
                caption = random.choice(self.txtData_augmentation[cap_id])
        return  caption_dict,  index, vis_id, cap_id, caption_labels_task3, mask_task3, vis_origin_frame_tensor, frame_mask

    def get_visId_by_capId(self, cap_id):
        vis_id = cap_id.split('#', 1)[0]
        return vis_id

    def get_negation_augumentation(self, captions, negcaptions):
        dataset = {}
        for capid, neginfo in negcaptions.items():
            if neginfo[0]["postive_mask"]:
                dataset[capid] = negation_augumentation(captions[capid])
        return dataset

    def __len__(self):
        return self.length


class NegTextDataset(TextDataset):
    """
    得到 文字的 Dataset, self.get_caption_by_id(cap_id)可以得到第几个 caption.
    """

    def __init__(self, params):
        capfile = params['capfile']

        self.context_length=params['max_txtlength']
        self.mask_sent = {}

        self.falsecaptions = {}
        self.captions = {}
        self.cap_ids = []
        self.pre_calculate_feat_files = {}
        # mask 0：negtive 1：positive
        with open(capfile, 'r') as reader:
            lines = reader.readlines()

            for line in lines:

                capinfo = json.loads(line)
                cap_id = capinfo["id"]
                self.captions[cap_id] = capinfo["truth"]
                if "false" in capinfo.keys():
                    for falseinfo in capinfo["false"]:
                        # 生成的是肯定句，postive_mask为1
                        if "Fp" in falseinfo["id"].split("#")[-1]:
                            postive_mask = 1
                        else:
                            postive_mask = 0
                        falsefeature = falseinfo
                        falsefeature['postive_mask'] = postive_mask

                        if cap_id not in self.falsecaptions:
                            self.falsecaptions[cap_id] = [falsefeature]

                        else:
                            self.falsecaptions[cap_id].append(falsefeature)

                self.cap_ids.append(cap_id)

        self.length = len(self.cap_ids)

        self.tokenizer=clip.tokenize
    def __getitem__(self, index):

        cap_id = self.cap_ids[index]
        caption_dict = self.get_caption_by_id(cap_id)


        falsecaption_dict, mask = self.get_falsecaption_by_id(cap_id)

        return caption_dict, index, cap_id, falsecaption_dict

    def get_falsecaption_by_id(self, cap_id):
        if cap_id in self.falsecaptions:
            caption = self.falsecaptions[cap_id]
            caption = random.choice(caption)
            captiondict={}
            captiondict["clipcaption"], captiondict["textmask"], captiondict["EOS_pos"] = self.tokenizer( caption["falsesent"], context_length=self.context_length)

            mask = caption["postive_mask"]

        else:
            mask = -1
            captiondict = None

        return captiondict, mask

    def get_caption_by_id(self, cap_id):
        caption_dict={}
        caption = self.captions[cap_id]
        caption_dict["clipcaption"],caption_dict["textmask"],caption_dict["EOS_pos"]=self.tokenizer(caption,context_length=self.context_length)

        return caption_dict


class adhocTextDataset(NegTextDataset
                       ):
    """
    得到 文字的 Dataset, self.get_caption_by_id(cap_id)可以得到第几个 caption.
    """

    def __init__(self, params):
        capfile = params['capfile']
        self.neginfo = params['neginfo']
        if self.neginfo:
            self.negcaps = {}
            self.poscaps = {}
        self.context_length=params['max_txtlength']
        self.mask_sent = {}
        self.falsecaptions = {}
        self.captions = {}
        self.cap_ids = []
        self.video_id={}
        self.stastic={}
        # mask 0：negtive 1：positive
        with open(capfile, 'r') as reader:
            lines = reader.readlines()

            for line in lines:
                capinfo = eval(line)
                cap_id = capinfo["cap_id"]
                self.captions[cap_id] = capinfo["caption"]
                if "video_ids" in capinfo.keys():
                    self.video_id[cap_id] = capinfo["video_ids"]
                else:
                    self.video_id[cap_id] =capinfo["cap_id"].split("#")[0]
                self.cap_ids.append(cap_id)
                if self.neginfo:
                    if "negstionfirst" in capinfo.keys():
                        res = re.split(r'and|while', capinfo["caption"])
                        if capinfo["negstionfirst"] == 1:
                            poscap = res[1]
                            negcap = res[0]

                        else:
                            poscap = res[0]
                            negcap = res[1]
                        negcap = re.sub('not|don\'t|doesn\'t', "", negcap)
                        self.negcaps[cap_id] = negcap.strip()
                        self.captions[cap_id] = poscap.strip()
                    elif "negcap" in capinfo:
                            self.negcaps[cap_id] = capinfo["negcap"]
                            self.captions[cap_id] = " , ".join(capinfo["poscap"])
                    else:

                        res = re.split(r" not | n't ", capinfo["caption"])

                        if len(res) > 1:
                            self.negcaps[cap_id] = res[1]
                            self.captions[cap_id] = res[0]

        self.pre_calculate_feat_files = {}
        self.length = len(self.cap_ids)
        self.tokenizer=clip.tokenize
    def __getitem__(self, index):

        cap_id = self.cap_ids[index]
        caption_dict = self.get_caption_by_id(cap_id)
        video_ids=self.video_id[cap_id]
        if not self.neginfo:
            return caption_dict, index, cap_id, video_ids

        else:
            if cap_id in self.negcaps:
                negcap = self.negcaps[cap_id]
                negcap_token, _, _ = self.tokenizer(negcap,  context_length=self.context_length)
                negmask=1
            else:
                negcap_token=np.zeros((1))
                negmask= 0
            return caption_dict, index, cap_id, video_ids,negcap_token,negmask



    def get_precalculate_file(self, config, TextPath):
        precalculate_feat_files = {}
        for each_encoding_name in config.text_encoding:
            if 'no' in config.text_encoding[each_encoding_name]['name']:
                continue
            each_encoding_dict = config.text_encoding[each_encoding_name]
            if 'dir_name' in each_encoding_dict:
                precalculate_feat_files[each_encoding_name] = BigFile(
                    os.path.join(TextPath, each_encoding_dict['dir_name']))
                print('load pretrained', each_encoding_dict['dir_name'])

        return precalculate_feat_files


    def get_caption_by_id(self, cap_id):
        caption_dict={}
        caption = self.captions[cap_id]
        caption_dict["clipcaption"],caption_dict["textmask"],caption_dict["EOS_pos"]=self.tokenizer(caption,context_length=self.context_length)

        return caption_dict


class PairDatasetwithNeg(PairDataset):
    """
    得到 vis_feat, caption, capfile_task2, index, vis_id, cap_id
    """

    def __init__(self, params):
        """

        :param params: params['vis_muti_feat_dicts']: Faster-rcnn 特征
        """
        self.params = params
        self.visData = VisionDataset(params)
        # self.txtData = TextDataset(params,task3=True)
        self.txtData = NegTextDataset(params)

        self.txtData_augmentation = self.get_negation_augumentation(self.txtData.captions, self.txtData.falsecaptions)
        self.cap_ids = self.txtData.cap_ids
        self.length = len(self.cap_ids)
        self.tokenizer=clip.tokenize
    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        vis_id = self.get_visId_by_capId(cap_id)

        # cap_id: 'video7768#14'
        # 多视频特征
        vis_output_dict = self.visData.get_feat_by_id(vis_id)
        # 原始视频帧
        vis_origin_frame_tensor = vis_output_dict['vis_origin_frame_tensor']
        frame_mask = vis_output_dict['frame_mask']

        falsecaption, mask_task3 = self.txtData.get_falsecaption_by_id(cap_id)
        if mask_task3 == 1:
            caption_dict={}
            caption = random.choice(self.txtData_augmentation[cap_id])
            caption_dict["clipcaption"],caption_dict["textmask"],caption_dict["EOS_pos"]=self.tokenizer(caption,context_length=self.txtData.context_length)

        else:
            caption_dict = self.txtData.get_caption_by_id(cap_id)

        if mask_task3 == -1:
            falsecaption=None
        return  caption_dict,index, vis_id, cap_id,  falsecaption, mask_task3, vis_origin_frame_tensor, frame_mask

    def get_negation_augumentation(self, captions, negcaptions):
        dataset = {}
        for capid, neginfo in negcaptions.items():
            if neginfo[0]["postive_mask"]:
                dataset[capid] = negation_augumentation(captions[capid])
        return dataset


class Kinetics(Kinetics400):
    def __init__(
            self,
            root,
            frames_per_clip,
            step_between_clips=1,
            frame_rate=None,
            extensions=("mp4", "avi", "webm"),
            transform=None,
            _precomputed_metadata=None,
            num_workers=4,
            _video_width=0,
            _video_height=0,
            _video_min_dimension=0,
            _audio_samples=0,
            _audio_channels=0,
    ):

        super(Kinetics, self).__init__(
            root,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            extensions,
            transform,
            _precomputed_metadata,
            num_workers,
            _video_width,
            _video_height,
            _video_min_dimension,
            _audio_samples,
            _audio_channels,
        )
        self.videoId_to_indx = {}
        for i, video_path in enumerate(self.metadata['video_paths']):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.videoId_to_indx[video_name] = i

    def __getitem__(self, idx):
        video, _, info, video_idx = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)
        return video, label, video_idx, clip_idx

    def get_input_by_vis_id(self, video_name: str):
        ircsn_input, label, video_idx, clip_idx = self.__getitem__(self.videoId_to_indx[video_name])

        return ircsn_input


class PairDatasetCsn(PairDataset):
    def __init__(self, params):
        super().__init__(params)

        # 加上 ircsn 的 dataset
        if 'num_frames' not in params:
            raise Exception("params has no attribute num_frame")
        if 'video_root' not in params:
            raise Exception("params has no attribute video_root")
        import torchvision
        from vmz.common import log, utils, transforms as T

        transform_test = torchvision.transforms.Compose(
            [
                T.ToTensorVideo(),
                T.Resize((256, 324)),
                T.NormalizeVideo(
                    mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
                ),
                T.CenterCropVideo(224),
            ]
        )
        metadata_save_dir = os.path.join(params['video_root'], "{}fms.pth".format(params['num_frames']))

        if os.path.isfile(metadata_save_dir):
            metadata = torch.load(metadata_save_dir)
        else:
            metadata = None

        _dataset = Kinetics(
            params['video_root'], params['num_frames'], transform=transform_test, _precomputed_metadata=metadata
        )
        if not os.path.isfile(metadata_save_dir):
            utils.save_on_master(
                _dataset.metadata,
                # "{}_{}_{}fms.pth".format(args.dataset, split, args.num_frames),
                metadata_save_dir,
            )
        print("by default we're extracting all clips at given fps with 50percent overlap")
        _dataset.video_clips.compute_clips(
            params['num_frames'], params['num_frames'] // 2, frame_rate=15
        )

        self.ircsnVisData = _dataset

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        vis_id = self.get_visId_by_capId(cap_id)

        caption_dict = self.txtData.get_caption_dict_by_id(cap_id)
        vis_input_ircsn = self.ircsnVisData.get_input_by_vis_id(vis_id)

        # task2
        caption_labels_task2 = self.txtData_task2.get_caption_dict_by_id(vis_id)  # 由于 task2 名词去掉了‘#’，可以使用video_id 来查找

        return vis_input_ircsn, caption_dict, caption_labels_task2, index, vis_id, cap_id


def vis_provider(params):
    data_loader = DataLoaderX(dataset=VisionDataset(params),
                              batch_size=params.get('batch_size', 1),
                              shuffle=params.get('shuffle', False),
                              pin_memory=params.get('pin_memory', False),
                              num_workers=params.get('num_workers', 0),
                              sampler=params.get('sampler', None),
                              collate_fn=collate_vision)
    return data_loader


def txt_provider(params):
    data_loader = DataLoaderX(dataset=TextDataset(params, task3=params.get('task3')),
                              batch_size=params.get('batch_size', 1),
                              shuffle=params.get('shuffle', False),
                              pin_memory=params.get('pin_memory', False),
                              num_workers=params.get('num_workers', 0),
                              sampler=params.get('sampler', None),
                              collate_fn=collate_text)
    return data_loader

def adhoctxt_provider(params):
    data_loader = DataLoaderX(dataset=adhocTextDataset(params),
                              batch_size=params.get('batch_size', 1),
                              shuffle=params.get('shuffle', False),
                              pin_memory=params.get('pin_memory', False),
                              num_workers=params.get('num_workers', 0),
                              sampler=params.get('sampler', None),
                              collate_fn=collate_adhoc_text)
    return data_loader



def adhoctxt_provider_withneginfo(params):
    data_loader = DataLoaderX(dataset=adhocTextDataset(params),
                              batch_size=params.get('batch_size', 1),
                              shuffle=params.get('shuffle', False),
                              pin_memory=params.get('pin_memory', False),
                              num_workers=params.get('num_workers', 0),
                              sampler=params.get('sampler', None),
                              collate_fn=collate_text_withneginfo)
    return data_loader


def pair_provider(params):
    data_loader = DataLoaderX(dataset=PairDataset(params),
                              batch_size=params.get('batch_size', 1),
                              shuffle=params.get('shuffle', False),
                              pin_memory=params.get('pin_memory', False),
                              num_workers=params.get('num_workers', 0),
                              sampler=params.get('sampler', None),
                              collate_fn=collate_pair,
                              )
    return data_loader


def pair_provider_withneg(params):
    dataset=PairDatasetwithNeg(params)
    if params['sampler'] is not None:
        params['sampler'] = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoaderX(dataset=PairDatasetwithNeg(params),
                              batch_size=params.get('batch_size', 1),
                              shuffle=params.get('shuffle', False),
                              pin_memory=params.get('pin_memory', False),
                              num_workers=params.get('num_workers', 0),
                              sampler=params.get('sampler', None),
                              collate_fn=collate_pair_withneg,
                              )
    return data_loader


def pair_provider_csn(params):
    # todo: meger it with pair_provider
    data_loader = torch.utils.data.DataLoader(dataset=PairDatasetCsn(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              sampler=params.get('sampler', None),
                                              collate_fn=collate_pair_ircsn)
    return data_loader


def pair_provider_subset(params, induce):
    subset = torch.utils.data.dataset.Subset(PairDataset(params), induce)
    if params['sampler'] is not None:
        params['sampler'] = torch.utils.data.distributed.DistributedSampler(subset, shuffle=True)
        print(params)
    data_loader = torch.utils.data.DataLoader(subset,
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              sampler=params.get('sampler', None),
                                              collate_fn=collate_pair_subset)
    return data_loader


if __name__ == '__main__':
    import os

    data_path = '/data2/hf/VisualSearch'
    collection = 'tgif-msrvtt10k'
    vid_feat = 'mean_resnext101_resnet152'
    vid_feat_dir = os.path.join(data_path, collection, 'FeatureData', vid_feat)

    vis_loader = vis_provider({'vis_feat_files': vid_feat_dir, 'batch_size': 100, 'num_workers': 2})

    for i, (feat_vecs, idxs, vis_ids) in enumerate(vis_loader):
        print(i, feat_vecs.shape, len(idxs))
        break

    capfile = os.path.join(data_path, collection, 'TextData', '%s.caption.txt' % collection)

    txt_loader = txt_provider({'capfile': capfile, 'batch_size': 100, 'num_workers': 2})

    for i, (captions, idxs, cap_ids) in enumerate(txt_loader):
        print(i, captions, len(cap_ids))
        print([len(cap) for cap in captions])
        break

    capfile_task2 = os.path.join(data_path, collection, 'TextData', '%s.caption.nouns.txt' % collection)
    pair_loader = pair_provider({'vis_feat_files': vid_feat_dir, 'capfile': capfile,
                                 'capfile_task2': capfile_task2, 'batch_size': 100, 'num_workers': 2, 'shuffle': True})
    for i, (vis_feats, captions, captions_task2, idxs, vis_ids, cap_ids) in enumerate(pair_loader):
        print(i, vis_feats.shape, captions[:10], len(cap_ids))
        print("next")
        print(idxs)
        print(vis_ids)
        print(cap_ids)
        print(captions_task2)
        # print [len(cap) for cap in captions]
        break
