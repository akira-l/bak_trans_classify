from __future__ import print_function

import os
import sys
import random
import numpy as np
import pickle
import copy

import cv2
from PIL import ImageFont, ImageDraw, Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.utils.data as Data
import torchvision.utils as tushow
import torchvision.transforms as transforms
from torch.autograd import Variable

import pdb

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_ = transforms.Compose([
    transforms.ToTensor()
])

class ImgDataset(Data.Dataset):
    def __init__(self, state):
        if state not in ('train', 'val', 'test'):
            raise ValueError('{} invalid'.format(state))
        self.state = state
        data_path = './../git/bottom-up-attention/vg_save_nms2'
        self.data_path = data_path
        print('    >>>> load roidb')
        roidb_path = os.path.join(data_path, 'roidb.pkl')
        roidb_file = open(roidb_path, 'rb')
        self.roidb = pickle.load(roidb_file, encoding='bytes')
        roidb_file.close()
        #self.gt_load_name = 'gt_data-cls400-rela100-400-tri399-img80-1421'
        self.gt_load_name = 'gt_data-cls400-rela90-hardsample'
        gt_info = torch.load('./'+self.gt_load_name+'.pt')
        #self.gt_name = 'gt_data-cls400-rela100-400-tri399-img80-1421-addcls'
        self.gt_name = 'gt_data-cls400-rela90-hardsample'
        self.gt_dict, self.cls_cluster = gt_info
        self.cls_cluster = np.array(self.cls_cluster)
        self.img_data_dir = './data/vg/VG_100K_total'

        ####
        self.name_list, self.rawcls_list, self.cls_list, self.box_list, self.rela_list = self.pretrained_data(self.roidb, self.gt_dict, self.cls_cluster)
        self.check_box(self.name_list, self.box_list, self.rawcls_list, self.rela_list)

        train_split = list(range(len(self.rela_list)))
        test_num = 500
        test_split = random.sample(train_split, test_num)
        map(lambda x: train_split.remove(x), test_split)

        if self.state == 'train':
            self.index_list = train_split
        if self.state == 'test':
            self.index_list = test_split

        self.transform = transform_

    def check_box(self, name_list, box_list, cls_list, rela_list):
        vocab_path = './../faster-rcnn.pytorch/data/genome2/1600-400-500'
        obj_vocab_file = open(os.path.join(vocab_path, 'objects_vocab.txt'))
        obj_vocab = obj_vocab_file.readlines()
        rela_file = open(os.path.join(vocab_path, 'relations_vocab.txt'))
        rela_vocab = rela_file.readlines()

        for check_number in range(20):
            test_num = random.randint(0, len(box_list))
            img_name = os.path.join(self.img_data_dir, name_list[test_num])
            box = box_list[test_num]
            cls = cls_list[test_num]
            rela = rela_list[test_num]
            img = cv2.imread(img_name)

            bottomLeftCornerOfText = (30,30)
            fontScale              = 0.8
            fontColor              = (255,0,0)
            lineType               = 2
            font                   = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(img, '-'.join([obj_vocab[cls[0]][:-1], rela_vocab[rela][:-1], obj_vocab[cls[1]][:-1]]),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

            for box_item in box:
                cv2.rectangle(img, (box_item[0], box_item[1]), (box_item[2], box_item[3]), (0, 255, 0), 2)
            cv2.imwrite('test_box'+str(check_number)+'.png', img)
        pdb.set_trace()

    def pretrained_data(self, roidb, gt_dict, cls_cluster):

        reserve_name_list = []
        reserve_cls_list = []
        reserve_rawcls_list = []
        reserve_box_list = []
        reserve_rela_list = []
        for roidb_piece in roidb:
            rela = copy.deepcopy(roidb_piece[b'gt_relations'])
            rela_raw = copy.deepcopy(roidb_piece[b'gt_relations'])
            cls_ = copy.deepcopy(roidb_piece[b'gt_classes'])
            name_ = copy.deepcopy(roidb_piece[b'image'])
            boxes = copy.deepcopy(roidb_piece[b'boxes'])
            width = copy.deepcopy(roidb_piece[b'width'])
            height = copy.deepcopy(roidb_piece[b'height'])
            if max(width, height) > 500:
                ratio = max(width, height) / 500
                boxes_float = boxes.astype('float64')
                boxes_float *= ratio
                boxes = boxes_float.astype('int16')
            name_ = str(name_)[2:-1].split('/')[-1]
            rela[:, 0] = self.cls_cluster[cls_[rela[:, 0]]-1]
            rela[:, 2] = self.cls_cluster[cls_[rela[:, 2]]-1]
            rela[:, 1] -= 1
            rela_list = rela.tolist()
            rela_raw_list = rela_raw.tolist()
            box_list = boxes.tolist()

            rela_key = ['-'.join([str(x) for x in rela_piece]) for rela_piece in rela_list]
            rela_key = list(set(rela_key))
            for rela_item_counter in range(len(rela_key)):
                rela_item = rela_key[rela_item_counter]
                if rela_item in self.gt_dict:
                    reserve_name_list.append(name_)
                    rawcls_tmp = [cls_[rela_raw_list[rela_item_counter][0]]-1, \
                                cls_[rela_raw_list[rela_item_counter][2]]-1]
                    box_tmp = [box_list[rela_raw_list[rela_item_counter][0]], \
                                box_list[rela_raw_list[rela_item_counter][2]]]
                    reserve_box_list.append(box_tmp)
                    reserve_rawcls_list.append(rawcls_tmp)
                    cls_tmp = [int(rela_item.split('-')[0]),\
                                int(rela_item.split('-')[2])]
                    rela_tmp = int(rela_item.split('-')[1])
                    reserve_cls_list.append(cls_tmp)
                    reserve_rela_list.append(rela_tmp)
        return reserve_name_list, reserve_rawcls_list, reserve_cls_list, reserve_box_list, reserve_rela_list

    def test_getitem(self, idx):
        idx = self.index_list[idx]
        fname = self.name_list[idx].split('/')[-1]
        img_path = os.path.join(self.img_data_dir, fname)
        rawcls = self.rawcls_list[idx]
        c_cls = self.cls_list[idx]
        box = self.box_list[idx]
        rela = self.rela_list[idx]

        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        obj_img = img.crop(box[0])
        sub_img = img.crop(box[1])
        img, obj_img, sub_img = self.random_flip([img, obj_img, sub_img])

        img = self.transform(img)
        obj_img = self.transform(obj_img)
        sub_img = self.transform(sub_img)
        return img, obj_img, sub_img, rawcls, c_cls, rela, box, fname

    def collect_fn(self, batch):
        imgs = [x[0] for x in batch]
        obj_imgs = [x[1] for x in batch]
        sub_imgs = [x[2] for x in batch]
        rawcls = [x[3] for x in batch]
        c_cls = [x[4] for x in batch]
        relas = [x[5] for x in batch]

        return torch.stack(imgs), torch.stack(obj_imgs), torch.stack(sub_imgs), torch.stack(rawcls), torch.stack(c_cls), torch.stack(relas)

    def random_flip(self, img_list):
        # flip interface not implement
        return img_list

    def __len__(self):
        return len(self.index_list)




if __name__ == '__main__':
    loader = ImgDataset('train')
    img, obj_img, sub_img, rawcls, c_cls, rela, box, fname = loader.test_getitem(random.randint(0, 500))
    img_grid = tushow.make_grid(img)
    tushow.save_image(img_grid, 'img_grid.png')
    obj_grid = tushow.make_grid(obj_img)
    tushow.save_image(obj_grid, 'obj_grid.png')
    sub_grid = tushow.make_grid(sub_img)
    tushow.save_image(sub_grid, 'sub_grid.png')
    pdb.set_trace()
