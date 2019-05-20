from __future__ import print_function

import os
import sys
import random
import numpy as np
import pickle
import copy

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

import pdb


def reclean(save_num=32, split=10):
    data_path = './../git/bottom-up-attention/vg_save_nms2'
    rela_gt_path = os.path.join(data_path, 'use_rela_gt.pt')
    print('>>>> load relation gt')
    rela_gt_all = torch.load(rela_gt_path)
    split_num= len(rela_gt_all)//split
    rela_gt = rela_gt_all[:-split_num]
    test_rela_gt = rela_gt_all[-split_num:]
    print('>>>> re clean train')
    rela_gt = (torch.stack(rela_gt) > 0).int()
    rela_count = rela_gt.sum(0)
    rela_pos = (rela_count>save_num).nonzero().squeeze(1)
    rela_gt_has = rela_gt[:, rela_pos]
    gt_has_sum = rela_gt_has.sum(1)
    rela_pos_has = gt_has_sum.nonzero().squeeze(1)
    print('>>>> re clean test')
    test_rela_gt = (torch.stack(test_rela_gt) > 0).int()
    test_rela_gt_has = test_rela_gt[:, rela_pos]
    test_gt_has_sum = test_rela_gt_has.sum(1)
    test_rela_pos_has = test_gt_has_sum.nonzero().squeeze(1)
    return rela_pos, rela_pos_has, test_rela_pos_has


class RoidbListDataset(data.Dataset):
    def __init__(self, state, rela_pos, rela_pos_has, test_pos_has):
        self.hard_relation = self.hardsample_relation()
        self.state = state
        data_path = './../git/bottom-up-attention/vg_save_nms2'
        self.data_path = data_path
        print('    >>>> load roidb')
        roidb_path = os.path.join(data_path, 'roidb.pkl')
        roidb_file = open(roidb_path, 'rb')
        roidb = pickle.load(roidb_file, encoding='bytes')
        roidb_file.close()
        #self.gt_load_name = 'gt_data-cls400-rela100-400-tri399-img80-1421'
        self.gt_load_name = 'gt_data-cls400-rela90-hardsample'
        gt_info = torch.load('./'+self.gt_load_name+'.pt')
        #self.gt_name = 'gt_data-cls400-rela100-400-tri399-img80-1421-addcls'
        self.gt_name = 'gt_data-cls400-rela90-hardsample'
        self.gt_dict, self.cls_cluster = gt_info
        self.relation_label = self.get_relation_label(self.gt_dict)
        self.cls_cluster_num = 400
        self.cls_cluster = np.array(self.cls_cluster)

        '''
        rela_gt_path = os.path.join(data_path, 'use_rela_gt.pt')
        rela_gt = torch.load(rela_gt_path)

        rela_prob = torch.load(os.path.join(data_path, 'use_rela_select_prob.pt'))
        data_pkl_path = os.path.join(data_path, 'pkl')
        '''
        print('    >>>> make list')
        self.data_list_total, self.gt_list_total = self.make_list(roidb)
        img_dict = self.generate_img_dict(roidb)
        for dict_loop in list(img_dict.keys()):
            if len(img_dict[dict_loop]) == 0:
                pdb.set_trace()
        self.test_list = self.generate_test_name_list(img_dict)
        self.test_list = [os.path.join(self.data_path, 'pkl', test_name_) for test_name_ in self.test_list]
        if state == 'train':
            self.data_list = []
            self.gt_list = []
            for data_num in range(len(self.data_list_total)):
                if not self.data_list_total[data_num] in self.test_list:
                    self.data_list.append(self.data_list_total[data_num])
                    self.gt_list.append(self.gt_list_total[data_num])
        if state == 'test':
            self.gt_list = []
            self.data_list = self.test_list
            for test_data_ in self.data_list:
                self.gt_list.append(self.gt_list_total[self.data_list_total.index(test_data_)])
        print('-------- loaded %d data for %s --------\n' %(len(self.data_list), state))


        '''
        split_test = len(self.data_list_total)//5
        print('    >>>> re-clean relation')
        if state == 'train':
            self.rela_pos = rela_pos
            self.has_rela = rela_pos_has
            self.data_list = self.data_list_total[:-split_test]
            self.gt_list = self.gt_list_total[:-split_test]
            #self.pos_list = self.has_rela.tolist()
        elif state == 'test':
            self.rela_pos = rela_pos
            self.has_rela = test_pos_has
            self.data_list = self.data_list_total[-split_test:]
            self.gt_list = self.gt_list_total[-split_test:]
            #self.pos_list = [x for x in range(100)]
        '''
    def get_relation_label(self, gt_dict):
        rela_key = list(gt_dict.keys())
        relation_list = [int(sub_.split('-')[1]) for sub_ in rela_key]
        return list(set(relation_list))

    def hardsample_relation(self):
        relation=[3, 9, 12, 18, 22, 23, 33, 34,
                49, 52, 56, 60, 61, 65, 68, 72,
                73, 75, 83, 87, 94, 98, 105, 107,
                114, 115, 117, 119, 122, 127, 136,
                141, 150, 161, 170, 172, 189, 195,
                198, 204, 214, 215, 219, 222, 224,
                228, 231, 234, 254, 293, 295, 296,
                304, 306, 309, 314, 320, 328, 331,
                332, 338, 349, 356, 358, 359, 368,
                369, 376, 388, 395, 398, 406, 412,
                419, 424, 430, 438, 441, 444, 456,
                461, 462, 472, 473, 474, 476, 490,
                492, 494, 497]
        return relation

    def generate_img_dict(self, roidb):
        print('>>>> generate image dict')
        img_dict = {}
        hard_sample = True
        if hard_sample:
            for dict_key in self.hard_relation:
                img_dict[dict_key] = []
        else:
            for dict_key in list(self.gt_dict.keys()):
                img_dict[dict_key] = []
        if hard_sample:
            for roidb_piece in roidb:
                rela = copy.deepcopy(roidb_piece[b'gt_relations'])
                rela[:, 1] -= 1
                rela_list = rela[:, 1].tolist()
                rela_list = list(set(rela_list))
                for rela_item in rela_list:
                    if rela_item in self.hard_relation:
                        img_dict[rela_item].append(str(roidb_piece[b'image'])[2:-1])
            print('>>>> image_dict done')
            #count = [len(img_dict[tmp_key]) for tmp_key in list(img_dict.keys())]
            #pdb.set_trace()
            return img_dict

        for roidb_piece in roidb:
            rela = copy.deepcopy(roidb_piece[b'gt_relations'])
            cls = copy.deepcopy(roidb_piece[b'gt_classes'])
            rela[:, 0] = self.cls_cluster[cls[rela[:, 0]]-1]
            rela[:, 2] = self.cls_cluster[cls[rela[:, 2]]-1]
            rela[:, 1] -= 1
            rela_list = rela.tolist()
            rela_key = ['-'.join([str(x) for x in rela_piece]) for rela_piece in rela_list]
            rela_key = list(set(rela_key))
            for rela_item in rela_key:
                if rela_item in self.gt_dict:
                    img_dict[rela_item].append(str(roidb_piece[b'image'])[2:-1])
        print('>>>> image_dict done')
        count = [len(img_dict[tmp_key]) for tmp_key in list(img_dict.keys())]
        pdb.set_trace()
        return img_dict


    def generate_test_name_list(self, img_dict, select_num=2):
        print('>>>> generate test image list')
        test_name_list = []
        hard_sample = True
        if hard_sample:
            for rela_ in self.hard_relation:
                test_name_list += img_dict[rela_][:2]
            return ['-'.join(test_name_list_sub.split('/')[-3:])+'.pt' for test_name_list_sub in test_name_list]

        for rela_key in list(img_dict.keys()):
            #test_name_list += random.sample(img_dict[rela_key], 2)
            test_name_list += img_dict[rela_key][:2]
        return ['-'.join(test_name_list_sub.split('/')[-3:])+'.pt' for test_name_list_sub in test_name_list]



    def check_gt(self):
        rela_gt_path = os.path.join(self.data_path, 'use_rela_gt.pt')
        rela_gt = torch.load(rela_gt_path)

    def make_list(self, roidb):
        name_list = []
        gt_list = []
        cls_gt_list = []
        name_idx_list = []
        relation_label_list = []
        print('        >>>> traverse images')
        for roidb_piece in roidb:
            fname = roidb_piece[b'image'].decode('utf-8')
            save_name = '-'.join(fname.split('/')[-3:])+'.pt'
            save_data_folder = os.path.join(self.data_path, 'pkl_with_cls')
            if not os.path.exists(save_data_folder):
                os.mkdir(save_data_folder)
            data_path = os.path.join(self.data_path, 'pkl', save_name)
            gt_save_name = 'gt-'+save_name
            gt_folder = os.path.join(self.data_path, 'gt-pkl_'+self.gt_name)
            if not os.path.exists(gt_folder):
                os.mkdir(gt_folder)
            gt_path = os.path.join(gt_folder, gt_save_name)
            if os.path.exists(gt_path):
                name_list.append(data_path)
                gt_list.append(gt_path)
            if not os.path.exists(gt_path):
                rela = copy.deepcopy(roidb_piece[b'gt_relations'])
                cls = copy.deepcopy(roidb_piece[b'gt_classes'])
                rela[:,0] = cls[rela[:,0]]
                rela[:,2] = cls[rela[:,2]]
                rela[:,0] = self.cls_cluster[rela[:,0]-1]
                rela[:,2] = self.cls_cluster[rela[:,2]-1]
                rela[:,1] -= 1
                rela_tri = rela.tolist()
                gt = torch.zeros(len(self.gt_dict))
                cls_gt = torch.zeros(self.cls_cluster_num)
                relation_label_gt = torch.zeros(len(self.relation_label))
                has_gt = 0
                for rela_item in rela_tri:
                    rela_key = '-'.join([str(x) for x in rela_item])
                    if rela_key in self.gt_dict:
                        rela_tmp = list(set(rela[:, 1].tolist()))
                        relation_label_gt[self.relation_label.index(rela_item[1])] = 1
                        #gt[self.gt_dict[rela_key]] += 1
                        name_idx_list.append(rela_key)
                        gt[self.gt_dict[rela_key]] = 1
                        cls_gt[rela_item[0]] = 1
                        cls_gt[rela_item[2]] = 1
                        has_gt = 1
                if has_gt != 0:
                    name_list.append(data_path)
                    torch.save([gt, cls_gt, relation_label_gt], gt_path)
                    gt_list.append(gt_path)
        print('image set list generate done!')
        print('total image number : %d ' % len(name_list))
        return name_list, gt_list

    def triplet_size(self):
        return len(self.gt_dict)#len(self.rela_pos)

    def __getitem__(self, idx):
        data = torch.load(self.data_list[idx])#[self.pos_list[idx]])
        feat = data[0]
        gt, cls_gt, rela_gt = torch.load(self.gt_list[idx])#[self.pos_list[idx]])#data[1][self.rela_pos]
        freq = data[2]
        return feat, gt, cls_gt, rela_gt, freq

    def collate_fn(self, batch):
        data = torch.cat([x[0].unsqueeze(0) for x in batch], 0)
        gt = torch.cat([x[1].unsqueeze(0) for x in batch], 0)
        cls_gt = torch.cat([x[2].unsqueeze(0) for x in batch], 0)
        rela_gt = torch.cat([x[3].unsqueeze(0) for x in batch],0)
        '''
        freq = torch.tensor([x[3] for x in batch])
        if len(freq) == 1:
            freq = torch.tensor([1.0])

        rela_prob_sampler = torch.distributions.bernoulli.Bernoulli(freq)
        rela_mask = rela_prob_sampler.sample().nonzero()

        rela_gt_out = torch.zeros_like(gt)
        cls_gt_out = torch.zeros_like(cls_gt)
        feat_out = torch.zeros_like(data)

        rela_gt_out[:rela_mask.size(0)] = gt[rela_mask.squeeze(1)]
        cls_gt_out[:rela_mask.size(0)] = cls_gt[rela_mask.squeeze(1)]
        feat_out[:rela_mask.size(0)] = data[rela_mask.squeeze(1)]
        '''
        feat_out = data
        rela_gt_out = gt
        cls_gt_out = cls_gt
        return feat_out, rela_gt_out, cls_gt_out, rela_gt

    def __len__(self):
        return len(self.data_list)#len(self.pos_list)


if __name__ == '__main__':
    data_loader = RoidbListDataset('train')
    data_loader.reclean(8)
    pdb.set_trace()
    tmp = data_loader.triplet_size()
