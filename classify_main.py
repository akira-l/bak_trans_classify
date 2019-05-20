from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import datetime
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as tushow
torch.backends.cudnn.enabled=False

from collections import OrderedDict

from classify_model import transformer, conv3d_model, bilinear_model
from classify_loss import cls_loss
from pkl_dataloader import RoidbListDataset, reclean

import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='agent main')
    parser.add_argument('--epoch', dest='epoch',
                        default=60, type=int)
    parser.add_argument('--save_prefix', dest='save_prefix',
                        default="./transformer_save", type=str)
    parser.add_argument('--bs', dest='batch_size',
                        default=256, type=int)

    parser.add_argument('--lr', type=float,
                        default=0.001)
    parser.add_argument('--p_c', dest='precision_checkpoint',
                        default=10, type=int)
    # train val smalltrain smallval minitrain minival
    parser.add_argument('--data_version', dest='data_version',
                        default='minitrain', type=str)
    parser.add_argument('--optim', dest='optim',
                        default='adam', type=str)
    parser.add_argument('--single_gpu', dest='single_gpu',
                        action='store_true')
    parser.add_argument('--check_epoch', dest='check_epoch',
                        default=1, type=int)

    args = parser.parse_args()
    return args

def train(args):
    # maintenance log
    print('generate related content ....')
    save_suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    save_dir = args.save_prefix + save_suffix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_title = 'log_'+save_suffix+'.txt'
    log_path = os.path.join(save_dir, log_title)
    test_log_file = 'test_log.txt'
    precision_title = 'precision_'+save_suffix+'.txt'
    precision_path = os.path.join(save_dir, precision_title)
    train_info_log = 'train_info'+save_suffix+'.txt'
    train_info_path = os.path.join(save_dir, train_info_log)
    message_record(train_info_path, train_info_path)
    message_record(train_info_path, str(args))
    check_epoch = args.check_epoch

    # prepare data
    print('prepare data ....')
    #rela_pos, rela_has, test_has = reclean()

    dataset = RoidbListDataset('train', rela_pos=None, rela_pos_has=None, test_pos_has=None)
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=dataset.collate_fn)
    triplet_size = dataset.triplet_size()
    iteration = data_loader.__len__()
    print('prepare model ....')
    get_model = bilinear_model(triplet_size)#transformer(triplet_size)
    get_loss = cls_loss(triplet_size)

    message_record(train_info_path, str(get_model))


    if not args.single_gpu:
        get_model = torch.nn.DataParallel(get_model, device_ids=range(torch.cuda.device_count()))
    get_model.cuda()
    get_model.train()

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, get_model.parameters()), lr=args.lr)#0.0002)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(get_model.parameters(),
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=1e-4)
    else:
        raise NotImplementedError

    precision_checkpoint = args.precision_checkpoint
    iter_time = 10
    # training
    for epoch in range(args.epoch):
        print('\n\n\n\n---------- start training in epoch %d  ----------' % epoch)
        for batch_idx, (inputs, gt, cls_gt, rela_gt) in enumerate(data_loader):
            if batch_idx != 0:
                time_end = time.time()
                iter_time = time_end - time_start
                time_start = time_end
            else:
                time_start = time.time()
            time_start = time.time()
            feature = Variable(inputs)
            relation = Variable(gt)
            obj_label = Variable(cls_gt)
            rela_label = Variable(rela_gt)

            cls_out, obj_out, rela_out = get_model(feature)
            torch.cuda.synchronize()
            rela_loss, re_gt, obj_loss, re_cls_gt, relation_label_loss, re_rela_gt = get_loss(cls_out, relation, obj_out, obj_label, rela_out, rela_label)
            loss = rela_loss+obj_loss+relation_label_loss
            #loss = obj_loss + relation_label_loss

            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
            torch.cuda.synchronize()

            print('training epoch: [%d / %d], iteration: [%d / %d], loss: %.4f, rela_loss: %.4f, obj_loss: %.4f, rela_label_loss: %.4f ' % (epoch,
                    args.epoch,
                    batch_idx,
                    iteration,
                    loss,
                    rela_loss,
                    obj_loss,
                    relation_label_loss))
            #print('loss %.4f' % (loss))
            print('iter time: %.4f, rest time: %.4f' \
                   % (iter_time, (iteration-batch_idx+iteration*(args.epoch-epoch))*iter_time/60.0 ))

            write_log(log_path,
                    [epoch,
                    args.epoch,
                    batch_idx,
                    iteration,
                    round(float(loss), 4),
                    round(float(rela_loss),4),
                    round(float(obj_loss),4),
                    round(float(relation_label_loss), 4)],
                    ['current_epoch',
                        'total_epoch',
                        'current_times',
                        'iteration',
                        'loss',
                        'rela_loss',
                        'cls_loss',
                        'rela_label_loss'])

            if (batch_idx % precision_checkpoint == 0) and (batch_idx != 0):
                precision_, recall_, f_val_, thresh_ = count_p_r(cls_out, re_gt, step=0.1)
                obj_prec, obj_recall, obj_fval, obj_thresh = count_p_r(obj_out, re_cls_gt, step=0.1)
                rela_prec, rela_recall, rela_fval, rela_thresh = count_p_r(rela_out, re_rela_gt, step=0.1)
                write_log(precision_path,
                    [epoch,
                    args.epoch,
                    batch_idx,
                    iteration,
                    round(precision_, 4),
                    round(recall_, 4),
                    round(f_val_, 4),
                    round(thresh_, 4),
                    round(obj_prec, 4),
                    round(obj_recall, 4),
                    round(obj_fval, 4),
                    round(obj_thresh, 4),
                    round(rela_prec, 4),
                    round(rela_recall, 4),
                    round(rela_fval, 4),
                    round(rela_thresh, 4)],
                    ['current_epoch',
                        'total_epoch',
                        'current_times',
                        'iteration',
                        'precision',
                        'recall',
                        'f_val',
                        'thresh',
                        'obj_prec',
                        'obj_recall',
                        'obj_fval',
                        'obj_thresh',
                        'rela_prec',
                        'rela_recall',
                        'rela_fval',
                        'rela_thresh'])
            if (batch_idx % (iteration-1) == 0 ) and (epoch != 0):
                torch.save(get_model.state_dict(), save_dir+'/retrieval_save'+str(epoch)+'.pkl')

        if (epoch % check_epoch ==0) and (epoch !=0):
            test(args, save_dir, rela_pos=None, rela_has=None, test_log=test_log_file)
    return save_dir, log_path, precision_path, test_log_file

######################################################
######################################################
######################################################
######################################################
def test(args, save_path, rela_pos, rela_has, test_log='test_log.txt'):
    file_list = os.listdir(save_path)
    save_num = 1
    for name in file_list:
        if name[-4:] == '.pkl':
            if name.split('_')[0] == 'retrieval':
                tmp_num = int(re.findall('.*save(.*).pkl', name)[0])
                save_num = tmp_num if tmp_num > save_num else save_num
    retrieval_pkl = 'retrieval_save'+ str(save_num) +'.pkl'
    load_pkl_path = os.path.join(save_path, retrieval_pkl)

    test_log_path = os.path.join(save_path, test_log)

    print('\n\n\n\n-------- test model in save number : %d --------' % save_num)
    # prepare data
    print('prepare data ....')
    #rela_pos, rela_has = reclean()
    dataset = RoidbListDataset(state='test', rela_pos=rela_pos, rela_pos_has=None, test_pos_has=rela_has)
    iteration = dataset.__len__()
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=iteration//4,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=dataset.collate_fn)
    triplet_size = dataset.triplet_size()

    get_model = bilinear_model(triplet_size)#transformer(triplet_size)
    get_loss = cls_loss(triplet_size)

    if not args.single_gpu:
        get_model = torch.nn.DataParallel(get_model, device_ids=range(torch.cuda.device_count()))
    get_model.cuda()
    get_model.eval()
    state_dict = torch.load(load_pkl_path)
    try:
        get_model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        get_model.load_state_dict(new_state_dict)

    message_record(test_log_path, '---------------')
    prec_rec = []
    obj_prec_rec = []
    rela_prec_rec = []
    for batch_idx, (inputs, gt, cls_gt, rela_gt) in enumerate(data_loader):
        feature = Variable(inputs)
        relation = Variable(gt)
        obj_label = Variable(cls_gt)
        rela_label = Variable(rela_gt)

        #input_rpn = Variable(feature)
        cls_out, obj_out, rela_out = get_model(feature)
        rela_loss, re_gt, obj_loss, re_cls_gt, rela_label_loss, re_rela_gt = get_loss(cls_out, relation, obj_out, obj_label, rela_out, rela_label)
        loss = rela_loss+obj_loss+rela_label_loss

        #print('iteration: [%d / %d]' % (batch_idx, iteration))
        #print('loss %.4f' % (loss))

        obj_prec, obj_recall, obj_fval, obj_thresh = count_p_r(obj_out, re_cls_gt, step=0.1)
        precision_, recall_, f_val_, thresh_ = count_p_r(cls_out, re_gt, step=0.1)
        rela_prec, rela_recall, rela_fval, rela_thresh = count_p_r(rela_out, re_rela_gt, step=0.1)
        print('rela loss: %.4f |rela precision: %.4f |rela recall: %.4f | f_val: %.4f |rela threshold: %.4f' %( rela_loss, rela_prec, rela_recall, rela_fval, rela_thresh) )
        rela_prec_rec.append(rela_prec)
        print('obj loss: %.4f |obj precision: %.4f |obj recall: %.4f | f_val: %.4f |obj threshold: %.4f' %( obj_loss, obj_prec, obj_recall, obj_fval, obj_thresh) )
        obj_prec_rec.append(obj_prec)
        print('loss: %.4f | ****precision: %.4f | recall: %.4f | f_val: %.4f | threshold: %.4f' %(loss, precision_, recall_, f_val_, thresh_) )
        prec_rec.append(precision_)
        write_log(test_log_path,
        [save_num,
        batch_idx,
        iteration,
        round(float(loss), 4),
        round(precision_, 4),
        round(recall_, 4),
        round(f_val_, 4),
        round(thresh_, 4),
        round(obj_prec, 4),
        round(obj_recall, 4),
        round(obj_fval, 4),
        round(obj_thresh, 4),
        round(rela_prec, 4),
        round(rela_recall, 4),
        round(rela_fval, 4),
        round(rela_thresh, 4)],
        ['epoch',
            'current_times',
            'iteration',
            'loss',
            'precision',
            'recall',
            'f_val',
            'threshold',
            'obj_prec',
            'obj_recall',
            'obj_fval',
            'obj_thresh',
            'rela_prec',
            'rela_recall',
            'rela_fval',
            'rela_thresh'])
    print('-------- average precision %.4f --------' %  round(sum(prec_rec)/len(prec_rec), 4))
    print('-------- average object precision %.4f --------' %  round(sum(obj_prec_rec)/len(obj_prec_rec), 4))
    print('-------- average relation precision %.4f --------' %  round(sum(rela_prec_rec)/len(rela_prec_rec), 4))

    write_log(test_log_path,
            [round(sum(prec_rec)/len(prec_rec), 4),
            round(sum(obj_prec_rec)/len(obj_prec_rec), 4),
            round(sum(rela_prec_rec)/len(rela_prec_rec), 4)],
            ['average_precision',
             'obj_averge_precision',
             'rela_average_precision'])



####################################################
####################################################
####################################################
####################################################

def count_p_r(cur_pred, gt, step, average=True):
    bs = cur_pred.size(0)
    #pred_softmax = F.softmax(cur_pred, 1)
    pred_softmax = F.sigmoid(cur_pred)
    pred_min_matrix = pred_softmax.min(1)[0].unsqueeze(1).repeat(1, cur_pred.size(1))
    pred_max_matrix = pred_softmax.max(1)[0].unsqueeze(1).repeat(1, cur_pred.size(1))
    pred = (pred_softmax-pred_min_matrix) / (pred_max_matrix-pred_min_matrix)
    thresh_list = [step*x for x in range(10)]
    precision_record = []
    recall_record = []
    thresh_record = []
    f_record = []
    for batch_counter in range(bs):
        gt_tmp = gt[batch_counter]
        pred_tmp = pred[batch_counter]
        precision_list = [((pred_tmp > thresh).float()*gt_tmp.float()).sum().float() / ((pred_tmp>thresh).float().sum()+1e-5) for thresh in thresh_list]
        recall_list = [((pred_tmp > thresh).float()*gt_tmp.float()).sum().float() / (gt_tmp.float().sum()+1e-5) for thresh in thresh_list]
        precision_list = [0 if x>1 else x for x in precision_list]
        recall_list = [0 if x>1 else x for x in recall_list]
        f_list = [p*r*2/(p+r+1e-5) for p,r in zip(precision_list, recall_list)]
        f_list = [0 if x>1 else x for x in f_list]
        ind = f_list.index(max(f_list))
        precision_record.append(precision_list[ind])
        recall_record.append(recall_list[ind])
        thresh_record.append(thresh_list[ind])
        f_record.append(f_list[ind])
    if average:
        return (sum(precision_record) / len(precision_record)).item(),\
               (sum(recall_record) / len(recall_record)).item(),\
               (sum(f_record) / len(f_record)).item(), \
               (sum(thresh_record) / len(thresh_record))
    else:
        result_pos = precision_record.index(max(precision_record))
        return precision_record[result_pos].item(), \
               recall_record[result_pos].item(), \
               f_record[result_pos].item(), \
               thresh_record[result_pos].item()

def message_record(log_path, content):
    file_obj = open(log_path, 'a')
    file_obj.write('args:'+content+'\n')
    file_obj.close()

def write_log(log_path, record_list, name_list):
    file_obj = open(log_path, 'a')
    content = ''
    for val, name in zip(record_list, name_list):
        content += '_' + name + '=' + str(val) + '_'
    content += '\n'
    file_obj.write(content)
    file_obj.close()

def result_plot(target_path, save_path, key_word, state='test'):
    file_obj = open(target_path, 'r')
    logs = file_obj.readlines()

    color_list = ['b', 'g', 'r', 'm', 'y', 'k']
    mark_list = ['--', '-.', ':', '.', 'o', ',', 's', 'p', '*', '+',
            'X', 'D', 'd']
    if len(key_word) > len(color_list):
        plot_type_list = color_list+[random.sample(color_list, 1)[0] + random.sample(mark_list, 1)[0] for x in range(len(key_word)-len(color_list))]
    else:
        plot_type_list = color_list
    plt.cla()
    key_word_counter = 0
    for key_word_ in key_word:
        plot_var = []
        for log_line in logs:
            log_line_list = re.split('__|=', log_line[1:-2])
            if key_word_ in log_line_list:
                plot_var.append(float(log_line_list[log_line_list.index(key_word_)+1]))
        if plot_var == []:
            raise ValueError('No this key word in log named: %s' % key_word_)
        coordx = [x for x in range(len(plot_var))]
        plt.plot(coordx, plot_var, plot_type_list[key_word_counter])
        key_word_counter += 1
    label = key_word
    plt.legend(label, loc='center left', bbox_to_anchor=(1.05, 1))

    name = state + '-'+'_'.join(key_word[:1]) + '.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, name))



if __name__ == '__main__':
    args = parse_args()
    save_path, train_path, precision_path, test_log = train(args)

    #save_path = './bin_test2018-11-09-07_07_51'
    log_path = os.path.join(save_path, test_log)
    save_key_word = ['average_precision', 'obj_averge_precision', 'rela_average_precision']
    result_plot(log_path, save_path, save_key_word)
    save_key_word = ['precision', 'recall', 'obj_prec', 'obj_recall', 'rela_prec', 'rela_recall']
    result_plot(log_path, save_path, save_key_word)
    save_key_word = ['loss']
    result_plot(log_path, save_path, save_key_word)

    #train_path = os.path.join(save_path, 'log_2018-11-09-07_07_51.txt')
    save_key_word = ['loss', 'rela_loss', 'cls_loss', 'rela_label_loss']
    result_plot(train_path, save_path, save_key_word, state='train')

    #precision_path = os.path.join(save_path, 'precision_2018-11-09-07_07_51.txt')
    save_key_word = ['precision', 'recall', 'obj_prec', 'obj_recall', 'rela_prec', 'rela_recall']
    result_plot(precision_path, save_path, save_key_word, state='train')

    #result_plot(precision_path, save_path, ['precision', 'recall', 'f_val', 'thresh'])

    #save_path = './transformer_save2018-10-28-13_25_11'
    #log_path = os.path.join(save_path, 'log_2018-09-29-06_51_57.txt')
    #precision_path = os.path.join(save_path, 'precision_2018-09-29-06_51_57.txt')
    #rela_pos, rela_has, test_has = reclean()
    #test(args, save_path, rela_pos, test_has)
