import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import pdb
import scipy.io as sio

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):#定义学习率的参数
    decay = (1 + gamma * iter_num / max_iter) ** (-power)##衰退率
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_test(resize_size=256, crop_size=224, alexnet=False):##图像预处理
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])##图片预处理

def data_load(args):
    ## prepare data数据载入模块
    dsets = {}
    dset_loaders = {}
    txt_src_test = open(args.s_dset_path).readlines()  # 源域的txt
    txt_tar_test = open(args.t_dset_path).readlines()  # 测试的txt

    dsets["source_test"] = ImageList(txt_src_test, transform=image_test())
    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=32, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    dsets["target_test"] = ImageList(txt_tar_test, transform=image_test())
    dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size=32, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):##只有测试过程用到了acc
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def test(args):  ##测试函数
    dset_loaders = data_load(args)
    # loader = dset_loaders
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    ##bok
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    ##classier
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    ##load model
    args.modelpath = '/media/ubuntu/HDD/lyt/Proto_2019/Proto_Private/object/ckps/source/uda/mi3dor/R/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = '/media/ubuntu/HDD/lyt/Proto_2019/Proto_Private/object/ckps/source/uda/mi3dor/R/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = '/media/ubuntu/HDD/lyt/Proto_2019/Proto_Private/object/ckps/source/uda/mi3dor/R/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    

    netF.eval()
    netB.eval()
    netC.eval()

    features_s = []
    labels_s = []
    features_t = []
    labels_t = []
    ## extract features and label

    with torch.no_grad():
        # iter_source_test = iter(loader)
        # iter_target_test = iter(loader)

        iter_source_test = iter(dset_loaders['source_test'])
        iter_target_test = iter(dset_loaders['target_test'])

        for i in range(len(iter_source_test)):
            # pdb.set_trace()
            data_s = next(iter_source_test)
            #data_s0 = iter_source_test.next()
            inputs_s = data_s[0]
            inputs_s = inputs_s.cuda()
            labels_s_0 = data_s[1]
            features_s_0 = netB(netF(inputs_s))
            features_s.extend(features_s_0.tolist())
            labels_s.extend(labels_s_0.tolist())




        for i in range(len(iter_target_test)):
            # pdb.set_trace()
            data_t = next(iter_target_test)
            inputs_t = data_t[0]
            inputs_t = inputs_t.cuda()
            labels_t_0 = data_t[1]
            features_t_0 = netB(netF(inputs_t))          
            features_t.extend(features_t_0.tolist())
            labels_t.extend(labels_t_0.tolist())


    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name,
                                                                                            acc_os2, acc_os1,
                                                                                            acc_unknown)
    else:
        if args.dset == 'VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc1, _ = cal_acc(dset_loaders['source_test'], netF, netB, netC, False)
            acc2, _ = cal_acc(dset_loaders['target_test'], netF, netB, netC, False)
            log_str = '\nTesting: Accuracy1 = {:.2f}% ;Accuracy2 = {:.2f}%'.format(acc1, acc2)##acc1 acc2


    print(log_str)
    return features_s, labels_s, features_t, labels_t


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='mi3dor', choices=['VISDA-C', 'office', 'mi3dor', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    args = parser.parse_args()


    if args.dset == 'mi3dor':
        names = ['rgb', 'view']
        args.class_num = 21

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        SEED = args.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        folder = './data/'  # path--------------------------------------
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_test_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_test_list.txt'
        test(args)
        features_s, labels_s, features_t, labels_t = test(args)
        sio.savemat('results.private_source_model.mat',
                {'source_feature': features_s, 'source_label': labels_s,
                 'target_feature': features_t, 'target_label': labels_t, })


