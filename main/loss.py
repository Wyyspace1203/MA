from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        #pdb.set_trace()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class ProtoLoss(nn.Module):
    def __init__(self, nav_t: float):
        super(ProtoLoss, self).__init__()
        self.nav_t = nav_t
       
    
    def EuclideanDistances(self, src, tgt):
        # pdb.set_trace()
        vecProd = torch.matmul(src, tgt.t())
        SqA = src ** 2
        sumSqA = torch.sum(SqA, dim=1)
        sumSqA = torch.reshape(sumSqA, (vecProd.shape[0], 1))
        sumSqAEx = sumSqA.repeat(1, vecProd.shape[1])
        SqB = tgt ** 2
        sumSqB = torch.sum(SqB, dim=1)
        sumSqBEx = sumSqB.repeat(vecProd.shape[0], 1, )
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd
        ED = torch.sqrt(SqED)
        return ED


    def TripletCenterLoss(self, f_t, y_t, mu_s): ##f_t: 32x256
        self_t_mu = mu_s.cuda()
        
        ## chose high confidence target data
        sim_mat = torch.matmul(mu_s, f_t.T)  ##21x96
        #pdb.set_trace()
        #sim_mat_sum = torch.sum(sim_mat, dim=0)##96 
        #sim_mat_sum_sort = torch.sort(sim_mat_sum)[1]##96  ##32
        #sim_mat_sum_sort_new = sim_mat_sum_sort[:12] ##12
        #f_t_new=[]
        #for i in range(12):
        #    f_t_new.append(f_t[sim_mat_sum_sort_new[i]])
        #f_t_new = torch.tensor([item.cpu().detach().numpy() for item in f_t_new]).cuda()  #12x256

        #t_feature = f_t_new.cuda()
        t_feature = f_t.cuda()
        #s_labels, t_labels = labels_s, torch.max(y_t, 1)[1]
        t_labels = torch.max(y_t, 1)[1]

        # calculating TripletCenterLoss of source and target data
        #dist_s = self.EuclideanDistances(s_feature, self_s_mu)
        dist_t = self .EuclideanDistances(t_feature, self_t_mu)
        #sorted_s, indices_s = torch.sort(dist_s)
        sorted_t, indices_t = torch.sort(dist_t)
        #dist_s_ap = dist_s[torch.arange(bs), s_labels]
        #dist_s_an = sorted_s[:, 1]
        dist_t_ap = sorted_t[:, 0]
        #dist_t_an = sorted_t[:, 1]
        #dist_hinge_s = torch.clamp(10 + dist_s_ap - dist_s_an, min=0.0)
        dist_hinge_t = torch.clamp(dist_t_ap , min=0.0)
        #TCL_loss = 0.1 * torch.mean(dist_hinge_s) + 0.1 * torch.mean(dist_hinge_t)
        TCL_loss = 0.1 * torch.mean(dist_hinge_t)
        return TCL_loss


    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:##pred:new model output logits
        sim_mat = torch.matmul(mu_s, f_t.T)
        real_dist = F.softmax(sim_mat/self.nav_t, dim=0) 
        fake_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        target_loss = (cost_mat*fake_dist).sum(1).mean()
        consloss = 0.05 * self.TripletCenterLoss(f_t, y_t, mu_s)
        #loss = source_loss + target_loss 
        loss = target_loss + consloss
        return loss
                                      
