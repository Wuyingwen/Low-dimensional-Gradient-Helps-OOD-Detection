from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import torch
import os
import numpy as np

from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar10,Cifar100,Cifar10_for_ood
from data.dataloader_for_ood import IN_DATA, get_loader_out

from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.multi_gpu import setup_for_distributed

import sys; sys.path.append("..")
from sam import SAM
from sam_wyw import SAM_wyw
from model.vgg_lt import VGG16BN, VGG19BN
from model.wide_res_net import WideResNet
from model.resnet import resnet18
import matplotlib.pyplot as plt
from functorch import vmap
from utils_ntk import autograd_ntk, autograd_components_ntk
import jax
import torch.nn as nn
import scipy
from model.lenet5 import LeNet
from data.mnist import MNIST
from sklearn.decomposition import TruncatedSVD
import time
from functorch import make_functional_with_buffers, vmap, grad, jvp
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from functools import partial
from advertorch.attacks import LinfPGDAttack
import copy
import faiss
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.imagenet import ImageNet
from mahalanobis_lib import get_Mahalanobis_score,sample_estimator,merge_and_generate_labels,block_split,load_characteristics
from model.smooth_cross_entropy import smooth_crossentropy
import math

def get_model_vec_torch(model):
    vec = []
    for n,p in model.named_parameters():
        vec.append(p.data.detach().reshape(-1)) 
    return torch.cat(vec, 0)
       
def get_model_grad_vec_torch(optimizer):
    vec = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            vec.append(p.grad.data.detach().reshape(-1)) 
    return torch.cat(vec, 0)

def get_model_grad_vec_torch_2(model):
    vec = []
    for n,p in model.named_parameters():
        # print(n,p.shape)
        vec.append(p.grad.data.detach().reshape(-1)) 
    # print(fsdfs)
    return torch.cat(vec, 0)

def update_param(model, param_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.shape
        size = arr_shape.numel()
        param.data = param_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size
    return

class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#AUTHOR: ANDREW ENGEL
#andrew.engel@pnnl.gov
def vmap_ntk_loader(model :torch.nn.Module, xloader :torch.utils.data.DataLoader, device='cuda'): #y :torch.tensor):
    """Calculates the Components of the NTK and places into a dictionary whose keys are the named parameters of the model. 
    
    While torch.vmap function is still in development, there appears to me to be an issue with how
    greedy torch.vmap allocates reserved memory. Batching the calls to vmap seems to help. Just like
    with training a model: you can go very fast with high batch size, but it requires an absurd amount 
    of memory. Unlike training a model, there is no regularization, so you should make batch size as high
    as possible
    
    We suggest clearing the cache after running this operation.
    
        parameters:
            model: a torch.nn.Module object that terminates to a single neuron output
            xloader: a torch.data.utils.DataLoader object whose first value is the input data to the model
            device: a string, either 'cpu' or 'cuda' where the model will be run on
            
        returns:
            NTKs: a dictionary whose keys are the names parameters and values are said parameters additive contribution to the NTK
    """
    NTKs = {}
        
    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)

    for i,z in enumerate(model.named_parameters()):
        if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
            continue
        name, param = z
        J_layer=[]
        for j,data in enumerate(xloader):
            inputs = data[0]
            inputs = inputs.to(device, non_blocking=True)
            basis_vectors = torch.eye(len(inputs),device=device,dtype=torch.bool) 
            pred = model(inputs)#[:,0]
            y = torch.sum(torch.exp(pred), dim=1)

            # Seems like for retain_graph=False, you might need to do multiple forward passes.
            def torch_row_Jacobian(v): # y would have to be a single piece of the batch
                return torch.autograd.grad(y,param,v)[0].reshape(-1)
            J_layer.append(vmap(torch_row_Jacobian)(basis_vectors).detach())  # [N*p] matrix ？
            if device=='cuda':
                torch.cuda.empty_cache()
        J_layer = torch.cat(J_layer)
        NTKs[name] = J_layer @ J_layer.T

    return NTKs

class Energy_Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        y = torch.log(torch.sum(torch.exp(self.model(x)),dim=1))
        # print('y:', y.shape)
        return y

class Equal(nn.Module):
    def __init__(self):
        super(Equal, self).__init__()
        
    def forward(self, x):
        y = x
        return y
    
class Linear_Probe(nn.Module):
    def __init__(self, in_dim, out_dim, ifbn=True):
        super(Linear_Probe, self).__init__()
        self.equal = Equal()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.ifbn = ifbn
        
    def forward(self, x):
        x = self.equal(x)
        if self.ifbn == True:
            x = self.bn(x)
        y = self.fc(x)
        return y


def matrix_jacobian_product(model, x, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(N, k), N为样本个数
    def vjp_single(model, x, M, kernel, z_theta, I_2):
        bz = x.shape[0]
        fmodel, params, buffers = make_functional_with_buffers(model)
        def compute_loss_stateless_model(params, buffers, sample):
            # prenum = 1000
            # loss = 0
            # for k in range(int(bz/prenum)):
            #     predictions = fmodel(params, buffers,sample[k*prenum:min((k+1)*prenum,bz)])
            #     loss = loss + torch.sum(M[k*prenum:min((k+1)*prenum,bz)]*predictions)
            predictions = fmodel(params, buffers, sample)
            loss = torch.sum(M*predictions)
            return loss
        ft_compute_grad = grad(compute_loss_stateless_model)
        z_t_ = ft_compute_grad(params, buffers, x)
        #### 将梯度 flatten 成一维向量
        z_t = []
        for i in range(len(z_t_)):
            z_t.append(z_t_[i].view(-1))
        z_t = torch.cat(z_t, dim=0)
        if kernel == 'NFK':
            z_t = I_2 * (z_t - torch.sum(M)*z_theta)
        return z_t

    mjp = vmap(vjp_single, in_dims=(None, None, 1, None, None, None), out_dims=(1))
    omiga = mjp(model, x, M, kernel, z_theta, I_2)

    return omiga


def matrix_jacobian_product_2(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(N, k), N为样本个数
    num = 0
    omiga = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        omiga = omiga + matrix_jacobian_product(model, inputs, M[num:num+inputs.shape[0]], kernel=kernel, z_theta=z_theta, I_2=I_2)
        num = num + inputs.shape[0]
    return omiga


def jacobian_matrix_product(model, x, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(|theta|, k)，|theta|为参数维度
    def jvp_single(model, x, M, kernel, z_theta, I_2):
        # model.zero_grad()
        fmodel, params, buffers = make_functional_with_buffers(model)
        def compute_loss_stateless_model(params, buffers, sample):
            predictions = fmodel(params, buffers, sample)
            return predictions
        # ft_compute_grad = grad(compute_loss_stateless_model)
        function = partial(compute_loss_stateless_model, buffers=buffers, sample=x)

        if kernel == 'NFK':
            M = I_2*M

        _, M_temp, _ = make_functional_with_buffers(model)
        M_temp = list(M_temp)
        idx = 0
        for i in range(len(M_temp)):
            arr_shape = M_temp[i].shape
            size = arr_shape.numel()
            M_temp[i] = M[idx:idx+size].reshape(arr_shape).clone()
            idx += size
        M_temp = tuple(M_temp)     
            
        value, grad2 = jvp(function, (params,), (M_temp,))
        grad2 = grad2.detach()
        del M_temp
        if kernel == 'NFK':
            grad2 = grad2 - torch.dot(z_theta, M)
        return grad2
    
    jmp = vmap(jvp_single, in_dims=(None, None, 1, None, None, None), out_dims=(1))
    omiga = jmp(model, x, M, kernel, z_theta, I_2)

    # y = model(x)
    # theta = get_model_vec_torch(model)
    # eps = 1
    # if kernel == 'NTK':
    #     I_2 = torch.ones(theta.shape)
    # delta_w = I_2*M.squeeze()
    # print('delta_w:', delta_w)
    # print((theta + delta_w*eps))
    # update_param(model, theta + delta_w*eps)
    # dy = (model(x) - y)/eps
    # print(dy)
    # print(fsdf)
    # if kernel == 'NFK':
    #     dy = dy - torch.dot(z_theta, delta_w)
    
    # print('diff:', torch.norm(dy-omiga))
    # print(dy)
    # print(omiga)
    # print(fsfsd)

    return omiga


def jacobian_matrix_product_2(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(|theta|, k)，|theta|为参数维度
    omiga = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        omiga.append(jacobian_matrix_product(model, inputs, M, kernel=kernel, z_theta=z_theta, I_2=I_2))
    omiga = torch.cat(omiga, dim=0)
    return omiga


def pre_compute(model, dataloader):
    sum = 0
    nexample = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()

        nexample += inputs.shape[0]

        model.zero_grad()
        y = model(inputs)
        loss = torch.mean(y)
        loss.backward()
        sum = sum + get_model_grad_vec_torch_2(model)*inputs.shape[0]

    print('nexample: ', nexample)
    z_theta = sum/nexample

    L = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        for i in range(inputs.shape[0]):
            model.zero_grad()
            y_i = model(inputs[i].unsqueeze(0))
            y_i.backward()
            z_i = get_model_grad_vec_torch_2(model)
            u_i =  z_i - z_theta
            ### diagonal approximation
            L_i = torch.square(u_i)
            L = L + L_i
    L = L / nexample
    # L^(-1/2)
    I_2 = 1/ torch.sqrt(L)
    #### deal with inf_value in I_2
    I_2_ = torch.where(torch.isinf(I_2), torch.full_like(I_2, 1), I_2)
    I_2 = torch.where(I_2_==1, torch.full_like(I_2_, float(I_2_.max())), I_2_)

    return z_theta, L, I_2


def truncated_svd(model, x, k=1, iter=10, kernel='NTK', z_theta=None, I_2=None):
    n = x.shape[0]  # 样本数
    omiga = torch.randn((n, k)).cuda()
    omiga, _ = torch.linalg.qr(omiga)
    # print(omiga.shape, omiga[:,0].shape, torch.norm(omiga[:,0]))
    omiga = matrix_jacobian_product(model, x, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2)
    # print('omiga:', omiga.shape)
    for i in range(iter):
        omiga = jacobian_matrix_product(model, x, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2)
        # print('omiga:', omiga.shape)
        # print(fdsfsd)
        omiga = matrix_jacobian_product(model, x, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2)
        # omiga = omiga / torch.norm(omiga, p=np.inf)
        omiga, _ = torch.linalg.qr(omiga)

    b = jacobian_matrix_product(model, x, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2)
    p, sigma, q = torch.svd(b.T)
    p = omiga@p
    return p, sigma, q


def truncated_svd_2(model, dataloader, k=1, iter=10, kernel='NTK', z_theta=None, I_2=None):
    n = 0
    for batch in dataloader:
        inputs, targets = batch
        n = n + inputs.shape[0]
    print('sample number:', n)

    omiga = torch.randn((n, k)).cuda()
    omiga, _ = torch.linalg.qr(omiga)
    # print(omiga.shape, omiga[:,0].shape, torch.norm(omiga[:,0]))
    omiga = matrix_jacobian_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
    # print('omiga:', omiga.shape)
    for i in range(iter):
        omiga = jacobian_matrix_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # print('omiga:', omiga.shape)
        # print(fdsfsd)
        omiga = matrix_jacobian_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # omiga = omiga / torch.norm(omiga, p=np.inf)
        omiga, _ = torch.linalg.qr(omiga)

    b = jacobian_matrix_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
    p, sigma, q = torch.svd(b.T)
    p = omiga@p
    return p, sigma, q


def cal_fisher_vector(model, x, kernel='NTK', z_theta=None, I_2=None):
    bz = x.shape[0]
    fmodel, params, buffers = make_functional_with_buffers(model)
    def compute_loss_stateless_model(params, buffers, sample):
        batch = sample.unsqueeze(0)
        predictions = fmodel(params, buffers, batch).squeeze()
        # print(predictions.shape)
        return predictions
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x)
    ft_per_sample_grads_ = []
    for i in range(len(ft_per_sample_grads)):
        ft_per_sample_grads_.append(ft_per_sample_grads[i].view(bz, -1))
    ft_per_sample_grads_ = torch.cat(ft_per_sample_grads_, dim=1)
    # print('ft_per_sample_grads_: ', ft_per_sample_grads_.shape)
    # print(len(ft_per_sample_grads))
    # print(ft_per_sample_grads[0].shape)
    if kernel == 'NFK':
        # print(ft_per_sample_grads_.shape, z_theta.shape)
        ft_per_sample_grads_ =  ft_per_sample_grads_ - z_theta
        def dot_per_element(I_2, z_i):
            out = I_2 * z_i
            return out
        dot = vmap(dot_per_element, in_dims=(None, 0))
        ft_per_sample_grads_ = dot(I_2, ft_per_sample_grads_)
        # print(I_2)
        # print(fsdfs)
        # for i in range(bz):
        #     ft_per_sample_grads_[i] = I_2 * ft_per_sample_grads_[i]

    # v = []
    # for i in range(x.shape[0]):
    #     model.zero_grad()
    #     y_i = model(x[i].unsqueeze(0))
    #     y_i.backward()
    #     z_i = get_model_grad_vec_torch_2(model)
    #     if kernel == 'NFK':
    #         z_i =  z_i - z_theta
    #         z_i = I_2 * z_i
    #     v.append(z_i.unsqueeze(0))
    # v = torch.cat(v, 0)  

    # for per_sample_grad, ft_per_sample_grad in zip(v, ft_per_sample_grads_):
    #     assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=3e-3, rtol=1e-5)
    
    return ft_per_sample_grads_
    

def cal_avg_fisher_vector_per_class(model, dataloader, kernel='NTK', z_theta=None, I_2=None, class_num=0):
    nexample = 0
    avg_z = 0
    sum_z = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        for i in range(inputs.shape[0]):
            if targets[i] == class_num:
                nexample += 1
                model.zero_grad()
                y_i = model(inputs[i].unsqueeze(0))
                y_i.backward()
                z_i = get_model_grad_vec_torch_2(model)
                
                if kernel == 'NFK':
                    z_i =  z_i - z_theta
                    z_i = I_2 * z_i
                sum_z = sum_z + z_i
    print('nexample:', nexample)
    avg_z = sum_z/nexample
    
    return avg_z


def cal_fisher_vector_per_sample(model, x, kernel='NTK', z_theta=None, I_2=None):
    model.zero_grad()
    y_i = model(x.unsqueeze(0))
    y_i.backward()
    z_i = get_model_grad_vec_torch_2(model)
    if kernel == 'NFK':
        z_i =  z_i - z_theta
        z_i = I_2 * z_i

    return z_i


def obtain_feature_embedding_by_jvp(model, x, P, kernel='NTK', z_theta=None, I_2=None):
    ## x:(bz,), bz为样本数，P:(|theta|, k)，|theta|为参数维度
    def jvp_single(x, P, kernel, z_theta, I_2):
        fmodel, params, buffers = make_functional_with_buffers(model)
        _, P_temp, _ = make_functional_with_buffers(model)
        def compute_loss_stateless_model(params, buffers, sample):
            predictions = fmodel(params, buffers, sample)
            return predictions
        function = partial(compute_loss_stateless_model, buffers=buffers, sample=x)

        if kernel == 'NFK':
            # print(P.shape, I_2.shape)
            P = I_2*P

        P_temp = list(P_temp)
        idx = 0
        for i in range(len(P_temp)):
            arr_shape = P_temp[i].shape
            size = arr_shape.numel()
            P_temp[i] = P[idx:idx+size].reshape(arr_shape).clone()
            idx += size
        P_temp = tuple(P_temp)     
            
        value, grad2 = jvp(function, (params,), (P_temp,))
        grad2 = grad2.detach()
        if kernel == 'NFK':
            grad2 = grad2 - torch.dot(z_theta, P)
        return grad2
    
    jmp = vmap(jvp_single, in_dims=(None, 1, None, None, None), out_dims=(1))
    feature = jmp(x, P, kernel, z_theta, I_2)

    # y = model(x)
    # theta = get_model_vec_torch(model)
    # eps = 1
    # if kernel == 'NTK':
    #     I_2 = torch.ones(theta.shape)
    # delta_w = I_2*M.squeeze()
    # print('delta_w:', delta_w)
    # print((theta + delta_w*eps))
    # update_param(model, theta + delta_w*eps)
    # dy = (model(x) - y)/eps
    # print(dy)
    # print(fsdf)
    # if kernel == 'NFK':
    #     dy = dy - torch.dot(z_theta, delta_w)
    
    # print('diff:', torch.norm(dy-omiga))
    # print(dy)
    # print(omiga)
    # print(fsfsd)

    # # v_x:(bz,M), p:(M,k), M为参数维度,k为降维维度
    # feature = torch.mm(v_x, p)
    return feature


def reconstruct_gradient_feature_by_p(model, x, P, kernel='NTK', z_theta=None, I_2=None):
    ## x:(bz,), bz为样本数，P:(|theta|, k)，|theta|为参数维度
    coeff = obtain_feature_embedding_by_jvp(model, x, P, kernel, z_theta, I_2)
    print('coeff:', coeff.shape)
    reconstruct_grad_feature = coeff @ P.T
    return reconstruct_grad_feature


def obtain_feature_embedding(v_x, p):
    # v_x:(bz,M), p:(M,k), M为参数维度,k为降维维度
    feature = torch.mm(v_x, p)
    return feature

    # fea = []
    # for i in range(p.shape[0]):
    #     fea.append((torch.dot(v_x, p[i])/N).unsqueeze(0))
    # feature = torch.cat(fea, 0)
    # return feature


def select_cifar10_100(test_loader, model, num=100, type='None'):
    model.eval()
    data = []
    label = [] 
    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.cpu().squeeze().numpy()
        index = np.argwhere(correct==True).flatten()
        for j in index:
            if len(data)<num:
                label.append(target[j].cpu().numpy())
                data.append(input[j].cpu().numpy())
            else:
                x_test = np.array(data)
                y_test = np.array(label)
                print(x_test.shape, y_test.shape)
                np.save('data/cifar10_pre_resnet18_' + type + '_imgs_0.npy', x_test)
                np.save('data/cifar10_pre_resnet18_' + type + '_lbls_0.npy', y_test)
                return
            
        
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def threshold_for_detect(reconstruct_error, confidence=0.95):
    # reconstruct_error: [len(val_set),], torch type
    min_value = torch.median(reconstruct_error)
    max_value = torch.max(reconstruct_error)
    lamda = (min_value + max_value)/2
    conf = 0
    while conf < confidence or (max_value-min_value)>10:
        conf = (reconstruct_error < lamda).sum()/reconstruct_error.shape[0]
        if conf < confidence:
            min_value = lamda
            lamda = (min_value + max_value)/2
        else:
            max_value = lamda
            lamda = (min_value + max_value)/2

    return lamda

##### 大于 threshold 为 ID data
def get_curve(known, novel):
    ##### 大于 threshold 为 ID data
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()   ### sort 函数返回：从小到大的数组
    novel.sort()

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    # if method == 'row':
    #     threshold = -0.5
    # else:
    threshold = known[round(0.05 * num_k)]
    print('threshold:', threshold)

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def cal_metric(known, novel):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)
    
        # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results
    

class MyDataset(Dataset):
    def __init__(self, data, label):
        # data_min = np.min(data, axis=-1).reshape(data.shape[0], 1)
        # data_max = np.max(data, axis=-1).reshape(data.shape[0], 1)
        # self.normalizer = lambda x: (x-data_min) / (data_max-data_min)
        # data = self.normalizer(data)
        
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label 

    def __len__(self):
        return self.data.shape[0]


def msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
          
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            
            logits = model(x)
            
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def mahalanobis(data_loader, mean, variance, num_classes=10, input_preprocess=False, magnitude=100):
    Mahalanobis = []
    for _, batch in enumerate(data_loader):
        data, target = batch
        data = data.cuda()
        target = target.cuda()

        if input_preprocess:
            for j in range(target.shape[0]):
                inputs = data[j]
                inputs = Variable(inputs, requires_grad = True)
                for i in range(num_classes):
                    distance = ((inputs-mean[i]).unsqueeze(0) @ variance @ (inputs-mean[i]).unsqueeze(1))
                    if i==0:
                        min_distance = distance
                    elif distance < min_distance:
                        min_distance = distance
                loss = min_distance
                loss.backward()

                gradient =  torch.ge(inputs.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2

                input_adv = torch.add(inputs, -magnitude*gradient)
                for i in range(num_classes):
                    distance = ((input_adv-mean[i]).unsqueeze(0) @ variance @ (input_adv-mean[i]).unsqueeze(1)).item()
                    if i==0:
                        min_distance = distance
                    elif distance < min_distance:
                        min_distance = distance
                Mahalanobis.append(min_distance)
        else:
            # compute Mahalanobis score
            for j in range(target.shape[0]):
                inputs = data[j]
                for i in range(num_classes):
                    distance = ((inputs-mean[i]).unsqueeze(0) @ variance @ (inputs-mean[i]).unsqueeze(1)).item()
                    if i==0:
                        min_distance = distance
                    elif distance < min_distance:
                        min_distance = distance
                Mahalanobis.append(min_distance)    
    return np.array(Mahalanobis)

def odin(data_loader, model, odin_temperature, odin_epsilon):
    score = []
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for i, batch in enumerate(data_loader):
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()

        inputs = Variable(inputs, requires_grad=True)
        outputs = model(inputs)
        # inputs = net.bn(inputs)
        # inputs = Variable(inputs, requires_grad=True)
        # print(inputs.max(), inputs.min())
        # print(fdsf)
        # outputs = net.fc(inputs)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / odin_temperature

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -odin_epsilon*gradient)
        tempInputs = Variable(tempInputs)

        outputs = model(tempInputs)

        outputs = outputs / odin_temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        score.extend(np.max(nnOutputs, axis=1))

    score = np.array(score)
    return score

def grad_norm(data_loader, model, gradnorm_temperature, num_classes=10, kl_loss=True, p_norm=1):
    score = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    # p_norm=1
    # print('p-norm:', p_norm)
    for i, batch in enumerate(data_loader):
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs = Variable(inputs, requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / gradnorm_temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        if kl_loss==False:
            # loss.backward()
            # layer_grad =get_model_grad_vec_torch_2(model)
            # layer_grad = model.fc.weight.grad.data
            layer_grad = torch.autograd.grad(outputs=loss, inputs=model.fc.weight, retain_graph=True)[0]
            # print(layer_grad.shape)
            layer_grad_norm = torch.norm(layer_grad, p=p_norm)
        elif kl_loss==True:
            layer_grad_norm = loss.detach()
        score.append(layer_grad_norm.cpu().numpy())
        # print(i)
    score = np.array(score)
    return score

def react(data_loader, model, temper, threshold, start_dim):
    score = []
    m = torch.nn.Softmax(dim=-1).cuda()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            ####### for cifar10
            inputs[:,start_dim:] = inputs[:,start_dim:].clip(max=threshold)
            ####### for imagenet
            # inputs[:,start_dim:] = inputs[:,start_dim:].clip(min=threshold)
            # inputs = inputs.clip(max=threshold)
            
            # inputs[:,start_dim:] = torch.where(inputs[:,start_dim:]<threshold, 0.0, inputs[:,start_dim:])
            # inputs = torch.where(inputs>threshold,threshold,inputs)

            # logits = model.fc(inputs)
            logits = model(inputs)
            
            conf = temper * (torch.logsumexp(logits / temper, dim=1)) 
            score.extend(conf.data.cpu().numpy())
    score = np.array(score) 
    return score

from visualize_utils import get_feas_by_hook
def obtain_feature_loader(model, dataloader, save_path, save_name, args):
    if not os.path.exists(save_path + save_name + '_foward_feature.npy'):
        forward_feature = []
        label = []
        total = 0
        for i, batch in enumerate(dataloader):
            if total > 50000:
                break
            
            inputs, targets = batch   
            inputs = inputs.cuda()
            targets = targets.cuda()

            total = total + targets.shape[0]
            # 定义提取中间层的 Hook
            if args.data == 'cifar10':
                fea_hooks = get_feas_by_hook(model, extract_module=['avg_pool'])
            elif args.data == 'imagenet':
                fea_hooks = get_feas_by_hook(model, extract_module=['avgpool'])

            output = model(inputs)

            # print(len(fea_hooks))
            
            features = fea_hooks[0].fea.squeeze()
            # print(features.shape)
            # print(fdsds)

            forward_feature.append(features.detach().cpu())
            label.append(targets.detach().cpu())

        forward_feature = torch.cat(forward_feature, 0)
        label = torch.cat(label, dim=0)
        forward_feature = forward_feature.numpy()
        label = label.numpy()
        print(forward_feature.shape, label.shape)
        np.save(save_path + save_name + '_foward_feature.npy', forward_feature)
        np.save(save_path + save_name + '_label.npy', label)
    else:
        forward_feature = np.load(save_path + save_name + '_foward_feature.npy')
        label = np.load(save_path + save_name + '_label.npy')
    
    dataset = MyDataset(forward_feature, label)
    feature_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    return forward_feature, label, feature_dataloader


##### feature在bn之前做bats 
class TrBN(nn.Module):
    def __init__(self, bn, lam, start_dim):
        super().__init__()
        self.bn = bn
        self.lam = lam
        self.sigma = bn.weight
        self.mu = bn.bias
        # print(bn.weight)
        # print(bn.bias)
        # print(fdsfsd)
        self.upper_bound = self.sigma*self.lam + self.mu
        self.lower_bound = -self.sigma*self.lam + self.mu
        self.start_dim = start_dim
        
    def forward(self, x):
        y = self.bn(x)
        upper_bound = self.upper_bound.view(1,self.upper_bound.shape[0])
        lower_bound = self.lower_bound.view(1,self.lower_bound.shape[0])
        # print(x.shape, y.shape, self.mu.shape, self.sigma.shape, self.bn.running_mean.shape, self.bn.running_var.shape)
        y[:,self.start_dim:] = torch.where(y[:,self.start_dim:]<upper_bound[:,self.start_dim:], y[:,self.start_dim:], upper_bound[:,self.start_dim:])
        y[:,self.start_dim:] = torch.where(y[:,self.start_dim:]>lower_bound[:,self.start_dim:], y[:,self.start_dim:], lower_bound[:,self.start_dim:])
        return y
    
    def get_static(self):
        return self.upper_bound, self.lower_bound
    
# 核心函数，参考了torch.quantization.fuse_modules()的实现
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def bats(data_loader, model, temper, upper_bound, lower_bound):
    score = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()

            # if args.data == 'cifar10':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['avg_pool'])
            # elif args.data == 'imagenet':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['avgpool'])
            # _ = model(inputs)
            # features = fea_hooks[0].fea.squeeze()
            # features = torch.where(features<upper_bound, features, upper_bound)
            # features = torch.where(features>lower_bound, features, lower_bound)
            # logits = model.fc(features)

            logits = model(inputs)
            
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            score.extend(conf.data.cpu().numpy())
    score = np.array(score) 
    return score

def knn(feat_log, feat_log_val, K=5):
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    # normalizer = lambda x: (x-np.expand_dims(x.min(1), axis=1))/np.expand_dims(x.max(1)-x.min(1), axis=1)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only

    ftrain = prepos_feat(feat_log)
    ftest = prepos_feat(feat_log_val)

    #################### KNN score OOD detection #################
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)
   
    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    
    return scores_in

def save_low_dim_grad_feature(data_loader, model, z_theta, I_2, p, kernel='NFK', save_dir='', save_name=''):
    reconstruct_error = []
    label = []
    num=0
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # with torch.enable_grad():
        #     inputs = adversary.perturb(inputs, targets)
        #     inputs = inputs.detach()

        grad_feature = cal_fisher_vector(model, inputs, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # grad_feature = cal_fisher_vector_per_sample(model, inputs[0], kernel=args.kernel, z_theta=z_theta, I_2=I_2).detach()
        # grad_feature = grad_feature.cpu()

        # distance = []
        # for j in range(classnums):
        #     distance.append(torch.norm(grad_feature-avg_feature_embedding[j], dim=1).unsqueeze(1))
        # distance = torch.cat(distance, dim=1)
        # min_distance, pred = torch.min(distance, dim=1)
        # num += (pred==targets).sum()

        # distance = []
        # for j in range(classnums):
        #     distance.append(torch.sum(grad_feature * avg_feature_embedding[j], dim=1).unsqueeze(1))
        # distance = torch.cat(distance, dim=1)
        # min_distance, pred = torch.max(distance, dim=1)
        # num += (pred==targets).sum()

        # reconstruct_error.append(min_distance)

        # reconstruct_grad_feature = reconstruct_gradient_feature_by_p(model, inputs, p, kernel=args.kernel, z_theta=z_theta, I_2=I_2).detach()
        # reconstruct_grad_feature = reconstruct_grad_feature.cpu()
        # reconstruct_error.append(torch.norm(grad_feature-reconstruct_grad_feature, dim=1, p=np.inf))

        # for j in range(inputs.shape[0]):
        #     re_grad = grad_feature[j] @ p @ p.T
        #     error = torch.norm(re_grad-grad_feature[j])
        #     # print(error)
        #     reconstruct_error.append(error.unsqueeze(0))

        # for j in range(inputs.shape[0]):
        #     influ_score = torch.dot(grad_feature[j], grad_feature[j])
        #     # print(influ_score)
        #     reconstruct_error.append(influ_score.unsqueeze(0))

        grad_feature = grad_feature.cpu()
        feature = grad_feature @ p
        reconstruct_error.append(feature)
        label.append(targets.cpu())

        # feature = grad_feature[:, -10000:-1].cpu()
        # reconstruct_error.append(feature)
        num = num + targets.shape[0]
        # print('num:', num)

    reconstruct_error = torch.cat(reconstruct_error, dim=0)
    label = torch.cat(label, dim=0)
    # gradients = torch.cat(gradients, dim=0)
    print('feature:', reconstruct_error.shape, 'label:', label.shape)
    # os.makedirs('./2023_04_17_adv_detect/'+ method + '/', exist_ok=True)
    np.save(save_dir + save_name + '_feature.npy', reconstruct_error.cpu().numpy())
    np.save(save_dir + save_name + '_label.npy', label.cpu().numpy())
    return

def save_avg_feature_embedding(data_loader, model, z_theta, I_2, kernel='NFK', classnums=10, save_dir=''):
    avg_feature_embedding = []
    for i in range(classnums):
        avg_feature_embedding.append(cal_avg_fisher_vector_per_class(model, data_loader, kernel=kernel, z_theta=z_theta, I_2=I_2, class_num=i).unsqueeze(0))
    avg_feature_embedding = torch.cat(avg_feature_embedding, dim=0)
    print('avg_feture_embedding: ', avg_feature_embedding.shape)
    np.save(save_dir + 'avg_feature_embedding.npy', avg_feature_embedding.cpu().detach().numpy())
    return

def test(data_loader, model):
    num = 0
    num_all = 0
    model.eval()
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
      
        predictions = model(inputs)

        correct = (torch.argmax(predictions, 1) == targets)
        num = num + (torch.nonzero(correct==True)).shape[0]
        num_all = num_all + targets.shape[0]
    acc = num/num_all
    print('test_acc:', acc)
    return acc

def train(model, optimizer, criterion, train_dataloader, test_dataloader, epochs=10, print_freq=100, save_dir='', save_name=''):
    for epoch in range(epochs):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        start = time.time()
        for i, batch in enumerate(train_dataloader):
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
        
            output = model(inputs)

            loss = criterion(output, targets)
            # loss = smooth_crossentropy(output, targets, smoothing=0.1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            prec1 = accuracy(output.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            if i % print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Time {3:.2f}'.format(
                    epoch, i, len(train_dataloader), end - start, losses=losses, top1=top1))
                start = time.time()

        print('train_accuracy {top1.avg:.3f}'.format(top1=top1))    
    
        model.eval()
        with torch.no_grad():
            losses_ce_test = AverageMeter()
            top1_test = AverageMeter()   
            start = time.time()
            for i, batch in enumerate(test_dataloader):
                inputs, targets = batch
                inputs = inputs.cuda()
                targets = targets.cuda()
            
                output = model(inputs)

                loss = criterion(output, targets)
                # loss = smooth_crossentropy(output, targets, smoothing=0.1).mean()
                
                output = output.float()
                loss_ce = loss.float()
                prec1 = accuracy(output.data, targets)[0]
                losses_ce_test.update(loss_ce.item(), inputs.size(0))
                top1_test.update(prec1.item(), inputs.size(0))

                if i % print_freq == 0:
                    end = time.time()
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Loss_ce {losses_ce.val:.4f} ({losses_ce.avg:.4f})\t'
                        'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Time {3:.2f}'.format(
                        epoch, i, len(test_dataloader), end - start, losses_ce=losses_ce_test, top1=top1_test))
                    start = time.time()

            print('test_accuracy {top1.avg:.3f}'.format(top1=top1_test))  
            torch.save(model.state_dict(), save_dir + save_name + '.pt')
    return

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def mahalanobis_official(model, train_loader, test_loader, ood_loader, num_classes, magnitude=0.01, save_path=None, save_name=None, data=None):
    # if not os.path.exists(save_path + save_name + '.npy'):
    if True:
        if data == 'cifar10':
            extract_module = ['equal']
        elif data == 'imagenet':
            extract_module = ['equal']
        sample_class_mean, variance = sample_estimator(model, num_classes, train_loader, extract_module)
        print('mean:', sample_class_mean[0].shape, 'var:', variance[0].shape)

        Mahalanobis_test = []
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_class_mean, variance, magnitude, extract_module)
            Mahalanobis_test.extend(Mahalanobis_scores)

        Mahalanobis_test = np.asarray(Mahalanobis_test, dtype=np.float32)
        print('Mahalanobis_test:', Mahalanobis_test.shape)

        Mahalanobis_ood = []
        for batch in ood_loader:
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_class_mean, variance, magnitude, extract_module)
            Mahalanobis_ood.extend(Mahalanobis_scores)

        Mahalanobis_ood = np.asarray(Mahalanobis_ood, dtype=np.float32)
        print('Mahalanobis_ood:', Mahalanobis_ood.shape)

        Mahalanobis_data, Mahalanobis_labels = merge_and_generate_labels(Mahalanobis_test, Mahalanobis_ood)
        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(save_path + save_name + '.npy', Mahalanobis_data)

        X_train = np.concatenate((Mahalanobis_test[:500], Mahalanobis_ood[1000:1500]))
        Y_train = np.concatenate((np.ones(Mahalanobis_test[:500].shape[0]), np.zeros(Mahalanobis_ood[1000:1500].shape[0])))
    else:
        total_X, total_Y = load_characteristics(file_name = save_path + save_name + '.npy')
        X_val, Y_val, X_test, Y_test = block_split(total_X, total_Y, data = data)
        X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
        Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
        # X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
        # Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
        # print(X_train.shape, X_val_for_test.shape)
        if data == 'cifar10':
            partition = 10000
        elif data == 'imagenet':
            partition = 50000
        Mahalanobis_test = total_X[:partition]
        Mahalanobis_ood = total_X[partition: :]
    
    # print(X_train.shape, Y_train.shape)
    # lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
    # regressor = lr
    # scores_in = regressor.predict_proba(Mahalanobis_test)[:, 1]
    # scores_out = regressor.predict_proba(Mahalanobis_ood)[:, 1]

    scores_in = Mahalanobis_test.flatten()
    scores_out = Mahalanobis_ood.flatten()

    return scores_in, scores_out

def load_omiga(omiga_name=[]):
    his_omiga = []
    for path in omiga_name:
        a = torch.from_numpy(np.load(path))
        his_omiga.append(a)
    his_omiga = torch.cat(his_omiga, 1)
    return his_omiga

def try_1(model, kernel, z_theta, I_2, p, p_norm, dataloader):
    reconstruct_error = []
    reconstruct_error_2 = []
    reconstruct_error_3 = []
    reconstruct_error_4 = []
    reconstruct_error_5 = []
    reconstruct_error_6 = []
    reconstruct_error_7 = []
    num=0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        grad_feature = cal_fisher_vector(model, inputs, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # print('grad_norm:', torch.mean(torch.norm(grad_feature, dim=1, p=p_norm)))

        grad_feature = grad_feature.cpu()
        # feature = grad_feature @ p @ p.T
        # coeff = grad_feature @ p
       
        # reconstruct_error.append(-torch.norm(grad_feature - feature, dim=1, p=p_norm))
        # reconstruct_error_2.append(torch.norm(feature, dim=1, p=p_norm))
        # reconstruct_error_7.append(torch.norm(coeff, dim=1, p=p_norm))
        # reconstruct_error_6.append(torch.norm(grad_feature, dim=1, p=p_norm))
        ## 单位化
        grad_feature = grad_feature / torch.norm(grad_feature, dim=1).unsqueeze(1)
        cos = grad_feature @ p
        # fea = cos @ p.T
        # reconstruct_error_3.append(torch.max(torch.abs(cos), dim=1)[0])
        # reconstruct_error_4.append(torch.sum(torch.abs(cos), dim=1))
        # reconstruct_error_5.append(-torch.norm(fea, dim=1, p=p_norm))
        reconstruct_error_5.append(torch.norm(cos, dim=1, p=p_norm))

        num = num + targets.shape[0]
        print('num:', num)
        # if num > 10:
        #     break

    # reconstruct_error = torch.cat(reconstruct_error, dim=0).numpy()
    # reconstruct_error_2 = torch.cat(reconstruct_error_2, dim=0).numpy()
    # reconstruct_error_3 = torch.cat(reconstruct_error_3, dim=0).numpy()
    # reconstruct_error_4 = torch.cat(reconstruct_error_4, dim=0).numpy()
    reconstruct_error_5 = torch.cat(reconstruct_error_5, dim=0).numpy()
    # reconstruct_error_6 = torch.cat(reconstruct_error_6, dim=0).numpy()
    # reconstruct_error_7 = torch.cat(reconstruct_error_7, dim=0).numpy()
    # print('reconstruct_error:', reconstruct_error.shape, reconstruct_error_2.shape, reconstruct_error_3.shape, reconstruct_error_4.shape)
    return reconstruct_error, reconstruct_error_2, reconstruct_error_3, reconstruct_error_4, reconstruct_error_5, reconstruct_error_6, reconstruct_error_7

### 每个类别的平均梯度组成降维矩阵
def cal_grad_per_class(model, dataloader, label, kernel, z_theta, I_2, save_dir):
    mean_v = 0
    sum_v = 0
    num = 0
    # print(len(dataset))
    # indices = [i for i in range(label*1270, min((label+1)*1300,len(dataset)))]
    # sub_dataset = torch.utils.data.Subset(dataset, indices)
    # dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=70, shuffle=False, num_workers=2)
    
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        indices = torch.nonzero(targets==label)
        previous_num = num
        if indices.shape[0]!=0:
            inputs = inputs[indices[:,0]]
            targets = targets[indices[:,0]]
            # print(targets[0])
            model.zero_grad()
            y = model(inputs)
            loss = torch.sum(y)
            loss.backward()
            sum_v = sum_v + get_model_grad_vec_torch_2(model)
            num = num + inputs.shape[0]
            print(targets[0], 'num:', num)
        # if num >=1280 or num == previous_num:
        # if num == previous_num and num!=0:
        #     break

    mean_v = sum_v/num  
    print('mean_v:', mean_v.shape)
    if kernel == 'NFK':
        mean_v =  mean_v - z_theta
        mean_v = I_2 * mean_v 
    mean_v = mean_v.detach().cpu()
    # np.save(save_dir +'imagenet/avg_grad_per_class/'+ str(label) +'.npy', mean_v.numpy())
    np.save(save_dir +'cifar10/avg_grad_per_class/'+ str(label) +'.npy', mean_v.numpy())
    return mean_v


def try_2(model, train_dataset, test_dataloader, kernel, z_theta, I_2, num_classes, save_dir, save_name):
    os.makedirs(save_dir+'imagenet/'+save_name,exist_ok=True)
    os.makedirs(save_dir+'imagenet/avg_grad_per_class/',exist_ok=True)
    knn_feature = []
    for i in range(999, num_classes):
    # for i in range(43):
        reconstruct_error = []
        if not os.path.exists(save_dir + 'imagenet/avg_grad_per_class/' + str(i) +'.npy'):
            avg_grad = cal_grad_per_class(model, train_dataset, i, kernel, z_theta, I_2, save_dir)
        else:
            avg_grad = torch.from_numpy(np.load(save_dir + 'imagenet/avg_grad_per_class/' +str(i) +'.npy'))
        num = 0
        for batch in test_dataloader:
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            grad_feature = cal_fisher_vector(model, inputs, kernel=kernel, z_theta=z_theta, I_2=I_2).detach().cpu()
            feature = grad_feature @ avg_grad.unsqueeze(1)
            reconstruct_error.append(feature)
            num = num + feature.shape[0]
            if num > 50000:
                break
            print(num)
        reconstruct_error = torch.cat(reconstruct_error, dim=0)
        np.save(save_dir + 'imagenet/' + save_name + str(i) + '.npy', reconstruct_error.numpy())
        knn_feature.append(reconstruct_error)
    knn_feature = torch.cat(knn_feature, dim=1)
    return knn_feature


######## for imagenet with 1000-dimension subspace 
def generate_feature(model, dataloader, z_theta, I_2, save_dir, save_name, args):
    if not os.path.exists(save_dir + save_name + '0_1000.npy'):
        for start_k in np.arange(0, 1000, 150):
            avg_grads = []
            end_k=start_k+150
            end_k = min(end_k, 1000)
            print('start_k:', start_k, 'end_k:', end_k)
            for i in range(start_k, end_k):
                avg_grad = torch.from_numpy(np.load(save_dir + 'imagenet/avg_grad_per_class/' +str(i) +'.npy'))
                avg_grads.append(avg_grad.unsqueeze(1))
            avg_grads = torch.cat(avg_grads, 1)
            avg_grads_norm = torch.norm(avg_grads, dim=0).unsqueeze(0)
            print('norm:', avg_grads_norm, avg_grads_norm.shape)
            avg_grads = avg_grads/avg_grads_norm

            ood_feature = []
            num = 0
            label = []
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.cuda()
                targets = targets.cuda()
                grad_feature = cal_fisher_vector(model, inputs, kernel=args.kernel, z_theta=z_theta, I_2=I_2).detach().cpu()
                # grad_feature = grad_feature / torch.norm(grad_feature, dim=1).unsqueeze(1)
                feature = grad_feature @ avg_grads
                ood_feature.append(feature)
                num = num + feature.shape[0]
                # print(save_name, ',num=', num)
                label.append(targets.detach().cpu())
            ood_feature = torch.cat(ood_feature, dim=0).numpy()
            os.makedirs(save_dir + save_name, exist_ok=True)
            np.save(save_dir + save_name + str(start_k) + '_' + str(end_k) + '.npy', ood_feature)
            print(save_name, ood_feature.shape) 
            label = torch.cat(label, dim=0).numpy()
            print('label:', label.shape, label[0:10])
            np.save(save_dir + save_name + 'label.npy', label)
            
        ood_knn_feature=[]
        for start_k in np.arange(0, 1000, 150):
            end_k=start_k+150
            end_k = min(end_k, 1000)
            print('start_k:', start_k, 'end_k:', end_k)
            ood_knn_feature.append(torch.from_numpy(np.load(save_dir + save_name + str(start_k) + '_' + str(end_k) + '.npy')))
        ood_knn_feature = torch.cat(ood_knn_feature, 1).numpy()
        print(ood_knn_feature.shape)
        np.save(save_dir + save_name + '0_1000.npy', ood_knn_feature)
    else:
        ood_knn_feature = np.load(save_dir + save_name + '0_1000.npy')

    return ood_knn_feature

def generate_cos_feature(model, dataloader, z_theta, I_2, save_dir, save_name, args):
    if not os.path.exists(save_dir + save_name + '0_1000_cos.npy'):
        feat_log_val = torch.from_numpy(np.load(save_dir + save_name + '0_1000.npy'))
        num = 0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            grad_feature = cal_fisher_vector(model, inputs, kernel=args.kernel, z_theta=z_theta, I_2=I_2).detach().cpu()

            grad_feature_norm = torch.norm(grad_feature, dim=1).unsqueeze(1)

            feat_log_val[num:num+inputs.shape[0]] = feat_log_val[num:num+inputs.shape[0]]/grad_feature_norm

            num = num + inputs.shape[0]
            print(save_name, ',num=', num)

        feat_log_val = feat_log_val.numpy()
        np.save(save_dir + save_name + '0_1000_cos.npy', feat_log_val)
        print(save_name, feat_log_val.shape) 
    else:
        feat_log_val = np.load(save_dir + save_name + '0_1000_cos.npy')

    return feat_log_val


from visualize_utils import get_feas_by_hook
def obtain_feature_loader(model, dataloader, save_path, save_name, args):
    # for n, m in model.named_modules():
    #     print('name:', n)
    # print(fsdfs)
    # if not os.path.exists(save_path + save_name + '_foward_feature.npy'):
    if True:
        forward_feature = []
        label = []
        total = 0
        for i, batch in enumerate(dataloader):
            # if total > 50000:
            #     break
            
            inputs, targets = batch   
            inputs = inputs.cuda()
            targets = targets.cuda()

            total = total + targets.shape[0]
            # 定义提取中间层的 Hook
            if args.data == 'cifar10':
                fea_hooks = get_feas_by_hook(model, extract_module=['model.avg_pool'])
            elif args.data == 'imagenet':
                fea_hooks = get_feas_by_hook(model, extract_module=['model.avgpool'])

            output = model(inputs)

            # print(len(fea_hooks))
            
            features = fea_hooks[0].fea.squeeze()
            # print(features.shape)
            # print(fdsds)

            forward_feature.append(features.detach().cpu())
            label.append(targets.detach().cpu())

        forward_feature = torch.cat(forward_feature, 0)
        label = torch.cat(label, dim=0)
        forward_feature = forward_feature.numpy()
        label = label.numpy()
        print(forward_feature.shape, label.shape)
        np.save(save_path + save_name + '_foward_feature.npy', forward_feature)
        np.save(save_path + save_name + '_label.npy', label)
    else:
        forward_feature = np.load(save_path + save_name + '_foward_feature.npy')
        label = np.load(save_path + save_name + '_label.npy')
    
    dataset = MyDataset(forward_feature, label)
    feature_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    return forward_feature, label, feature_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    #parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of gpus.")
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--distort_grad", default=False, type=bool, help="True if you want to distort parameter adversarial noise.")
    parser.add_argument("--data", default='cifar10', type=str, help="cifar10 or cifar100.")
    parser.add_argument('--seed', default=42, type=int, help='randomize seed')
    parser.add_argument("--reweight_loss", default=False, type=bool, help="True if you want to reweight loss.")
    parser.add_argument("--method", default='distort_grad', type=str, help="version of WSAM")
    parser.add_argument("--model", default='resnet18', type=str, help="resnet18 or wideresnet or VGG16BN")
    parser.add_argument("--save_name", default='1', type=str, help="index of model")
    parser.add_argument("--kernel", default='NTK', type=str, help="NTK or NFK")
    parser.add_argument('--k', default=128, type=int, help='dimension reduce')
    parser.add_argument('--sample_num', default=1000, type=int, help='sample number of fisher kernel')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
    parser.add_argument("--base_method", default='knn', type=str, help="baseline method for detection")
    parser.add_argument("--ood_data", default='SVHN', type=str, help="places365, dtd, SVHN, iSUN, LSUN, places50, sun50, inat")
    parser.add_argument("--trans", default=False, type=str2bool, help="True or False")
    parser.add_argument("--ifbn", default=True, type=str2bool, help="True or False")
    parser.add_argument('--a', default=0, type=float, help='coefficient of combination with forward information')
    args = parser.parse_args()

    file_name=os.path.basename(__file__).split(".")[0]
    save_path = file_name + '/' + args.model + '_' + args.data + '_best_rho=' + str(args.rho) + '_labsmooth='+str(args.label_smoothing) + '/' 
    os.makedirs(save_path, exist_ok=True)
    sys.stdout = Logger(save_path + 'output.log')
    
    initialize(args, seed=args.seed)
    if args.num_gpus==1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.num_gpus>1:
        args.local_rank = int(os.environ["RANK"])
        local_world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            #world_size=args.num_gpus,
            world_size=local_world_size,
            rank=args.local_rank)
        setup_for_distributed(args.local_rank==0)

    if args.data == 'mnist':
        dataset = MNIST(args.batch_size, args.threads, args.num_gpus)
    else:
        dataset = IN_DATA(args.data, args.batch_size, args.threads, args.num_gpus)
    
    
    if args.model == 'vgg16bn':
        model = VGG16BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'vgg19bn':
        model = VGG19BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'resnet18':
        model = resnet18(10 if args.data == 'cifar10' else 100)
    elif args.model == 'wideresnet':
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10 if args.data == 'cifar10' else 100)
    elif args.model == 'lenet':
        model = LeNet()
    elif args.model == 'resnet50':
        import torchvision
        from torchvision import models
        # model = models.resnet50(pretrained=True)
        model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)


    if args.num_gpus==1:
        model.cuda()
    elif args.num_gpus>1:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model,
                    device_ids=[args.local_rank] 
                )
        
    if args.data == 'mnist':
        checkpoint = torch.load('2023_03_06_compare_sgd/lenet_mnist_best_rho=0.05_labsmooth=0.0/seed_42.pt')
        print('epoch:', checkpoint['epoch'])
        model.load_state_dict(checkpoint['model'], strict=False)
    elif args.data == 'cifar10':
        checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar10_best_rho=0.05_labsmooth=0.1/seed_42.pt')
        print('epoch:', checkpoint['epoch'])
        model.load_state_dict(checkpoint['model'], strict=False)
        model.set_nor(False) 
    elif args.data == 'cifar100':
        # checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar100_best_rho=0.05.pt')
        # print('epoch:', checkpoint['epoch'])
        # model.load_state_dict(checkpoint['model'], strict=False)
        # model.set_nor(False)
        checkpoint = torch.load('weights/SGD_resnet18_cifar100_labsmooth=0.1/epoch199.pt')
        print('epoch:', checkpoint['epoch']) 
        model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    ######################## eval
    # test(dataset.test, model)
    
    ######################## load z_theta, I_2, p
    model = Energy_Model(model)
    model.eval()
    # if args.data == 'cifar10':
    #     # save_dir = '2023_04_17_adv_detect_2/' + args.model + '_' + args.data + '_' + args.kernel +'/'
    #     save_dir = './2023_05_26_cal_p_fast/resnet18_cifar10_NFK/sample_num=49998_k=512/'
    # elif args.data == 'imagenet':
    #     save_dir =  '2023_04_18_cal_p_imagenet/resnet18_cifar10_NFK/sample_num=50000_k=128/'
    # elif args.data == 'cifar100':
    #     save_dir =  './2023_05_26_cal_p_fast/resnet18_cifar100_NFK/sample_num=50000_k=128/'
    # elif args.data == 'mnist':
    #     save_dir =  './2023_05_26_cal_p_fast/lenet_mnist_NFK/sample_num=60000_k=2/'
    # # z_theta = torch.from_numpy(np.load(save_dir + 'z_theta.npy')).cuda()
    # # I_2 = torch.from_numpy(np.load(save_dir + 'I_2.npy')).cuda()
    # # p = torch.from_numpy(np.load(save_dir + 'p.npy'))#.cuda()

    # # p_name = []
    # # p_name.append('./2023_05_26_cal_p_fast/resnet18_cifar10_NFK/sample_num=49998_k=512/0-128.npy')
    # # p_name.append('./2023_05_26_cal_p_fast/resnet18_cifar10_NFK/sample_num=49998_k=512/128-256.npy')
    # # p_name.append('./2023_05_26_cal_p_fast/resnet18_cifar10_NFK/sample_num=49998_k=512/256-263.npy')
    # # p = load_omiga(p_name)
    # # print('p:', p.shape)
    # # print('finish load p')


    if args.data == 'cifar10' or args.data == 'mnist':
        classnums = 10 
    elif args.data == 'imagenet':
        classnums = 1000
    elif args.data == 'cifar100':
        classnums = 100

    # reduce_k = 200
    # reduce_k = 128
    # reduce_k = 1000
    # reduce_k = p.shape[1]
    # method = '5_gradient_feature_knn_reduceK='+str(reduce_k)
    # method = args.data + '_reduceK=' + str(reduce_k)
    # method = args.data + '_reduceK=' + str(reduce_k) + '/forward'
    # method = args.data + '_p=263' #+ st/r(p.shape[1])
    # print('reduce_K=', reduce_k)
    if args.data == 'imagenet':
        reduce_k = 1000
        method = args.data + '_p=1000'
    else:
        reduce_k = 200
        method = args.data + '_p=263' #+ st/r(p.shape[1])
    # print('reduce_K=', reduce_k)
    # method = args.data + '_p=128'
    save_dir = file_name + '/' + method + '/'
    # save_dir = file_name + '/cifar10_reduceK=128/forward/'
    # save_dir = './2023_05_31_detect_base_method/compare_method/'
    os.makedirs(save_dir, exist_ok=True)
    
    # p = torch.from_numpy(np.load(save_dir + 'p.npy'))
    # avg_grads = []
    # for i in range(10):
    #     avg_grad = torch.from_numpy(np.load(save_dir + 'cifar10/avg_grad_per_class/' +str(i) +'.npy'))
    #     avg_grads.append(avg_grad.unsqueeze(1))
    # avg_grads = torch.cat(avg_grads, 1)
    # avg_grads_norm = torch.norm(avg_grads, dim=0).unsqueeze(0)
    # print('norm:', avg_grads_norm, avg_grads.shape)
    # avg_grads = avg_grads/avg_grads_norm
    # p = avg_grads
    # np.save(save_dir + 'p.npy', p.numpy())


    ############# save low-dimensional gradient feature
    # if not os.path.exists(save_dir + args.data + '_test_feature.npy'):
    #     save_low_dim_grad_feature(dataset.test, model, z_theta, I_2, p, kernel=args.kernel, save_dir=save_dir, save_name = args.data + '_test')
    # if not os.path.exists(save_dir + args.data + '_train_feature.npy'):
    #     save_low_dim_grad_feature(dataset.train, model, z_theta, I_2, p, kernel=args.kernel, save_dir=save_dir, save_name = args.data + '_train')
    # print('trainset trans')
    
    ood_data = args.ood_data 
    print('ood_data:', ood_data)
    # loader_test_dict = get_loader_out(args, dataset=(None, ood_data), split=('val'))
    # out_loader = loader_test_dict.val_ood_loader

    # if not os.path.exists(save_dir + args.data + '_' + ood_data + '_feature.npy'):
    #     save_low_dim_grad_feature(out_loader, model, z_theta, I_2, p, kernel=args.kernel, save_dir=save_dir, save_name=args.data + '_' + ood_data)
    
    # def get_label(dataloader, save_path, save_name):
    #     label = []
    #     for i, batch in enumerate(dataloader):
    #         inputs, targets = batch    
    #         targets = targets.cuda()
    #         label.append(targets.detach().cpu())

    #     label = torch.cat(label, dim=0)
    #     label = label.numpy()
    #     np.save(save_path + save_name + '_label.npy', label)
    #     return 
    
    # get_label(out_loader, save_dir, args.data + '_' + ood_data)
    # print(fsdfds)
    
    save_dir = '2023_05_24_detect/imagenet_p=1000/'
    feat_log = np.load(save_dir +  args.data + '_train_feature.npy')[:,0:reduce_k]
    feat_log_label = np.load(save_dir +  args.data + '_train_label.npy')
    # print(feat_log_label[0:10])
    feat_log_val = np.load(save_dir +  args.data + '_test_feature.npy')[:,0:reduce_k]
    feat_log_val_label = np.load(save_dir +  args.data + '_test_label.npy')
    # print(feat_log_val_label[0:10])
    ood_feat_log = np.load(save_dir + args.data + '_' + ood_data + '_feature.npy')[:,0:reduce_k]
    ood_feat_log_label = np.load(save_dir + args.data + '_' + ood_data + '_label.npy')
    # print(ood_feat_log_label[0:10])
    print(feat_log.shape, feat_log_val.shape, ood_feat_log.shape)
    

    def cal_threshold(feature, percent):
        feature = feature.flatten()
        threshold = np.percentile(feature, percent*100) # percent的数小于threshold
        # print((feature<threshold).sum())
        return threshold
    percent = 0.90
    threshold = cal_threshold(feat_log_val, percent=percent)
    print('threshold:', threshold, ',percent:', percent)

    feat_log_val = torch.from_numpy(feat_log_val)
    ood_feat_log = torch.from_numpy(ood_feat_log)

    def ratio(inputs, threshold):
        num_of_value = int(inputs.shape.numel())
        num_of_abnormal_value = (inputs>threshold).sum()
        num_of_abnormal_value = int(num_of_abnormal_value)
        abnormal_value_ratio = num_of_abnormal_value/num_of_value
        return abnormal_value_ratio
    
    print('test:', ratio(feat_log_val, threshold))
    print('ood:', ratio(ood_feat_log, threshold))
    print(fsdfsd)

    ############# load low-dimensional gradient feature
    # method = '5_gradient_feature_knn_reduceK=128'
    # save_dir = file_name + '/' + method + '/'
    # generate_feature(model, out_loader, z_theta, I_2, save_dir, ood_data+'/', args)
    # print(fsdfs)
    ################################################## hyper-parameter study ################################################
    # reduce_k = 200
    # FPR=[]
    # AUROC=[]
    # # for reduce_k in range(10, 260, 20):
    # for reduce_k in range(50, 1001, 50):
    # # for temper in np.arange(0, 3, 0.33):
    # #     temper = 10**temper
    # # for percent in np.arange(0.5, 0.6, 0.01):
    # # for lam in [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]:
    # # for K in [1, 5, 10, 50, 100, 500, 1000]:
    #     avg_fpr = 0
    #     avg_auroc = 0
    #     # for ood_data in ['SVHN','dtd','LSUN','iSUN']:
    #     for ood_data in ['inat','sun50','places50','dtd']:
    #         feat_log = np.load(save_dir +  args.data + '_train_feature.npy')[:,0:reduce_k]
    #         feat_log_label = np.load(save_dir +  args.data + '_train_label.npy')
    #         # print(feat_log_label[0:10])
    #         feat_log_val = np.load(save_dir +  args.data + '_test_feature.npy')[:,0:reduce_k]
    #         feat_log_val_label = np.load(save_dir +  args.data + '_test_label.npy')
    #         # print(feat_log_val_label[0:10])
    #         ood_feat_log = np.load(save_dir + args.data + '_' + ood_data + '_feature.npy')[:,0:reduce_k]
    #         ood_feat_log_label = np.load(save_dir + args.data + '_' + ood_data + '_label.npy')
    #         # print(ood_feat_log_label[0:10])
    #         print(feat_log.shape, feat_log_val.shape, ood_feat_log.shape)
            
    #         ############## train Linear Probe
    #         # ifbn = args.ifbn
    #         # net = Linear_Probe(reduce_k, classnums, ifbn).cuda()
    #         # if ifbn == True:
    #         #     save_name = args.data+'_linear_bn_' + str(reduce_k)
    #         # else:
    #         #     save_name = args.data+'_linear_' + str(reduce_k)
    #         # net.load_state_dict(torch.load(save_dir + save_name + '.pt'))
    #         # net.eval()
            
    #         # train_set = MyDataset(feat_log, feat_log_label)
    #         # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    #         # test_set = MyDataset(feat_log_val, feat_log_val_label)
    #         # test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    #         # ood_set = MyDataset(ood_feat_log, ood_feat_log_label)
    #         # ood_dataloader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    #         # # test(test_dataloader, net)
            
    #         ifbn = args.ifbn
    #         net = Linear_Probe(reduce_k, classnums, ifbn).cuda()
    #         optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    #         criterion = nn.CrossEntropyLoss()
    #         train_set = MyDataset(feat_log, feat_log_label)
    #         train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    #         test_set = MyDataset(feat_log_val, feat_log_val_label)
    #         test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
            
    #         print('ifbn:', ifbn)
    #         if ifbn == True:
    #             save_name = args.data+'_linear_bn_' + str(reduce_k)
    #         else:
    #             save_name = args.data+'_linear_' + str(reduce_k)
    #         if not os.path.exists(save_dir + save_name + '.pt'):
    #             train(net, optimizer, criterion, train_dataloader, test_dataloader, epochs=10, print_freq=100, save_dir=save_dir, save_name=save_name)
    #         print('finish train')

    #         ################# start detection
    #         base_method = args.base_method
    #         print('base_method:', base_method)
    #         ood_set = MyDataset(ood_feat_log, ood_feat_log_label)
    #         ood_dataloader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
            
    #         net.load_state_dict(torch.load(save_dir + save_name + '.pt'))
    #         net.eval()
    #         test(test_dataloader, net)
            
    #         ###########################################################################################################
    #         if base_method == 'msp':
    #             with torch.no_grad():
    #                 confs = msp(test_dataloader, net)
    #                 ood_confs =  msp(ood_dataloader, net)
    #         elif base_method == 'energy':
    #             temper = 1
    #             with torch.no_grad():
    #                 confs = energy(test_dataloader, net, temper)
    #                 ood_confs = energy(ood_dataloader, net, temper)
    #         elif base_method == 'maha':
    #             num_classes = classnums       
    #             confs, ood_confs = mahalanobis_official(net, train_dataloader, test_dataloader, ood_dataloader, num_classes, magnitude=0.0, save_path=save_dir, save_name='mahalanobis_'+args.data+'_'+args.ood_data, data=args.data)
    #         elif base_method == 'odin':
    #             if args.data =='cifar10':
    #                 odin_epsilon = 50
    #             elif args.data == 'imagenet':
    #                 odin_epsilon = 0.005
    #             # odin_temperature = 1000
    #             odin_temperature = 100
    #             print('eps:', odin_epsilon, 'temper:', odin_temperature)
    #             confs = odin(test_dataloader, net, odin_temperature, odin_epsilon)
    #             ood_confs = odin(ood_dataloader, net, odin_temperature, odin_epsilon)
    #         elif base_method == 'react':
    #             temper = 1
    #             def cal_threshold(feature, percent):
    #                 feature = feature.flatten()
    #                 threshold = np.percentile(feature, percent*100) # percent的数小于threshold
    #                 return threshold
    #             if args.data == 'cifar10':
    #                 percent = 0.90
    #                 if reduce_k == 10:
    #                     start_dim = 0
    #                 else:
    #                     start_dim = 10
    #             elif args.data == 'imagenet':
    #                 percent = 0.70
    #                 if reduce_k <=950:
    #                     start_dim = 0
    #                 else:
    #                     start_dim = 950
               
    #             threshold = cal_threshold(feat_log_val[:,start_dim:], percent=percent)
    #             print('threshold:', threshold, ',percent:', percent)
    #             confs = react(test_dataloader, net, temper, threshold, start_dim)
    #             ood_confs = react(ood_dataloader, net, temper, threshold, start_dim)
    #         elif base_method == 'bats':
    #             temper = 1
    #             lams = np.arange(0.1, 0.11, 0.1)
    #             for lam in lams:
    #                 net2 = copy.deepcopy(net)
    #                 if args.data =='cifar10':
    #                     truncated_module = ['bn'] 
    #                     if reduce_k == 10:
    #                         start_dim = 0
    #                     else:
    #                         start_dim = 10
    #                 elif args.data == 'imagenet':
    #                     truncated_module = ['bn']
    #                     if reduce_k <=950:
    #                         start_dim = 0
    #                     else:
    #                         start_dim = 950
                    
    #                 print('lam:', lam, ',bn_module:', truncated_module)
    #                 for n, module in net2.named_modules():
    #                         if n in truncated_module:
    #                             Trunc_BN = TrBN(module, lam, start_dim)
    #                             _set_module(net2, n, Trunc_BN)
    #                             upper_bound, lower_bound = Trunc_BN.get_static()
    #                 net2.eval()
    #                 confs = bats(test_dataloader, net2, temper, upper_bound, lower_bound)
    #                 ood_confs = bats(ood_dataloader, net2, temper, upper_bound, lower_bound)
    #                 print(confs.shape, ood_confs.shape)
    #                 results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
    #                 print('lam:', lam, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])       
    #         elif base_method == 'knn':
    #             if args.data == 'cifar10':
    #                 K=5
    #             elif args.data == 'imagenet':
    #                 K=10
    #             confs = knn(feat_log, feat_log_val, K) 
    #             ood_confs = knn(feat_log, ood_feat_log, K) 
                
    #         ###########################################################################################################
    #         # temper = 1 
    #         # net2 = copy.deepcopy(net)
    #         # truncated_module = ['bn']
    #         # start_dim = 950
    #         # for n, module in net2.named_modules():
    #         #         if n in truncated_module:
    #         #             Trunc_BN = TrBN(module, lam, start_dim)
    #         #             _set_module(net2, n, Trunc_BN)
    #         #             upper_bound, lower_bound = Trunc_BN.get_static()
    #         # net2.eval()
    #         # confs = bats(test_dataloader, net2, temper, upper_bound, lower_bound)
    #         # ood_confs = bats(ood_dataloader, net2, temper, upper_bound, lower_bound)
    #         ###########################################################################################################
    #         # temper = 1
    #         # def cal_threshold(feature, percent):
    #         #     feature = feature.flatten()
    #         #     threshold = np.percentile(feature, percent*100) # percent的数小于threshold
    #         #     return threshold
    #         # start_dim = 950
    #         # threshold = cal_threshold(feat_log_val[:,start_dim:], percent=percent)
    #         # print('threshold:', threshold, ',percent:', percent)
    #         # confs = react(test_dataloader, net, temper, threshold, start_dim)
    #         # ood_confs = react(ood_dataloader, net, temper, threshold, start_dim)
    #         ###########################################################################################################
    #         # temper = 1
    #         # with torch.no_grad():
    #         #     confs = energy(test_dataloader, net, temper)
    #         #     ood_confs = energy(ood_dataloader, net, temper)
    #         ###########################################################################################################
    #         # K=10
    #         # K=5
    #         # confs = knn(feat_log, feat_log_val, K) #+ knn(train_forward_feature, test_forward_feature, K)
    #         # ood_confs = knn(feat_log, ood_feat_log, K) #+ knn(train_forward_feature, ood_forward_feature, K)
    #         ###########################################################################################################
    #         results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            
    #         avg_fpr += 100*results['FPR']
    #         avg_auroc += 100*results['AUROC']
            
    #     avg_fpr = round(avg_fpr/4, 2)
    #     avg_auroc = round(avg_auroc/4, 2)
    #     FPR.append(avg_fpr)
    #     AUROC.append(avg_auroc)
    #     print('Reduce_K:', reduce_k, ',AVG_FPR:', avg_fpr, ',AVG_AUROC:', avg_auroc)
    #     # print('Temp:', temper, ',AVG_FPR:', avg_fpr, ',AVG_AUROC:', avg_auroc)
    #     # print('Percent:', percent, ',AVG_FPR:', avg_fpr, ',AVG_AUROC:', avg_auroc)
    #     # print('Lam:', lam, ',AVG_FPR:', avg_fpr, ',AVG_AUROC:', avg_auroc)
    #     # print('K:', K, ',AVG_FPR:', avg_fpr, ',AVG_AUROC:', avg_auroc)
    # print(FPR)
    # print(AUROC)
    # print(fsdfsad)
    #######################################################################################################################
    
    # if args.data == 'cifar10':
    #     classnums = 10 
    # elif args.data == 'imagenet':
    #     classnums = 1000
    # ood_data = args.ood_data 
    # save_dir = './2023_05_31_detect_base_method/compare_method/'
 
    # feat_log = np.load(save_dir +  args.data + '_train_foward_feature.npy') 
    # feat_log_label = np.load(save_dir +  args.data + '_train_label.npy')
    # print(feat_log_label[0:10])

    # feat_log_val = np.load(save_dir +  args.data + '_test_foward_feature.npy')
    # feat_log_val_label = np.load(save_dir +  args.data + '_test_label.npy')
    # print(feat_log_val_label[0:10])
    
    # ood_feat_log = np.load(save_dir + args.data + '_' + ood_data + '_foward_feature.npy')
    # ood_feat_log_label = np.load(save_dir + args.data + '_' + ood_data + '_label.npy')
    # print(ood_feat_log_label[0:10])
    # print(fsdfs)
    
    # for i, batch in enumerate(out_loader): # dataset.train, out_loader
    #     inputs, targets = batch    
    #     print(targets[0:10])
    #     print(fdsfs)

    # with torch.no_grad():
    #     # train_forward_feature, train_label, feature_train_dataloader = obtain_feature_loader(model, dataloader, save_dir, args.data+'_train', args)
    #     ood_forward_feature, ood_label, feature_ood_dataloader = obtain_feature_loader(model, out_loader, save_dir, args.data +'_'+ args.ood_data, args)
    # print(fsdfs)
    
    # feat_log_val = abs(feat_log_val)
    # ood_feat_log = abs(ood_feat_log)
    # print(abs(np.mean(feat_log_val, axis=0)).mean())
    # print(abs(np.mean(ood_feat_log, axis=0)).mean())
    # print(fdsfds)
    
    ###### forward feature principle component calculation
    # from sklearn.covariance import EmpiricalCovariance
    # ec = EmpiricalCovariance(assume_centered=True)
    # ec.fit(feat_log)
    # eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    # NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[0:classnums]]).T)
    # print(NS.shape)
    # feat_log_val = feat_log_val @ NS
    # ood_feat_log = ood_feat_log @ NS
    
    ######################## plot density distribution figure ##################################################
    # net = Linear_Probe(reduce_k, classnums, ifbn=args.ifbn).cuda()
    # save_name = args.data + '_linear_bn_' + str(reduce_k)
    # net.load_state_dict(torch.load(save_dir + save_name + '.pt'))
    # net.eval()
    
    # train_set = MyDataset(feat_log, feat_log_label)
    # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    # test_set = MyDataset(feat_log_val, feat_log_val_label)
    # test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    # ood_set = MyDataset(ood_feat_log, ood_feat_log_label)
    # ood_dataloader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    
    # def get_feature(dataloader, model):
    #     feature = []
    #     for i, batch in enumerate(dataloader):
    #         inputs, targets = batch    
    #         inputs = inputs.cuda()
    #         targets = targets.cuda()
    #         feature.append(model.bn(inputs).detach().cpu())

    #     feature = torch.cat(feature, dim=0)
    #     feature = feature.numpy()
    #     return feature
    
    # feat_log_val = get_feature(test_dataloader, net)
    # ood_feat_log = get_feature(ood_dataloader, net)
    # print(feat_log_val.shape, ood_feat_log.shape)
    
    # import pandas as pd
    # import seaborn as sns
    # import random
    # sns.set()
    # # channel = 0
    # index = random.sample(range(0, feat_log_val.shape[0]), ood_feat_log.shape[0])
    # for channel in range(991, 992):
    #     # test_x = feat_log_val[:,channel]
    #     test_x = feat_log_val[index[:], channel]
    #     ood_x = ood_feat_log[:, channel]
    #     print(test_x.shape, ood_x.shape)
    #     # print(test_x.mean(), ood_x.mean(), test_x.min(), ood_x.min(), test_x.max(), ood_x.max())
        
    #     # test_x.dtype = np.float32
    #     # ood_x.dtype = np.float32
    #     # print(test_x.mean(), ood_x.mean(), test_x.min(), ood_x.min(), test_x.max(), ood_x.max())
        
    #     f_list_clean = [test_x[j] for j in range(test_x.shape[0])]
    #     f_list_adv = [ood_x[j] for j in range(ood_x.shape[0])]
    #     f_list = f_list_clean + f_list_adv
    #     c_list_clean = ['test' for j in range(test_x.shape[0])]
    #     c_list_adv = ['ood' for j in range(ood_x.shape[0])]
    #     c_list = c_list_clean + c_list_adv
    #     dict = {'type':c_list, 'feature':f_list}
    #     data = pd.DataFrame(dict)
        
    #     sns.displot(data, x='feature', hue='type', color='r', alpha=0.4, legend=False)#, kde=True)
    #     sns.displot(data, x='feature', hue='type', color='b', alpha=0.4, legend=False)#, kde=True)
        
    #     # sns.kdeplot(data, x='feature', hue='type', color='r', alpha=0.3, fill=True, bw_adjust=0.5)
    #     # sns.kdeplot(data, x='feature', hue='type', color='b', alpha=0.3, fill=True, bw_adjust=0.5)                                                                                            
        
    #     # plt.savefig(save_dir + 'distribution_bn/channel_'+str(channel)+'.png', dpi=600)
    #     plt.legend(labels=['ood','test'])
    #     plt.savefig(save_dir + 'distribution_bn_pdf/channel_'+str(channel)+'.pdf', format='pdf', bbox_inches='tight')
    #     plt.savefig(save_dir + 'distribution_bn_pdf/channel_'+str(channel)+'.png', dpi=800)
    #     plt.clf()
    # print(fdsfs)
    
    ################ visualize low-dimensional gradient distribution ############################################
    ###### forward feature principle component calculation
    # from sklearn.covariance import EmpiricalCovariance
    # ec = EmpiricalCovariance(assume_centered=True)
    # ec.fit(feat_log)
    # eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    # NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[0:classnums]]).T)
    # print(NS.shape)
    # feat_log_val = feat_log_val @ NS
    # ood_feat_log = ood_feat_log @ NS
   
    # from sklearn.manifold import TSNE
    # fig_save_dir = 'low-dimensional feature visualization/' + 'forward' + '/'
    # fig_save_dir = 'low-dimensional feature visualization/' + 'gradient' + '/PCA/'

    # def scatter_plot(x, y, z, save_name):
    #     plt.clf()
    #     plt.scatter(x[:,0], x[:,1], c='#82B0D2', alpha=1, marker='.', label='Train')
    #     plt.scatter(y[:,0], y[:,1], c='#FFBE7A', alpha=1, marker='.', label='Test')
    #     plt.scatter(z[:,0], z[:,1], c='#FA7F6F', alpha=1, marker='.', label='Outlier')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(fig_save_dir+save_name+'.png', dpi=800)
    #     return
    
    # def scatter_plot_single(x, save_name):
    #     plt.clf()
    #     plt.scatter(x[:,0], x[:,1], c='#82B0D2', alpha=1, marker='.')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(fig_save_dir+save_name+'.png', dpi=800)
    #     return
    
    # # if not os.path.exists(fig_save_dir + args.data +'_' + args.ood_data +'.npy'):
    # if True:
    #     # normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    #     # feat_log = normalizer(feat_log)
    #     # feat_log_val = normalizer(feat_log_val)
    #     # ood_feat_log = normalizer(ood_feat_log)

    #     # feat = np.concatenate((feat_log,feat_log_val,ood_feat_log), axis=0)
    #     # print(feat.shape)
    #     # ts = TSNE(n_components=2, init='pca', random_state=0)
    #     # x_ts = ts.fit_transform(feat)
    #     # x = x_ts[0:50000]  # train
    #     # y = x_ts[50000:60000] # test
    #     # z = x_ts[60000:70000] # ood
        
    #     from sklearn.covariance import EmpiricalCovariance
    #     ec = EmpiricalCovariance(assume_centered=True)
    #     ec.fit(feat_log)
    #     eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    #     NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[0:classnums]]).T)
    #     # print(NS.shape)
    #     x = feat_log @ NS
    #     y = feat_log_val @ NS
    #     z = ood_feat_log @ NS
    
    # #     np.save(fig_save_dir+args.data+'_train.npy', x)
    # #     np.save(fig_save_dir+args.data+'_test.npy', y)
    # #     np.save(fig_save_dir+ args.data +'_' + args.ood_data +'.npy', z)
    # # else:
    # #     x = np.load(fig_save_dir+args.data+'_train.npy')
    # #     y = np.load(fig_save_dir+args.data+'_test.npy')
    # #     z = np.load(fig_save_dir+ args.data +'_' + args.ood_data +'.npy')

    # # scatter_plot(x[0:2000], y[0:500], z[0:500], args.data + '_' + args.ood_data)
    # # scatter_plot(x, y[0:2000], z[0:2000], args.data + '_' + args.ood_data)
    
    # # scatter_plot(x, y[0:2000], z[0:2000], args.data + '_' + args.ood_data)
    # scatter_plot_single(x, args.data + '_train')
    # scatter_plot_single(y[0:2000],args.data + '_test')
    # scatter_plot_single(z[0:2000], args.data + '_ood')
    
    # # scatter_plot(x, y, z, args.data + '_Outlier')
    # print('finish plot')
    # print(fdsf)

    ################################################################################################
    
    # print(fdsfsd)
    # feat_log = np.load(save_dir + 'imagenet/train/0_1000.npy')
    # feat_log_label = np.load(save_dir + 'imagenet/train/0-50000/label.npy')

    # feat_log_val = np.load(save_dir + 'imagenet/test/0_1000.npy')
    # feat_log_val_label = np.load(save_dir + 'imagenet/test/label.npy')

    # ood_feat_log = np.load(save_dir + 'imagenet/' + args.ood_data + '/0_1000.npy')
    # ood_feat_log_label = np.load(save_dir + 'imagenet/'+args.ood_data+'/label.npy')

    # print('test_feature:', feat_log_val.min(), '-', feat_log_val.max())
    # print('train_feature:', feat_log.min(), '-', feat_log.max())
    # print('ood_feature:', ood_feat_log.min(), '-', ood_feat_log.max())
    # print(fdsfsd)

    ##################### low dimension feature ##################################################### 
    # if args.data == 'cifar10':
    #     classnums = 10 
    # elif args.data == 'imagenet':
    #     classnums = 1000
    # ood_data = args.ood_data 
    # save_dir = './2023_05_31_detect_base_method/compare_method/'
    # v = np.load('./2023_05_31_detect_base_method/compare_method/'+args.data+'_train_foward_feature.npy')
    # reduce_k = 10
    # svd = TruncatedSVD(n_components=reduce_k, n_iter=10, random_state=42)
    # svd.fit(v)
    # print('ratio:', svd.explained_variance_ratio_.sum())
    # print('singular_values_:', svd.singular_values_)
    # p = svd.components_
 
    # feat_log = np.load(save_dir +  args.data + '_train_foward_feature.npy')
    # feat_log = np.dot(feat_log, p.T)
    # feat_log_label = np.load(save_dir +  args.data + '_train_label.npy')

    # feat_log_val = np.load(save_dir +  args.data + '_test_foward_feature.npy')
    # feat_log_val = np.dot(feat_log_val, p.T)
    # feat_log_val_label = np.load(save_dir +  args.data + '_test_label.npy')

    # ood_feat_log = np.load(save_dir + args.data + '_' + ood_data + '_foward_feature.npy')
    # ood_feat_log = np.dot(ood_feat_log, p.T)
    # ood_feat_log_label = np.load(save_dir + args.data + '_' + ood_data + '_label.npy')

    ############## train Linear Probe
    ifbn = args.ifbn
    net = Linear_Probe(reduce_k, classnums, ifbn).cuda()
    # # # net = Linear_Probe(1000, classnums).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # trans = args.trans 
    train_set = MyDataset(feat_log, feat_log_label)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    test_set = MyDataset(feat_log_val, feat_log_val_label)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    
    print('ifbn:', ifbn)
    if ifbn == True:
        save_name = args.data+'_linear_bn_' + str(reduce_k)
    else:
        save_name = args.data+'_linear_' + str(reduce_k)
    if not os.path.exists(save_dir + save_name + '.pt'):
        train(net, optimizer, criterion, train_dataloader, test_dataloader, epochs=10, print_freq=100, save_dir=save_dir, save_name=save_name)
    print('finish train')

    ################# start detection
    base_method = args.base_method
    ood_set = MyDataset(ood_feat_log, ood_feat_log_label)
    ood_dataloader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    
    net.load_state_dict(torch.load(save_dir + save_name + '.pt'))
    net.eval()
    test(test_dataloader, net)

    
    if base_method == 'msp':
        with torch.no_grad():
            confs = msp(test_dataloader, net)
            ood_confs =  msp(ood_dataloader, net)
    elif base_method == 'energy':
        temper = 1
        with torch.no_grad():
            confs = energy(test_dataloader, net, temper)
            ood_confs = energy(ood_dataloader, net, temper)
    elif base_method == 'maha':
        num_classes = classnums       
        confs, ood_confs = mahalanobis_official(net, train_dataloader, test_dataloader, ood_dataloader, num_classes, magnitude=0.0, save_path=save_dir, save_name='mahalanobis_'+args.data+'_'+args.ood_data, data=args.data)
    elif base_method == 'odin':
        if args.data =='cifar10':
            odin_epsilon = 50
        elif args.data == 'imagenet':
            odin_epsilon = 0.005
        # odin_temperature = 1000
        odin_temperature = 100
        print('eps:', odin_epsilon, 'temper:', odin_temperature)
        confs = odin(test_dataloader, net, odin_temperature, odin_epsilon)
        ood_confs = odin(ood_dataloader, net, odin_temperature, odin_epsilon)
    elif base_method == 'grad_norm':
        ####### batch_size should be set to 1
        num_classes = classnums
        gradnorm_temperature = 1
        kl_loss = False
        p_norm = 1
        # indices = [i for i in range(0,500)]
        # sub_dataset = torch.utils.data.Subset(dataset.test_set, indices)
        # dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        # confs = grad_norm(dataloader, model, gradnorm_temperature, num_classes, kl_loss)
        # ood_confs = grad_norm(out_loader, model, gradnorm_temperature, num_classes, kl_loss)
        # print(confs.shape, ood_confs.shape)
        # results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
        # print(base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # print(fdsfsf)
        confs = -grad_norm(test_dataloader, net, gradnorm_temperature, num_classes, kl_loss, p_norm)
        ood_confs = -grad_norm(ood_dataloader, net, gradnorm_temperature, num_classes, kl_loss, p_norm)
    elif base_method == 'react':
        temper = 1
        # threshold = 1e6
        def cal_threshold(feature, percent):
            feature = feature.flatten()
            threshold = np.percentile(feature, percent*100) # percent的数小于threshold
            # print((feature<threshold).sum())
            return threshold
        if args.data == 'cifar10':
            percent = 0.90
            start_dim = 10
        elif args.data == 'imagenet':
            percent = 0.70
            start_dim = 950
        # threshold = cal_threshold(feat_log_val[:,start_dim:], percent=percent)
        # print('threshold:', threshold, ',percent:', percent)
        # ood_feat_log
        threshold = cal_threshold(feat_log_val[:,start_dim:], percent=percent)
        print('threshold:', threshold, ',percent:', percent)
        
        confs = react(test_dataloader, net, temper, threshold, start_dim)
        ood_confs = react(ood_dataloader, net, temper, threshold, start_dim)
    elif base_method == 'bats':
        temper = 1
        lams = np.arange(0.1, 0.11, 0.1)
        for lam in lams:
            net2 = copy.deepcopy(net)
            # for n, m in net2.named_modules():
            #         print(n)
            # print(fdsfsd)
            if args.data =='cifar10':
                # lam = 3.25  
                truncated_module = ['bn'] 
                start_dim = 10
            elif args.data == 'imagenet':
                # lam = 1.05
                truncated_module = ['bn']
                start_dim = 950
            
            print('lam:', lam, ',bn_module:', truncated_module)
            for n, module in net2.named_modules():
                    if n in truncated_module:
                        Trunc_BN = TrBN(module, lam, start_dim)
                        _set_module(net2, n, Trunc_BN)
                        upper_bound, lower_bound = Trunc_BN.get_static()
            # with torch.no_grad():
            #     test(test_dataloader, model)
            net2.eval()
            confs = bats(test_dataloader, net2, temper, upper_bound, lower_bound)
            ood_confs = bats(ood_dataloader, net2, temper, upper_bound, lower_bound)
            print(confs.shape, ood_confs.shape)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print('lam:', lam, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # print(fsds)
    elif base_method == 'knn':
        if args.data == 'cifar10':
            K=5
        elif args.data == 'imagenet':
            K=10
        confs = knn(feat_log, feat_log_val, K) #+ knn(train_forward_feature, test_forward_feature, K)
        ood_confs = knn(feat_log, ood_feat_log, K) #+ knn(train_forward_feature, ood_forward_feature, K)
        
        # train_forward_feature = np.load('2023_05_31_detect_base_method/compare_method/'+args.data + '_train' + '_foward_feature.npy')
        # test_forward_feature = np.load('2023_05_31_detect_base_method/compare_method/'+args.data + '_test' + '_foward_feature.npy')
        # ood_forward_feature = np.load('2023_05_31_detect_base_method/compare_method/'+args.data + '_' + args.ood_data + '_foward_feature.npy')
        
        # K = 5
        # a = 1
        # for K in [1, 5]:
        #     confs = a*knn(feat_log, feat_log_val, K) #+ knn(train_forward_feature, test_forward_feature, K)
        #     ood_confs = a*knn(feat_log, ood_feat_log, K) #+ knn(train_forward_feature, ood_forward_feature, K)
        #     results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
        #     print(K, base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])

        
        # plt.clf()
        # plt.figure(figsize=(16,10), dpi=800)
        # sns.kdeplot(-confs, fill=True, color='#FFBE7A', label="Test", alpha=.7)
        # sns.kdeplot(-ood_confs, fill=True, color='#FA7F6F', label="Outlier", alpha=.7)
        # # Decoration
        # plt.xlim((-0.1, 1.4))
        # plt.title('Density plot of the scores for IN and OOD data', fontsize=22)
        # plt.tight_layout()
        # plt.legend(fontsize=28)
        # plt.savefig(fig_save_dir + 'density.png', dpi=800)
        # print(dffs)
    
    elif base_method == '1':
        # knn_feature = []
        # for i in range(10):
        #     avg_grad = cal_grad_per_class(model, dataset.train, i, args.kernel, z_theta, I_2, save_dir)
        # print(fdsfsd)

        p_norm = 2
        a = torch.from_numpy(feat_log_val)
        b = torch.from_numpy(ood_feat_log)
        a = torch.abs(a)
        b = torch.abs(b)
        confs = torch.norm(a, dim=1, p=p_norm).numpy()
        ood_confs = torch.norm(b, dim=1, p=p_norm).numpy()
        # print(confs.shape, ood_confs.shape)

        # confs = confs.numpy()
        # ood_confs = ood_confs.numpy()

        # avg_grads = []
        # for i in range(0, 1000):
        #     avg_grad = torch.from_numpy(np.load(save_dir + 'cifar10/avg_grad_per_class/' +str(i) +'.npy'))
        #     avg_grads.append(avg_grad.unsqueeze(1))
        # avg_grads = torch.cat(avg_grads, 1)
        # avg_grads_norm = torch.norm(avg_grads, dim=0).unsqueeze(0)
        # # print('norm:', avg_grads_norm, avg_grads.shape)
        # avg_grads = avg_grads/avg_grads_norm

        # np.save(save_dir + 'imagenet/avg_grads.npy', avg_grads.numpy())
        # p = avg_grads

        # confs, confs2, confs3, confs4, confs5, confs6, confs7 = try_1(model, args.kernel, z_theta, I_2, p, p_norm, dataset.test)
        # np.save(save_dir + '1_confs5.npy', confs)
        confs5 = np.load(save_dir + '1_confs5.npy')
        ood_confs, ood_confs2, ood_confs3, ood_confs4, ood_confs5, ood_confs6, ood_confs7 = try_1(model, args.kernel, z_theta, I_2, p, p_norm, out_loader)
        np.save(save_dir + '1_ood_confs5_' + args.ood_data + '.npy', ood_confs5)
        # results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
        # print('1, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # results = cal_metric(confs2, ood_confs2) ##### 大于 threshold 为 ID data
        # print('2, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # results = cal_metric(confs3, ood_confs3) ##### 大于 threshold 为 ID data
        # print('3, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # results = cal_metric(confs4, ood_confs4) ##### 大于 threshold 为 ID data
        # print('4, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        results = cal_metric(confs5, ood_confs5) ##### 大于 threshold 为 ID data
        print('5, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # results = cal_metric(confs6, ood_confs6) ##### 大于 threshold 为 ID data
        # print('6, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # results = cal_metric(confs7, ood_confs7) ##### 大于 threshold 为 ID data
        # print('7, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        print(fdsfs)
    elif base_method == '2':  
        # train_indices = []
        # dataloader_ = torch.utils.data.DataLoader(dataset.train_set, batch_size=500, shuffle=False, num_workers=4)
        # num = 0
        # label = 0
        # for batch in dataloader_:
        #     inputs, targets = batch
        #     indices = torch.nonzero(targets==label)
        #     if indices.shape[0]!=inputs.shape[0]:
        #         label = label+1
        #         num = num + indices.shape[0]
        #         train_indices.append(num)
        #         num=inputs.shape[0]-indices.shape[0]
        #         print(train_indices)
        #         # print(targets[indices.shape[0]-1], targets[indices.shape[0]])
        #     else:
        #         num = num + inputs.shape[0]

        # train_indices = np.array(train_indices) 
        # np.save(save_dir + 'imagenet/train_indices.npy', train_indices)
        # train_indices_2 = np.zeros(train_indices.shape)
        # for i in range(train_indices.shape[0]):
        #     train_indices_2[i] = train_indices[0:i].sum()
        # print(train_indices_2)
        # np.save(save_dir + 'imagenet/train_indices_2.npy', train_indices_2)
        # print(fsdfds)

        # ood_knn_feature=[]
        # for start_k in np.arange(0, 1000, 150):
        #     end_k=start_k+150
        #     end_k = min(end_k, 1000)
        #     print('start_k:', start_k, 'end_k:', end_k)
        #     ood_knn_feature.append(torch.from_numpy(np.load(save_dir + 'imagenet/train/50000-150000/' + str(start_k) + '_' + str(end_k) + '.npy')))
        # ood_knn_feature = torch.cat(ood_knn_feature, 1).numpy()
        # print(ood_knn_feature.shape)
        # np.save(save_dir + 'imagenet/train/50000-150000/0_1000.npy', ood_knn_feature)
        # del ood_knn_feature
        
        # dataloader = torch.utils.data.DataLoader(dataset.test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        # label = []
        # for batch in out_loader:
        #     inputs, targets = batch
        #     label.append(targets)
        # label = torch.cat(label, dim=0).numpy()
        # print(label.shape)
        # np.save(save_dir + 'imagenet/' + args.ood_data + '/label.npy', label)
        # print(fdsfsd)

        # for classes in np.arange(540, 1000, 20):
            # print('classes:', classes)


        # feat_log = np.load(save_dir +  args.data + '_train_feature.npy')[:,0:reduce_k]
        # feat_log_label = np.load(save_dir +  args.data + '_train_label.npy')

        # feat_log_val = np.load(save_dir +  args.data + '_test_feature.npy')[:,0:reduce_k]
        # feat_log_val_label = np.load(save_dir +  args.data + '_test_label.npy')

        # ood_feat_log = np.load(save_dir + args.data + '_' + ood_data + '_feature.npy')[:,0:reduce_k]
        # ood_feat_log_label = np.load(save_dir + args.data + '_' + ood_data + '_label.npy')
        
        

        # ood_knn_feature = generate_feature(model, out_loader, z_theta, I_2, save_dir, 'imagenet/'+ args.ood_data + '/', args)
        # sample_num = 50000
        # kernel_dataset, _ = torch.utils.data.random_split(dataset.train_set, [sample_num, len(dataset.train_set)-sample_num], generator=torch.Generator().manual_seed(0))
        # dataloader = torch.utils.data.DataLoader(kernel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # feat_log = generate_feature(model, dataloader, z_theta, I_2, save_dir, 'imagenet/train/0-50000/', args)#[0:30000]
        # feat_log_label = np.load(save_dir + 'imagenet/train/0-50000/' + 'label.npy')
        # # feat_log_2 = generate_feature(model, dataloader, z_theta, I_2, save_dir, 'imagenet/train/50000-150000/', args)
        # # feat_log = torch.cat((torch.from_numpy(feat_log), torch.from_numpy(feat_log_2)), 0)
        # # feat_log = generate_cos_feature(model, dataloader, z_theta, I_2, save_dir, 'imagenet/train/0-50000/', args)

        # feat_log_val = generate_feature(model, dataset.test, z_theta, I_2, save_dir, 'imagenet/test/', args)
        # feat_log_val_label = np.load(save_dir + 'imagenet/test/' + 'label.npy')
        # # feat_log_val = generate_cos_feature(model, dataset.test, z_theta, I_2, save_dir, 'imagenet/test/', args)

        # ood_feat_log = generate_feature(model, out_loader, z_theta, I_2, save_dir, 'imagenet/'+args.ood_data+'/', args)
        # ood_feat_log_label = np.load(save_dir + 'imagenet/'+args.ood_data+'/' + 'label.npy')
        # ood_feat_log = generate_cos_feature(model, out_loader, z_theta, I_2, save_dir, 'imagenet/'+args.ood_data+'/', args)
        
        feat_log = np.load('./2023_06_15_cal_p_fast_per_class/resnet18_cifar10_NFK/cifar10_train.npy')
        feat_log_val = np.load('./2023_06_15_cal_p_fast_per_class/resnet18_cifar10_NFK/cifar10_test.npy')
        ood_feat_log = np.load('./2023_06_15_cal_p_fast_per_class/resnet18_cifar10_NFK/cifar10_'+args.ood_data+'.npy')

        train_knn_feature = feat_log
        test_knn_feature = feat_log_val
        ood_knn_feature = ood_feat_log

        # def cal_threshold(feature, percent):
        #     feature = feature.flatten()
        #     threshold = np.percentile(feature, percent*100) # percent的数小于threshold
        #     # print((feature<threshold).sum())
        #     return threshold
        # percent = 0.05
        # threshold = cal_threshold(feat_log_val, percent=percent)
        # feat_log_val = feat_log_val.clip(min=threshold)
        # ood_feat_log = ood_feat_log.clip(min=threshold)


        p_norm = np.inf
        print('p_norm:', np.inf)
        confs = torch.norm(torch.from_numpy(feat_log_val), dim=1, p=p_norm).numpy()
        ood_confs = torch.norm(torch.from_numpy(ood_feat_log), dim=1, p=p_norm).numpy()
        print(confs.shape, ood_confs.shape)
        results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
        print('Norm, FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # print(dsfs)

        # test_knn_feature = feat_log_val[0:50*classes]
        # feat_log = np.load(save_dir + 'imagenet/train/0_1000.npy')
        # feat_log_label = np.load(save_dir + 'imagenet/train/0-50000/label.npy')

        # feat_log_val = np.load(save_dir + 'imagenet/test/0_1000.npy')
        # feat_log_val_label = np.load(save_dir + 'imagenet/test/label.npy')

        # train_knn_feature = feat_log
        # feat_log = torch.from_numpy(feat_log)
        # feat_log_label = torch.from_numpy(feat_log_label)
        # for classes in np.arange(420, 1000, 100):
        #     # train_knn_feature = []
        #     # # classes = 300
        #     # for i in range(classes):
        #     # # start = 0
        #     # # end = 300
        #     # # for i in range(start, end):
        #     #     indices = torch.nonzero(feat_log_label==i)[:,0]
        #     #     train_knn_feature.append(feat_log[indices])
        #     # train_knn_feature = torch.cat(train_knn_feature, 0).numpy()

        #     # test_knn_feature = feat_log_val[0:50*classes]
        #     # indices = [i for i in range(50*start,50*end)]
        #     indices = [i for i in range(0, 50*classes)]
        #     sub_dataset = torch.utils.data.Subset(dataset.test_set, indices)
        #     dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        #     with torch.no_grad():
        #         test(dataloader, model)
        #     print(ffsdf)

        #     # test_knn_feature = feat_log_val[50*700:50*800]
        #     ood_knn_feature = ood_feat_log
        #     print('classes:', classes)
        #     print('train_knn_feature:', train_knn_feature.shape, 'test_knn_feature:', test_knn_feature.shape, 'ood_knn_feature:', ood_knn_feature.shape)

        for K in [1,5,10,20,30,100,1000]:
        # for K in [5]:
            confs = knn(train_knn_feature, test_knn_feature, K)
            ood_confs = knn(train_knn_feature, ood_knn_feature, K)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print(K, ',FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        print(fdsfsd)

        # print(fdsfsd)
       
        test_forward_feature, test_label, feature_test_dataloader = obtain_feature_loader(model, dataset.test, save_dir, args.data+'_test', args)
        train_forward_feature, train_label, feature_train_dataloader = obtain_feature_loader(model, dataset.train, save_dir, args.data+'_train', args)
        ood_forward_feature, ood_label, feature_ood_dataloader = obtain_feature_loader(model, out_loader, save_dir, args.data +'_'+ args.ood_data, args)

        temper = 1
        # threshold = 1e6
        def cal_threshold(feature, percent):
            feature = feature.flatten()
            threshold = np.percentile(feature, percent*100) # percent的数小于threshold
            return threshold
        percent = 0.90
        threshold = cal_threshold(test_forward_feature, percent=percent)
        print('threshold:', threshold, ',percent:', percent)
        a = 1
        for K in [5]:
            p_norm = np.inf
            # confs = a*torch.norm(torch.from_numpy(feat_log_val), dim=1, p=p_norm).numpy() + react(feature_test_dataloader, model, temper, threshold)
            # ood_confs = a*torch.norm(torch.from_numpy(ood_feat_log), dim=1, p=p_norm).numpy() + react(feature_ood_dataloader, model, temper, threshold)

            confs = a*knn(feat_log, feat_log_val, K) + react(feature_test_dataloader, model, temper, threshold)
            ood_confs = a*knn(feat_log, ood_feat_log, K) + react(feature_ood_dataloader, model, temper, threshold)

            # confs = a*knn(feat_log, feat_log_val, K)*react(feature_test_dataloader, model, temper, threshold)
            # ood_confs = a*knn(feat_log, ood_feat_log, K)*react(feature_ood_dataloader, model, temper, threshold)

            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print('FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])

        print(fdsfs)

        for start_k in np.arange(0, 1000, 150):
            avg_grads = []
            end_k=start_k+150
            end_k = min(end_k, 1000)
            print('start_k:', start_k, 'end_k:', end_k)
            for i in range(start_k, end_k):
                avg_grad = torch.from_numpy(np.load(save_dir + 'imagenet/avg_grad_per_class/' +str(i) +'.npy'))
                avg_grads.append(avg_grad.unsqueeze(1))
            avg_grads = torch.cat(avg_grads, 1)
            avg_grads_norm = torch.norm(avg_grads, dim=0).unsqueeze(0)
            print('norm:', avg_grads_norm, avg_grads.shape)
            avg_grads = avg_grads/avg_grads_norm

            # cosine = []
            # for i in range(k):
            #     cos = []
            #     for j in range(k):
            #         # if j!=i:
            #         co = torch.dot(avg_grads[:,i],avg_grads[:,j])
            #         cos.append(co)
            #     cosine.append(cos)
            # print(cosine[0], cosine[1])
            # cosine = torch.Tensor(cosine)
            # print('cos mean:', cosine.mean())
            # print(fsdfas)
            # indices = [i for i in range(0,500)]
            # sub_dataset = torch.utils.data.Subset(dataset.test_set, indices)

            # dataloader = torch.utils.data.DataLoader(dataset.test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
            # test_feature = []
            # num = 0
            # for batch in dataloader:
            #     inputs, targets = batch
            #     inputs = inputs.cuda()
            #     targets = targets.cuda()
            #     # print(targets)
            #     grad_feature = cal_fisher_vector(model, inputs, kernel=args.kernel, z_theta=z_theta, I_2=I_2).detach().cpu()
            #     feature = grad_feature @ avg_grads
            #     test_feature.append(feature)
            #     num = num + feature.shape[0]
            #     print('test, num=', num)
            # test_feature = torch.cat(test_feature, dim=0).numpy()
            # np.save(save_dir + 'imagenet/test/' + str(start_k) + '_' + str(end_k) + '.npy', test_feature)
            # print(test_feature.shape)
            # continue

            
            # indices = [i for i in range(0,12800)]
            # sub_dataset = torch.utils.data.Subset(dataset.train_set, indices)
            # dataloader = torch.utils.data.DataLoader(dataset.train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

            sample_num = 50000
            kernel_dataset, kernel_dataset_2 = torch.utils.data.random_split(dataset.train_set, [sample_num, len(dataset.train_set)-sample_num], generator=torch.Generator().manual_seed(0))
            
            indices = [i for i in range(0,100000)]
            sub_dataset = torch.utils.data.Subset(kernel_dataset_2, indices)

            dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            train_feature = []
            num = 0
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.cuda()
                targets = targets.cuda()
                if num==0:
                    print('targets:',targets)
                grad_feature = cal_fisher_vector(model, inputs, kernel=args.kernel, z_theta=z_theta, I_2=I_2).detach().cpu()
                feature = grad_feature @ avg_grads
                train_feature.append(feature)
                num = num + feature.shape[0]
                print('train, num=', num)
            train_feature = torch.cat(train_feature, dim=0).numpy()
            os.makedirs(save_dir + 'imagenet/train/50000-100000/', exist_ok=True)
            np.save(save_dir + 'imagenet/train/50000-100000/' + str(start_k) + '_' + str(end_k) + '.npy', train_feature)
            print(train_feature.shape) 
            continue


            ood_feature = []
            num = 0
            # indices = [i for i in range(500,1000)]
            # sub_dataset = torch.utils.data.Subset(dataset.test_set, indices)
            # out_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=10, shuffle=False, num_workers=2)
            for batch in out_loader:
                inputs, targets = batch
                inputs = inputs.cuda()
                targets = targets.cuda()
                # print(targets)
                grad_feature = cal_fisher_vector(model, inputs, kernel=args.kernel, z_theta=z_theta, I_2=I_2).detach().cpu()
                feature = grad_feature @ avg_grads
                ood_feature.append(feature)
                num = num + feature.shape[0]
                print('ood, num=', num)
            ood_feature = torch.cat(ood_feature, dim=0).numpy()
            os.makedirs(save_dir + 'imagenet/'+ args.ood_data + '/', exist_ok=True)
            np.save(save_dir + 'imagenet/'+ args.ood_data + '/' + str(start_k) + '_' + str(end_k) + '.npy', ood_feature)
            print('ood_feature:', ood_feature.shape) 
            continue


            for K in [1,5,10,100,1000]:
                confs = knn(train_feature, test_feature, K)
                ood_confs = knn(train_feature, ood_feature, K)
                print(confs.shape, ood_confs.shape)
                results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
                print(K, base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        print(fsdfs)

        train_knn_feature = try_2(model, dataset.train_set, dataset.train, args.kernel, z_theta, I_2, classnums, save_dir, 'train/').numpy()
        # np.save(save_dir + args.data + '_train_feature_1000.npy', train_knn_feature)
        test_knn_feature = try_2(model, dataset.train_set, dataset.test, args.kernel, z_theta, I_2, classnums, save_dir, 'test/').numpy()
        # np.save(save_dir + args.data + '_test_feature_1000.npy', test_knn_feature)
        ood_knn_feature = try_2(model, dataset.train_set, out_loader, args.kernel, z_theta, I_2, classnums, save_dir, args.ood_data+'/').numpy()
        # np.save(save_dir + args.data + '_' + args.ood_data + '_feature_1000.npy', ood_knn_feature)
        for K in [1,5,10,100,1000]:
            confs = knn(train_knn_feature, test_knn_feature, K)
            ood_confs = knn(train_knn_feature, ood_knn_feature, K)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print(K, base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
    elif base_method == '3':
        p_norm = 2
        a = torch.from_numpy(feat_log_val)
        b = torch.from_numpy(ood_feat_log)
        a = torch.abs(a)
        b = torch.abs(b)
        confs = torch.norm(a, dim=1, p=p_norm).numpy()
        ood_confs = torch.norm(b, dim=1, p=p_norm).numpy()
    else:
        print('No exist method')

    confs_f = np.load('2023_05_31_detect_base_method/detection score/'+args.data+'/'+base_method+'_'+args.data+'.npy')
    ood_confs_f = np.load('2023_05_31_detect_base_method/detection score/'+args.data+'/'+base_method+'_'+args.ood_data+'.npy')
    print(confs.shape, ood_confs.shape)
    print(confs_f.shape, ood_confs_f.shape)
    print('confs.mean:', confs.mean(), ', ood_confs.mean:', ood_confs.mean())
    print('confs_f:', confs_f.mean(), ', ood_confs_f:', ood_confs_f.mean())
    a = args.a
    print('a=', a)
    confs = confs + a*confs_f
    ood_confs = ood_confs + a*ood_confs_f
    
    # print(confs.shape, ood_confs.shape)
    results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
    print(base_method, ',', args.ood_data, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])


    # from advertorch.attacks import LinfPGDAttack
    # # adversary = LinfPGDAttack(
    # #     model_copy, loss_fn=nn.CrossEntropyLoss(), eps=0.3, nb_iter=20, eps_iter=0.03,
    # #     rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False
    # # )
    return

if __name__ == "__main__":
    main()

    


    

