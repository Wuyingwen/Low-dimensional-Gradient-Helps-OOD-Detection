# CUDA_VISIBLE_DEVICES=3 python 2023_05_31_detect_base_method.py --batch_size 256 --model resnet18 --data cifar10 --base_method msp --ood_data SVHN
# CUDA_VISIBLE_DEVICES=3 python 2023_05_31_detect_base_method.py --batch_size 20 --model resnetv2-101 --data imagenet --base_method msp --ood_data inat

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import torch
import os
import numpy as np

from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar10,Cifar100,Cifar10_for_ood
from data.dataloader_for_ood_2 import IN_DATA, get_loader_out, IN_DATA_2, get_loader_out_2

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
from sklearn.linear_model import LogisticRegressionCV
from model.densenet import densenet121
from torchsummary import summary


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

class Linear_Probe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear_Probe, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
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
    def __init__(self, data, label, trans=False):
        if trans == True:
            ##### 归一化 0-1 之间， data.shape:[bz,dim], e.g. [50000,128]
            normalizer = lambda x: (x-np.expand_dims(x.min(1), axis=1))/np.expand_dims(x.max(1)-x.min(1), axis=1)
            data = normalizer(data)

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

def odin(data_loader, model, odin_temperature, odin_epsilon):
    score = []
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for i, batch in enumerate(data_loader):
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()

        inputs = Variable(inputs, requires_grad=True)
        outputs = model(inputs)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / odin_temperature

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -odin_epsilon*gradient)
        # tempInputs = inputs.data-odin_epsilon*gradient
        tempInputs = Variable(tempInputs)

        outputs = model(tempInputs)

        outputs = outputs / odin_temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        score.extend(np.max(nnOutputs, axis=1))

        # nnOutputs = odin_temperature * (torch.logsumexp(outputs / odin_temperature, dim=1))
        # score.extend(nnOutputs.data.cpu().numpy())
    score = np.array(score)
    return score

def grad_norm(data_loader, model, gradnorm_temperature, num_classes=10, kl_loss=True):
    score = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    p_norm=1
    print('p-norm:', p_norm)
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
            # layer_grad = torch.autograd.grad(outputs=loss, inputs=model.head.conv.weight, retain_graph=False)[0]
            layer_grad = torch.autograd.grad(outputs=loss, inputs=model.weight, retain_graph=False)[0]
            # print(layer_grad.shape)
            layer_grad_norm = torch.norm(layer_grad, p=p_norm)
        elif kl_loss==True:
            layer_grad_norm = loss.detach()
        score.append(layer_grad_norm.cpu().numpy())
        # print(i)
    score = np.array(score)
    return score

def react(data_loader, model, temper, threshold, args):
    score = []
    num = []
    gap = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            # if inputs.max() > threshold:
            #     gap.append(float(inputs.max()-threshold))
            # error = (inputs-threshold)
            # index = torch.where(error>0)
            # error = error[index]
            # gap.append(float(error.mean()))

            # print(threshold, inputs.max())
            # num_of_value = int(inputs.shape.numel())
            # print(inputs>threshold)
            # print((inputs>threshold).shape)
            # num_of_abnormal_value = (inputs>threshold).sum()
            # num_of_abnormal_value = int(num_of_abnormal_value)
            # abnormal_value_ratio = num_of_abnormal_value/num_of_value
            # num.append(abnormal_value_ratio)
            # # print(abnormal_value_ratio)
            # # print(num_of_value, num_of_abnormal_value)
            # if len(num)>100:
            #     break
                # print(fsdfsd)

            inputs = inputs.clip(max=threshold)
            # inputs = inputs.clip(min=threshold)
            # inputs = torch.where(inputs<threshold,threshold,inputs)
            # inputs = torch.where(inputs>threshold,threshold,inputs)
            

            # if args.model =='resnet18':
            #     logits = model.fc(inputs)
            # elif args.model =='wideresnet':
            #     logits = model.f[8](inputs)
            # elif args.model =='densenet':
            #     logits = model.classifier(inputs)
            # elif args.model =='mobilenet':
            #     logits = model.classifier[-1](inputs)
            # elif args.model == 'vit':
            #     logits = model.head.layers[-1](inputs)
            logits = model.fc(inputs)

            conf = temper * (torch.logsumexp(logits / temper, dim=1)) 
            score.extend(conf.data.cpu().numpy())
    score = np.array(score) 

    # num = torch.Tensor(num)
    # gap = torch.Tensor(gap)
    # print(num.shape, gap.shape)
    # print(torch.mean(num), torch.mean(gap))
    # print(fdsfs)
    return score

##### feature在bn之前做bats 
class TrBN(nn.Module):
    def __init__(self, bn, lam):
        super().__init__()
        self.bn = bn
        self.lam = lam
        self.sigma = bn.weight
        self.mu = bn.bias
        self.upper_bound = self.sigma*self.lam + self.mu
        self.lower_bound = -self.sigma*self.lam + self.mu
        
    def forward(self, x):
        y = self.bn(x)
        upper_bound = self.upper_bound.view(1,self.upper_bound.shape[0],1,1)
        lower_bound = self.lower_bound.view(1,self.lower_bound.shape[0],1,1)
        # print(x.shape, y.shape, self.mu.shape, self.sigma.shape, self.bn.running_mean.shape, self.bn.running_var.shape)
        y = torch.where(y<upper_bound, y, upper_bound)
        y = torch.where(y>lower_bound, y, lower_bound)
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

def vim(feat_log, feat_log_val, model, DIM):
    from sklearn.covariance import EmpiricalCovariance
    from numpy.linalg import norm, pinv
    from scipy.special import logsumexp
    batch = 512
    logit_id_train = []
    for i in np.arange(0, feat_log.shape[0], batch):
        inputs = torch.from_numpy(feat_log[i:min(i+batch, feat_log.shape[0])]).cuda()
        output = model.fc(inputs)
        for j in range(inputs.shape[0]):
            logit_id_train.append(output[j].detach().cpu().numpy())
    logit_id_train = np.array(logit_id_train)
    
    logit_id_val = []
    for i in np.arange(0, feat_log_val.shape[0], batch):
        inputs = torch.from_numpy(feat_log_val[i:min(i+batch, feat_log.shape[0])]).cuda()
        output = model.fc(inputs)
        for j in range(inputs.shape[0]):
            logit_id_val.append(output[j].detach().cpu().numpy())
    logit_id_val = np.array(logit_id_val)
    
    # print('computing principal space...')
    u = -np.matmul(pinv((model.fc.weight).detach().cpu().numpy()), (model.fc.bias).detach().cpu().numpy())
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feat_log - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    
    # a = eig_vals[np.argsort(eig_vals * -1)]
    # b = 0.9 * np.sum(a)
    # c = 0
    # for i in range(len(a)):
    #     c = c + a[i]
    #     if c<b:
    #         continue
    #     else:
    #         break
    # DIM = i+1
    # print('adaptive DIM:', DIM)
    
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    # print('NS:', NS.shape)

    # b = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[0:DIM]]).T)
    # a = np.matmul(feat_log - u, b)
    # c = np.matmul(feat_log_val - u, b)
    # return a, c

    # print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feat_log - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    # print('alpha:', alpha)

    vlogit_id_val = norm(np.matmul(feat_log_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val
    
    return score_id

def residual(feat_log, feat_log_val, model, DIM):
    from sklearn.covariance import EmpiricalCovariance
    from numpy.linalg import norm, pinv
    from scipy.special import logsumexp
    batch = 512
    logit_id_train = []
    for i in np.arange(0, feat_log.shape[0], batch):
        inputs = torch.from_numpy(feat_log[i:min(i+batch, feat_log.shape[0])]).cuda()
        output = model.fc(inputs)
        for j in range(inputs.shape[0]):
            logit_id_train.append(output[j].detach().cpu().numpy())
    logit_id_train = np.array(logit_id_train)
    
    logit_id_val = []
    for i in np.arange(0, feat_log_val.shape[0], batch):
        inputs = torch.from_numpy(feat_log_val[i:min(i+batch, feat_log.shape[0])]).cuda()
        output = model.fc(inputs)
        for j in range(inputs.shape[0]):
            logit_id_val.append(output[j].detach().cpu().numpy())
    logit_id_val = np.array(logit_id_val)
    
    # print('computing principal space...')
    u = -np.matmul(pinv((model.fc.weight).detach().cpu().numpy()), (model.fc.bias).detach().cpu().numpy())
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feat_log - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    # print('NS:', NS.shape)

    vlogit_id_val = norm(np.matmul(feat_log_val - u, NS), axis=-1) 
    score_id = -vlogit_id_val 
    
    return score_id

def feature_low_dimension(feat_log, feat_log_val, DIM, K):
    from sklearn.covariance import EmpiricalCovariance
    from numpy.linalg import norm, pinv
    from scipy.special import logsumexp
        
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only
    feat_log = prepos_feat(feat_log)
    feat_log_val = prepos_feat(feat_log_val)
    
    # center = np.mean(feat_log, axis=0)
    # #### print(center.shape)
    # feat_log = feat_log -center
    # feat_log_val = feat_log_val - center
    
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feat_log)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    b = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    # b = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[0:DIM]]).T)
    a = np.matmul(feat_log, b)
    c = np.matmul(feat_log_val, b)

    ################# confs = knn(a, c, K)
    # normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    # prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only
    # a = prepos_feat(a)
    # c = prepos_feat(c)
    index = faiss.IndexFlatL2(a.shape[1])
    index.add(a)
    D, _ = index.search(c, K)
    scores_in = -D[:,-1]
    
    ################ confs = -norm(c)
    # scores_in = -np.linalg.norm(c, ord=2, axis=-1, keepdims=True).flatten()
    
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
        print('num:', num)

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
    index = []
    num = 0
    num_all = 0
    model.eval()
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda()
      
        predictions = model(inputs)
        # print(predictions.shape)
        # print(fsdfds)

        correct = (torch.argmax(predictions, 1) == targets)

        correct_index = correct.squeeze().nonzero().squeeze() + num_all
        # print(correct_index.shape)
        index.append(correct_index)

        num = num + (torch.nonzero(correct==True)).shape[0]
        num_all = num_all + targets.shape[0]
        # if num_all>10:
        #     break
    acc = num/num_all
    print('test_acc:', acc)

    # index = torch.cat(index, 0).cpu().detach().numpy()
    # np.save('sam_imagenet_correct_index.npy', index)
    # print('index:', index.shape)
    return acc

def select_dataloader(dataset, index, args):
    num = len(dataset)
    print('num:', num)

    all_index = np.arange(0, num, 1)
    rest_index = np.delete(all_index, index)

    sub_dataset = torch.utils.data.Subset(dataset, index)
    print('sub_dataset:', len(sub_dataset))
    sub_dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    rest_dataset = torch.utils.data.Subset(dataset, rest_index)
    print('rest_dataset:', len(rest_dataset))
    rest_dataloader = torch.utils.data.DataLoader(rest_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
   
    return sub_dataloader, rest_dataloader


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

from visualize_utils import get_feas_by_hook
def obtain_feature_loader(model, dataloader, save_path, save_name, args, check):
    if not os.path.exists(save_path + save_name + '_foward_feature'+check+'.npy'):
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
            # # 定义提取中间层的 Hook
            # if (args.data == 'cifar10' or args.data == 'cifar100') and args.model=='resnet18':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['avg_pool'])
            # elif (args.data == 'cifar10' or args.data == 'cifar100') and args.model=='wideresnet':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['f.7_flattening'])
            # elif (args.data == 'cifar10' or args.data == 'cifar100') and args.model=='densenet':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['equal'])
            # elif args.data == 'imagenet' and args.model=='mobilenet':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['classifier.0'])
            # elif args.data == 'imagenet' and args.model!='vit':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['avgpool'])
            # elif args.data == 'imagenet' and args.model=='vit':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['backbone.ln1'])
            
            # if args.model == 'wideresnet':
            #     fea_hooks = get_feas_by_hook(model, extract_module=['f.7_flattening'])

            output = model(inputs)
            forward_feature.append(output.detach().cpu())
            
            # print(len(fea_hooks))
            # features = fea_hooks[0].fea.squeeze()
            # if args.model == 'vit':
            #     features = features[:, 0]

            # forward_feature.append(features.detach().cpu())
            label.append(targets.detach().cpu())

        forward_feature = torch.cat(forward_feature, 0)
        label = torch.cat(label, dim=0)
        forward_feature = forward_feature.numpy()
        label = label.numpy()
        print(forward_feature.shape, label.shape)
        np.save(save_path + save_name + '_foward_feature'+check+'.npy', forward_feature)
        np.save(save_path + save_name + '_label'+check+'.npy', label)
    else:
        forward_feature = np.load(save_path + save_name + '_foward_feature'+check+'.npy')
        label = np.load(save_path + save_name + '_label'+check+'.npy')
        print(forward_feature.shape, label.shape)
    
    dataset = MyDataset(forward_feature, label)
    feature_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    return forward_feature, label, feature_dataloader


def mahalanobis(data_loader, model, mean, variance, num_classes=10, input_preprocess=False, magnitude=100):
    Mahalanobis = []
    for _, batch in enumerate(data_loader):
        data, target = batch
        data = data.cuda()
        target = target.cuda()

        if input_preprocess:
            for j in range(target.shape[0]):
                inputs = data[j]
                inputs = Variable(inputs, requires_grad = True)

                fea_hooks = get_feas_by_hook(model, extract_module=['avg_pool'])
                output = model(inputs)
                features = fea_hooks[0].fea.squeeze()

                for i in range(num_classes):
                    distance = ((features-mean[i]).unsqueeze(0) @ variance @ (features-mean[i]).unsqueeze(1))
                    if i==0:
                        min_distance = distance
                    elif distance < min_distance:
                        min_distance = distance
                loss = min_distance
                loss.backward()

                gradient =  torch.ge(inputs.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2

                input_adv = torch.add(inputs, -magnitude*gradient)
                fea_hooks = get_feas_by_hook(model, extract_module=['avg_pool'])
                output = model(input_adv)
                features = fea_hooks[0].fea.squeeze()

                for i in range(num_classes):
                    distance = ((features-mean[i]).unsqueeze(0) @ variance @ (features-mean[i]).unsqueeze(1)).item()
                    if i==0:
                        min_distance = distance
                    elif distance < min_distance:
                        min_distance = distance
                Mahalanobis.append(min_distance)
        else:
            # compute Mahalanobis score
            for j in range(target.shape[0]):
                inputs = data[j]

                fea_hooks = get_feas_by_hook(model, extract_module=['avg_pool'])
                output = model(inputs)
                features = fea_hooks[0].fea.squeeze()

                for i in range(num_classes):
                    distance = ((features-mean[i]).unsqueeze(0) @ variance @ (features-mean[i]).unsqueeze(1)).item()
                    if i==0:
                        min_distance = distance
                    elif distance < min_distance:
                        min_distance = distance
                Mahalanobis.append(min_distance)    
    return np.array(Mahalanobis)


def mahalanobis_official(model, train_loader, test_loader, ood_loader, num_classes, magnitude=0.01, save_path=None, save_name=None, data=None):
    # if not os.path.exists(save_path + save_name + '.npy'):
    if True:
        if args.data == 'cifar10' or args.data == 'cifar100':
            # extract_module = ['conv5_x']
            extract_module = ['avg_pool']
        elif args.data == 'imagenet':
            extract_module = ['layer4']
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


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key,value in state_dict.items()}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    #parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
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
    parser.add_argument("--base_method", default='knn', type=str, help="baseline meth0 od for detection")
    parser.add_argument("--ood_data", default='SVHN', type=str, help="places365, dtd, SVHN, iSUN, LSUN, places50, sun50, inat")
    parser.add_argument("--trans", default=False, type=str2bool, help="True or False")
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

    import torchvision as tv
    crop = 480
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    if args.data == 'mnist':
        dataset = MNIST(args.batch_size, args.threads, args.num_gpus)
    else:
        # dataset = IN_DATA(args.data, args.batch_size, args.threads, args.num_gpus, args.model)
        dataset = IN_DATA_2(args.data, args.batch_size, args.threads, args.num_gpus, args.model, val_tx)
    
    
    if args.model == 'vgg16bn':
        model = VGG16BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'vgg19bn':
        model = VGG19BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'resnet18':
        model = resnet18(10 if args.data == 'cifar10' else 100)
    elif args.model == 'wideresnet':
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10 if args.data == 'cifar10' else 100)
        checkpoint = torch.load('./weights/baseline/SGD_wideresnet-28-10_'+args.data+'_labsmooth=0.1/seed_111.pt')
        model.load_state_dict(checkpoint['model'], strict=False)
        check = '_wideresnet'
    elif args.model == 'lenet':
        model = LeNet()
    elif args.model == 'resnet50':
        import torchvision
        from torchvision import models
        # model = models.resnet50(pretrained=True)
        model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        check = ''
    elif args.model == 'vit':
        from mmpretrain.apis import init_model
        import mmengine
        torch.backends.cudnn.benchmark = True
        cfg = mmengine.Config.fromfile('model/vit-base-p16-384.py')
        model = init_model(cfg, '2023_11_22_dynamic_detection/vit_model/vit_weight.pth', 0).cuda().eval()
        check = '_vit'
    elif args.model == 'resnetv2-101':
        import model.resnetv2_for_gradnorm as resnetv2_for_gradnorm
        model = resnetv2_for_gradnorm.KNOWN_MODELS['BiT-S-R101x1'](head_size=1000)
        model.load_from(np.load('./model/BiT-S-R101x1.npz'))
    elif args.model == 'densenet':
        # model = densenet121(num_classes=10 if args.data == 'cifar10' else 100).to(device)
        # checkpoint = torch.load('./weights/baseline/SAM_densenet_'+args.data+'_best_rho=0.05_labsmooth=0.1/seed_42.pt')
        # model.load_state_dict(checkpoint['model'], strict=False)
        # check = '_densenet'
        import torchvision
        from torchvision import models
        model = models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
        check = '_densenet_imagenet'
    elif args.model == 'mobilenet':
        import torchvision
        from torchvision import models
        model = models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        check = '_mobilenet'

    #############
    # from DAL.models.wrn import WideResNet
    # model = WideResNet(40, 10 if args.data == 'cifar10' else 100, 2, dropRate=0.3).cuda()
    # model.load_state_dict(torch.load('./DAL/models/'+ args.data + '_baseline_vanilla.pt'))
    # check = '_wrn-40-2'
    # sys.path.append("../KNN")
    
    # from wide_resnet import SupConWideResNet
    # model = SupConWideResNet(name='wrn40_2', head='mlp', feat_dim=128).cuda()
    # model.load_state_dict(torch.load('../KNN/ckpt_epoch_1000.pth')['model'])
    # check = '_wrn-40-2_KNN+'

    if args.num_gpus==1:
        model.cuda()
    elif args.num_gpus>1:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model,
                    device_ids=[args.local_rank] 
                )
        
    # if args.data == 'mnist':
    #     checkpoint = torch.load('2023_03_06_compare_sgd/lenet_mnist_best_rho=0.05_labsmooth=0.0/seed_42.pt')
    #     print('epoch:', checkpoint['epoch'])
    #     model.load_state_dict(checkpoint['model'], strict=False)
    # elif args.data == 'cifar10':
    #     # checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar10_best_rho=0.05_labsmooth=0.1/seed_42.pt')
    #     # check = ''
    #     # checkpoint = torch.load('weights/SAM_resnet18_cifar10_rho=0.05_labsmooth=0.1/epoch199.pt')
    #     # check = '_sam'

    #     checkpoint = torch.load('2024_04_15_OE_detection/' + args.model +'_'+args.data+'_lr=0.07.pt')
    #     check = '_OE'

    #     # print('epoch:', checkpoint['epoch'])
    #     # model.load_state_dict(checkpoint['model'], strict=False)
    #     model.load_state_dict(checkpoint, strict=False)
    #     model.set_nor(False)

    # elif args.data == 'imagenet':
    #     # checkpoint = torch.load('weights/450.pt')
    #     # checkpoint = remove_prefix(checkpoint, 'module.')
    #     # model.load_state_dict(checkpoint, strict=True)
    #     # check = '_sam'
    #     # checkpoint = torch.load('2023_11_22_dynamic_detection/vit_model/vit_weight.pth')
    #     # check = '_vit'
    #     check = '_bit'
    #     # check = ''
    # elif args.data == 'cifar100':
    #     # checkpoint = torch.load('weights/SGD_resnet18_cifar100_labsmooth=0.1/epoch199.pt')
    #     # check = ''
    #     # checkpoint = torch.load('weights/SAM_resnet18_cifar100_rho=0.1_labsmooth=0.1/epoch199.pt')
    #     # check = '_sam'

    #     # checkpoint = torch.load('weights/SGD_wideresnet-28-10_cifar100_labsmooth=0.1/epoch199.pt')
    #     # check = '_wrt'

    #     # checkpoint = torch.load('weights/SAM_wideresnet-28-10_cifar100_rho=0.1_labsmooth=0.1/epoch199.pt')
    #     # check = '_sam_wrt'

    #     checkpoint = torch.load('2024_04_15_OE_detection/' + args.model +'_'+args.data+'_lr=0.07.pt')
    #     check = '_OE'

    #     # print('epoch:', checkpoint['epoch']) 
    #     # model.load_state_dict(checkpoint['model'], strict=False)
    #     model.load_state_dict(checkpoint, strict=False)
    #     model.set_nor(False)


    model.eval()
    ######################## eval
    print(len(get_model_vec_torch(model)))
    print(fdsfsd)
    with torch.no_grad():
        print('test accuracy: ')
        test(dataset.test, model)
    print(fsdfs)
    
    ######################## load OOD data
    if args.data == 'cifar10':
        classnums = 10 
    elif args.data == 'imagenet':
        classnums = 1000
    elif args.data == 'cifar100':
        classnums = 100
    
    method = 'compare_method'
    save_dir = file_name + '/' + method + '/'
    os.makedirs(save_dir, exist_ok=True)
    
    ood_data = args.ood_data 
    print('ood_data:', ood_data)
    loader_test_dict = get_loader_out(args, dataset=(None, ood_data), split=('val'))
    # loader_test_dict = get_loader_out_2(args, val_tx, dataset=(None, ood_data), split=('val'))
    out_loader = loader_test_dict.val_ood_loader

    ################# start detection
    base_method = args.base_method
    test_dataloader = dataset.test
    train_dataloader = dataset.train
    ood_dataloader = out_loader
    net = model

    # if args.data == 'cifar10':
        # net.set_nor(False) 
        # net.set_nor(True)
        # test(dataset.test, net)

    # for name,p in net.named_parameters():
    #     print(name, p.shape)
    # for n, m in model.named_modules():
    #     print(n)
    
    # # summary(model, (3,224,224))
    # print(fdsfsd)

    # index = np.load('sam_imagenet_correct_index.npy')
    # print(index.shape)
    # sub_test_dataloader, rest_test_dataloader = select_dataloader(dataset.test_set, index, args)
        
    ####################################################################################################
    # ood_dataloader   test_dataloader
    # record = []
    # # proj = torch.from_numpy(np.load('./2023_05_26_cal_p_fast/resnet18_cifar10_NFK/sample_num=49998_k=512/0-128.npy')).cuda()
    # # print(proj.shape)
    # for i, batch in enumerate(test_dataloader):
    #     inputs, targets = batch
    #     inputs = inputs.cuda()
    #     targets = targets.cuda()
        
    #     # model.zero_grad()
    #     # outputs = model(inputs)
    #     # pred = torch.argmax(outputs, 1)
    #     # # if j==0:
    #     # #     print('targets:', targets, 'pred:', torch.argmax(outputs, 1))
    #     # label = torch.Tensor([pred]).cuda()
    #     # label = label.to(torch.int64)
    #     # loss_ce = smooth_crossentropy(outputs, label, smoothing=0.0).mean()
    #     # loss_ce.backward()
    #     # mag = torch.norm(get_model_grad_vec_torch_2(model))
    #     # record.append(mag)
    #     # if mag>10:
    #     #     print('targets:', targets, 'pred:', torch.argmax(outputs, 1), mag)
    #     # # print(mag)

    #     gradient = []
    #     for j in range(10):
    #         model.zero_grad()
    #         outputs = model(inputs)
    #         # if j==0:
    #         #     print('targets:', targets, 'pred:', torch.argmax(outputs, 1))
    #         label = torch.Tensor([j]).cuda()
    #         label = label.to(torch.int64)
    #         loss_ce = smooth_crossentropy(outputs, label, smoothing=0.0).mean()
    #         loss_ce.backward()
    #         gradient.append(get_model_grad_vec_torch_2(model))
        
    #     sum = 0
    #     for j in range(10):
    #         sum = sum + gradient[j]
    #     avg = sum/10

    #     pred = torch.argmax(outputs, 1)
    #     mag = torch.norm(gradient[pred])
    #     cos = (gradient[pred] @ avg)/(torch.norm(gradient[pred])*torch.norm(avg))
    #     print(mag, cos)
    #     # record.append(torch.norm(avg)/mag)
    #     # if pred != targets:
    #     #     print('targets:', targets, 'pred:', pred)
    #     # for j in range(10):
    #     #     mag = torch.norm(gradient[j])
    #     #     cos = (gradient[j] @ avg)/(torch.norm(gradient[j])*torch.norm(avg))
    #     #     print(mag, cos)

    #     if i > 200:
    #         break

    # print(fdsfs)
    
    # confs = torch.Tensor(record).cpu().numpy()
    
    # record = []
    # for i, batch in enumerate(ood_dataloader):
    #     inputs, targets = batch
    #     inputs = inputs.cuda()
    #     targets = targets.cuda()
        
    #     # model.zero_grad()
    #     # outputs = model(inputs)
    #     # pred = torch.argmax(outputs, 1)
    #     # # if j==0:
    #     # #     print('targets:', targets, 'pred:', torch.argmax(outputs, 1))
    #     # label = torch.Tensor([pred]).cuda()
    #     # label = label.to(torch.int64)
    #     # loss_ce = smooth_crossentropy(outputs, label, smoothing=0.0).mean()
    #     # loss_ce.backward()
    #     # mag = torch.norm(get_model_grad_vec_torch_2(model))
    #     # record.append(mag)

    #     gradient = []
    #     for j in range(10):
    #         model.zero_grad()
    #         outputs = model(inputs)
    #         # if j==0:
    #         #     print('targets:', targets, 'pred:', torch.argmax(outputs, 1))
    #         label = torch.Tensor([j]).cuda()
    #         label = label.to(torch.int64)
    #         loss_ce = smooth_crossentropy(outputs, label, smoothing=0.0).mean()
    #         loss_ce.backward()
    #         gradient.append(get_model_grad_vec_torch_2(model))
        
    #     sum = 0
    #     for j in range(10):
    #         sum = sum + gradient[j]
    #     avg = sum/10

    #     pred = torch.argmax(outputs, 1)
    #     mag = torch.norm(gradient[pred])
    #     cos = (gradient[pred] @ avg)/(torch.norm(gradient[pred])*torch.norm(avg))
    #     record.append(torch.norm(avg)/mag)
       
    #     if i > 200:
    #         break

    # ood_confs = torch.Tensor(record).cpu().numpy()
    # print('confs: ', confs.shape, ',ood_confs: ', ood_confs.shape)
    # print('confs:', confs)
    # print('ood_confs:', ood_confs)
    # results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
    # print(base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
    # results = cal_metric(-confs, -ood_confs) ##### 大于 threshold 为 ID data
    # print(base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])


    # print(fsdfs)
    ####################################################################################################
    with torch.no_grad():
        test_forward_feature, test_label, feature_test_dataloader = obtain_feature_loader(model, test_dataloader, save_dir, args.data+'_test', args, check)
        train_forward_feature, train_label, feature_train_dataloader = obtain_feature_loader(model, train_dataloader, save_dir, args.data+'_train', args, check)
        ood_forward_feature, ood_label, feature_ood_dataloader = obtain_feature_loader(model, ood_dataloader, save_dir, args.data +'_'+ args.ood_data, args, check)

    print('test_feature:', test_forward_feature.min(), '-', test_forward_feature.max(), test_forward_feature.shape, test_forward_feature.mean())
    print('train_feature:', train_forward_feature.min(), '-', train_forward_feature.max(), train_forward_feature.shape, train_forward_feature.mean())
    print('ood_feature:', ood_forward_feature.min(), '-', ood_forward_feature.max(), ood_forward_feature.shape, ood_forward_feature.mean())

    # ood_forward_feature, ood_label, feature_ood_dataloader = obtain_feature_loader(model, ood_dataloader, save_dir, args.data +'_'+ args.ood_data, args, check)

    # # print(fsdfsd)
    # ###### forward feature principle component calculation
    # # from sklearn.covariance import EmpiricalCovariance
    # # ec = EmpiricalCovariance(assume_centered=False)
    # # ec.fit(train_forward_feature)
    # # eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    # # NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[0:classnums]]).T)
    # # # print(NS.shape)
    # # a = eig_vals
    # # print(check, a[0:1].sum()/a.sum(), a[0:2].sum()/a.sum(), a[0:3].sum()/a.sum(), a[0:10].sum()/a.sum(),  a[0:10])
    # # print(fsdfsd)
   
    # from sklearn.manifold import TSNE
    # fig_save_dir = '2023_05_31/'
    # os.makedirs(fig_save_dir, exist_ok=True)
    # # fig_save_dir = 'low-dimensional feature visualization/' + 'gradient' + '/PCA/'

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
    #     # plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(fig_save_dir+save_name+'.png', dpi=800)
    #     return
    
    
    # # normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    # # feat_log = normalizer(feat_log)
    # # feat_log_val = normalizer(feat_log_val)
    # # ood_feat_log = normalizer(ood_feat_log)

    # feat = np.concatenate((train_forward_feature, test_forward_feature, ood_forward_feature), axis=0)
    # print(feat.shape)
    # ts = TSNE(n_components=2, init='pca', random_state=0)
    # x_ts = ts.fit_transform(feat)
    # x = x_ts[0:50000]  # train
    # y = x_ts[50000:60000] # test
    # z = x_ts[60000:70000] # ood
    
    # # x = train_forward_feature @ NS
    # # y = test_forward_feature @ NS
    # # z = ood_forward_feature @ NS
    
    # scatter_plot_single(x, args.data + '_train_tsne' + check)
    # scatter_plot_single(y,args.data + '_test_tsne' + check)
    # scatter_plot_single(z, args.data + '_tsne' + args.ood_data + check)
    
    # # scatter_plot(x, y, z, args.data + '_Outlier')
    # print('finish plot')
    # print(fdsf)



    if base_method == 'msp':
        with torch.no_grad():
            if args.model =='vit':
                confs = msp(feature_test_dataloader, net.head.layers[-1])
                ood_confs =  msp(feature_ood_dataloader, net.head.layers[-1])  
            elif args.data == 'cifar10' and args.model =='densenet':
                confs = msp(feature_test_dataloader, net.classifier)
                ood_confs =  msp(feature_ood_dataloader, net.classifier)
            elif args.data == 'imagenet' and args.model =='mobilenet':
                confs = msp(feature_test_dataloader, net.classifier[-1])
                ood_confs =  msp(feature_ood_dataloader, net.classifier[-1])
            elif args.model =='resnet18':
                confs = msp(feature_test_dataloader, net.fc)
                ood_confs =  msp(feature_ood_dataloader, net.fc)
            else: 
                confs = msp(test_dataloader, net)
                ood_confs =  msp(ood_dataloader, net)  
    elif base_method == 'energy':
        with torch.no_grad():
            temper = 1
            if args.model =='vit':
                confs = energy(feature_test_dataloader, net.head.layers[-1], temper)
                ood_confs = energy(feature_ood_dataloader, net.head.layers[-1], temper)
            elif args.data == 'cifar10' and args.model =='densenet':
                confs = energy(feature_test_dataloader, net.classifier, temper)
                ood_confs = energy(feature_ood_dataloader, net.classifier, temper)
            elif args.data == 'imagenet' and args.model =='mobilenet':
                confs = energy(feature_test_dataloader, net.classifier[-1], temper)
                ood_confs = energy(feature_ood_dataloader, net.classifier[-1], temper)
            elif args.model =='resnet18':
                confs = energy(feature_test_dataloader, net.fc, temper)
                ood_confs = energy(feature_ood_dataloader, net.fc, temper)
            else:
                confs = energy(test_dataloader, net, temper)
                ood_confs = energy(ood_dataloader, net, temper)
    elif base_method == 'maha':
        num_classes = classnums       
        confs, ood_confs = mahalanobis_official(net, train_dataloader, test_dataloader, ood_dataloader, num_classes, magnitude=0.01, save_path=save_dir, save_name='mahalanobis_'+args.data+'_'+args.ood_data, data=args.data)
    elif base_method == 'odin':
        if args.data =='cifar10':
            # odin_epsilon = 0.01
            odin_epsilon = -0.001
        elif args.data == 'imagenet':
            odin_epsilon = 0.005
        odin_temperature = 1000
        # odin_temperature = 1
        print('eps:', odin_epsilon, 'temper:', odin_temperature)
        confs = odin(test_dataloader, net, odin_temperature, odin_epsilon)
        ood_confs = odin(ood_dataloader, net, odin_temperature, odin_epsilon)
    elif base_method == 'grad_norm':
        ####### batch_size should be set to 1
        num_classes = classnums
        gradnorm_temperature = 1
        kl_loss = False
        print('kl_loss:', kl_loss)
        # confs = grad_norm(test_dataloader, net, gradnorm_temperature, num_classes, kl_loss).flatten()
        # # np.save('grad_norm_imagenet_test_bit.npy', confs)
        # # confs = np.load('grad_norm_imagenet_test_bit.npy')
        # ood_confs = grad_norm(ood_dataloader, net, gradnorm_temperature, num_classes, kl_loss).flatten()

        if not os.path.exists('grad_norm_imagenet_test_vit.npy'):
            confs = grad_norm(feature_test_dataloader, net.head.layers[-1], gradnorm_temperature, num_classes, kl_loss).flatten()
            np.save('grad_norm_imagenet_test_vit.npy', confs)
        else:
            confs = np.load('grad_norm_imagenet_test_vit.npy')
        ood_confs = grad_norm(feature_ood_dataloader, net.head.layers[-1], gradnorm_temperature, num_classes, kl_loss).flatten()
    elif base_method == 'react':
        temper = 1
        # threshold = 1e6
        def cal_threshold(feature, percent):
            feature = feature.flatten()
            threshold = np.percentile(feature, percent*100) # percent的数小于threshold
            # print((feature<threshold).sum())
            return threshold
        if args.data == 'cifar10' or args.data == 'cifar100':
            percent = 0.95
        elif args.data == 'imagenet':
            percent = 0.90
        threshold = cal_threshold(test_forward_feature, percent=percent)
        print('threshold:', threshold, ',percent:', percent)
        confs = react(feature_test_dataloader, net, temper, threshold, args)
        ood_confs = react(feature_ood_dataloader, net, temper, threshold, args)
    elif base_method == 'bats':
        temper = 1
        # lams = np.arange(0.1, 10.0, 0.25)
        # lams = [0.10, 9.85]
        if args.data == 'cifar10' or args.data == 'cifar100':
            # lams = [0.10]
            lams = np.arange(0.1, 10, 0.25)
        elif args.data == 'imagenet':
            # lams = [9.85]
            lams = np.arange(0.1, 10, 0.25)
        for lam in lams:
            net = copy.deepcopy(model)
            if (args.data =='cifar10' or args.data == 'cifar100') and args.model =='resnet18':
                # lam = 3.25
                truncated_module = ['conv5_x.1.residual_function.4'] 
            elif (args.data =='cifar10' or args.data == 'cifar100') and args.model =='wideresnet':
                truncated_module = ['f.4_normalization'] 
            elif (args.data =='cifar10' or args.data == 'cifar100') and args.model =='densenet':
                truncated_module = ['features.norm5'] 
            elif (args.data =='cifar10' or args.data == 'cifar100') and args.model =='resnet18-2':
                truncated_module = ['bn1'] 
            elif args.data == 'imagenet' and args.model =='resnet50':
                # lam = 1.05
                truncated_module = ['layer4.2.bn3']
            elif args.data == 'imagenet' and args.model =='vit':
                truncated_module = ['backbone.pre_norm']
            elif args.data == 'imagenet' and args.model =='mobilenet':
                truncated_module = ['features.18.1']
            
            print('lam:', lam, ',bn_module:', truncated_module)
            for n, module in net.named_modules():
                if n in truncated_module:
                    Trunc_BN = TrBN(module, lam)
                    _set_module(net, n, Trunc_BN)
                    upper_bound, lower_bound = Trunc_BN.get_static()
            # with torch.no_grad():
            #     test(test_dataloader, model)
            with torch.no_grad():
                confs = bats(test_dataloader, net, temper, upper_bound, lower_bound)
                ood_confs = bats(ood_dataloader, net, temper, upper_bound, lower_bound)
            # print(confs.shape, ood_confs.shape)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print(base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # print(fsds)
    elif base_method == 'knn':
        # K = 5
        # for K in [1,5,10,100,1000,1200]:
        # for K in [1, 5, 10, 50, 100, 200]:
        if args.data == 'cifar10' or args.data == 'cifar100':
            K=5
        elif args.data == 'imagenet':
            K=10
        print(train_forward_feature.shape, test_forward_feature.shape, ood_forward_feature.shape)
        confs = knn(train_forward_feature, test_forward_feature, K)
        ood_confs = knn(train_forward_feature, ood_forward_feature, K)
        # results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
        # print(K, base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # print(dffs)
    elif base_method == 'vim':
        # DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
        # DIM = classnums
        # dims = [10, 50, 100, 150, 200, 300, 400, 500] ### cifar10
        dims =[100]
        # dims = [10, 100, 300, 512, 1000, 1500, 2000]  ### imagenet
        for DIM in dims:
            confs = vim(train_forward_feature, test_forward_feature, net, DIM)
            ood_confs = vim(train_forward_feature, ood_forward_feature, net, DIM)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print(base_method, ', DIM:', DIM, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        
        # DIM = 10
        # a, b  = vim(train_forward_feature, test_forward_feature, net, DIM)
        # a, c = vim(train_forward_feature, ood_forward_feature, net, DIM)
        # for K in [1,5,10,100,1000,1200]:
        #     confs = knn(a, b, K)
        #     ood_confs = knn(a, c, K)
        #     results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
        #     print(K, base_method, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
        # print(fsdfs)
    elif base_method == 'residual':
        # DIM = classnums
        # dims = [10, 50, 100, 150, 200, 300, 400, 500]
        dims = [100]
        for DIM in dims:
            confs = residual(train_forward_feature, test_forward_feature, net, DIM)
            ood_confs = residual(train_forward_feature, ood_forward_feature, net, DIM)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print(base_method, ', DIM:', DIM, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
    elif base_method == 'low_feature':
        DIM = 10
        # K = 1
        for K in [1, 5, 10, 50, 100, 200]:
            confs = feature_low_dimension(train_forward_feature, test_forward_feature, DIM, K)
            ood_confs = feature_low_dimension(train_forward_feature, ood_forward_feature, DIM, K)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print(base_method, ', K:', K, ', DIM:', DIM, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
    elif base_method == '1':
        temper = 1
        # threshold = 1e6
        def cal_threshold(feature, percent):
            feature = feature.flatten()
            threshold = np.percentile(feature, percent*100) # percent的数小于threshold
            # print((feature<threshold).sum())
            return threshold
        if args.data == 'cifar10':
            percent = 0.95
        elif args.data == 'imagenet':
            percent = 0.90
        threshold = cal_threshold(test_forward_feature, percent=percent)
        print('threshold:', threshold, ',percent:', percent)
        confs_1 = react(feature_test_dataloader, net, temper, threshold)
        ood_confs_1 = react(feature_ood_dataloader, net, temper, threshold)
        print('ood_confs_1:', ood_confs_1.mean())
        np.save('2023_05_31_detect_base_method/detection score/react_'+args.data+'.npy', confs_1)
        np.save('2023_05_31_detect_base_method/detection score/react_'+args.ood_data+'.npy', ood_confs_1)
        
        ####### batch_size should be set to 1
        num_classes = classnums
        gradnorm_temperature = 1
        kl_loss = False
        print('kl_loss:', kl_loss)
        confs_2 = grad_norm(test_dataloader, net, gradnorm_temperature, num_classes, kl_loss) 
        ood_confs_2 = grad_norm(ood_dataloader, net, gradnorm_temperature, num_classes, kl_loss) 
        print('ood_confs_2:', ood_confs_2.mean())
        np.save('2023_05_31_detect_base_method/detection score/grad_norm_'+args.data+'.npy', confs_2)
        np.save('2023_05_31_detect_base_method/detection score/grad_norm_'+args.ood_data+'.npy', ood_confs_2)
        
        for a in [0.000001]:
            confs = confs_1 + a*confs_2
            ood_confs = ood_confs_1 + a*ood_confs_2
            print('confs: ', confs.shape, ',ood_confs: ', ood_confs.shape)
            results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
            print('a=', a , ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
    else:
        print('No exist method')

    # os.makedirs('2023_05_31_detect_base_method/detection score/'+args.data+'/', exist_ok=True)
    # np.save('2023_05_31_detect_base_method/detection score/'+args.data+'/'+base_method+'_'+args.data+'.npy', confs)
    # np.save('2023_05_31_detect_base_method/detection score/'+args.data+'/'+base_method+'_'+args.ood_data+'.npy', ood_confs)
    print('confs: ', confs.shape, ',ood_confs: ', ood_confs.shape)
    # print('confs:', confs)
    # print('ood_confs:', ood_confs)
    results = cal_metric(confs, ood_confs) ##### 大于 threshold 为 ID data
    print(base_method, ',', args.ood_data, ',', args.model, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])
    results = cal_metric(-confs, -ood_confs) ##### 大于 threshold 为 ID data
    print(base_method, ',', args.ood_data, ',', args.model, ', FPR:', 100*results['FPR'], ', AUROC:', 100*results['AUROC'])


    


    

