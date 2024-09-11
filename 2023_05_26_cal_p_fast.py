import argparse
from re import I
import torch
import os
import numpy as np

from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar10,Cifar100
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.multi_gpu import setup_for_distributed

import sys
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
from functorch import make_functional_with_buffers, vmap, grad, jvp, vjp
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from functools import partial
import copy
from data.imagenet import ImageNet

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import timedelta
from time import sleep
import multiprocessing
import math
from data.dataloader_for_ood_2 import IN_DATA


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
        vec.append(p.grad.data.detach().reshape(-1)) 
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
        self.fc = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        y = self.fc(x)
        return y


def matrix_jacobian_product(model, x, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(N, k), N为样本个数
    def vjp_single(model, x, M, kernel, z_theta, I_2):
        bz = x.shape[0]
        fmodel, params, buffers = make_functional_with_buffers(model)
        def compute_loss_stateless_model(params, buffers, sample):
            predictions = fmodel(params, buffers, sample)
            loss = torch.sum(M*predictions)
            return loss
        ft_compute_grad = grad(compute_loss_stateless_model)
        z_t_ = ft_compute_grad(params, buffers, x)
        
        # fmodel, params, buffers = make_functional_with_buffers(model)
        # def compute_loss_stateless_model(params, buffers, sample):
        #     predictions = fmodel(params, buffers, sample)
        #     return predictions
        # function = partial(compute_loss_stateless_model, buffers=buffers, sample=x)

        # (_, vjp_fn) = vjp(function, params)
        # z_t_2 = vjp_fn(M)[0].detach()
        # print(z_t_2[0][0:5])
        # print(z_t_[0][0:5], z_t_2[0][0:5])
        # print(fsdfs)

        #### 将梯度 flatten 成一维向量
        z_t = []
        for i in range(len(z_t_)):
            z_t.append(z_t_[i].detach().view(-1))
        z_t = torch.cat(z_t, dim=0)
        del z_t_
        if kernel == 'NFK':
            z_t = I_2 * (z_t - torch.sum(M)*z_theta)
        return z_t
        
    mjp = vmap(vjp_single, in_dims=(None, None, 1, None, None, None), out_dims=(1))
    omiga = mjp(model, x, M, kernel, z_theta, I_2)

    # omiga = []
    # for i in range(M.shape[1]):
    #     M_current = M[:,i]
    #     model.zero_grad()
    #     predictions = model(x)
    #     loss = torch.sum(M_current*predictions)
    #     loss.backward()
    #     z_t = get_model_grad_vec_torch_2(model)
    #     if kernel == 'NFK':
    #         z_t = I_2 * (z_t - torch.sum(M_current)*z_theta)
    #     omiga.append(z_t.unsqueeze(1))

    # omiga = torch.cat(omiga, dim=1)  

    return omiga


def matrix_jacobian_product_2(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None, device=None, world_size=2, rank=0, num_example=10000):
    ## M:(N, k), N为样本个数
    interval = int(num_example/world_size)
    num = 0 + rank*interval
    omiga = 0
    # print('num:', num)
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        omiga = omiga + matrix_jacobian_product(model, inputs, M[num:num+inputs.shape[0]], kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        num = num + inputs.shape[0]
        # print('num:', num)
    # print(omiga.shape)
    omiga = omiga.contiguous()
    torch.distributed.barrier()
    # print('omiga:', omiga[0,0])
    # print(omiga.shape)
    omiga = reduce_value(omiga)
    # print('omiga:', omiga[0])
    # print(omiga.shape)
    # print(fdsdfs)
    return omiga

def matrix_jacobian_product_3(model, x, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(N, k), N为样本个数  
    model.zero_grad()
    y = model(x)
    loss = torch.sum(y*M)
    loss.backward()
    z_t = get_model_grad_vec_torch_2(model)
    
    if kernel == 'NFK':
        z_t = I_2 * (z_t - torch.sum(M)*z_theta)
        
    return z_t


def matrix_jacobian_product_4(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None, device=None, world_size=2, rank=0, num_example=10000):
    M_update = []
    for i in range(M.shape[1]):
        M_current = M[:,i].to(device)
        if rank == 0:
            print('mjp, i:', i, 'M_current:', M_current.shape)
        ## M:(N, k), N为样本个数
        interval = math.floor(num_example/world_size)
        num = 0 + rank*interval
        omiga = 0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            omiga = omiga + matrix_jacobian_product_3(model, inputs, M_current[num:num+inputs.shape[0]], kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
            num = num + inputs.shape[0]
        
        torch.distributed.barrier()
        omiga = reduce_value(omiga)

        M_update.append(omiga.cpu().unsqueeze(1))

    M_update = torch.cat(M_update, dim=1)
    return M_update


def matrix_jacobian_product_dataloader(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(N, k), N为样本个数
    M_update = []
    for i in range(M.shape[1]):
        M_current = M[:,i].cuda()
        print('mjp, i:',i,'M_current:', M_current.shape)
        num = 0
        z_t = 0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()

            # bz = inputs.shape[0]
            # fmodel, params, buffers = make_functional_with_buffers(model)
            # def compute_loss_stateless_model(params, buffers, sample):
            #     predictions = fmodel(params, buffers, sample)
            #     loss = torch.sum(M_current[num:num+inputs.shape[0]]*predictions)
            #     return loss
            # ft_compute_grad = grad(compute_loss_stateless_model)
            # z_t_ = ft_compute_grad(params, buffers, inputs)
            # #### 将梯度 flatten 成一维向量
            # z_temp = []
            # for i in range(len(z_t_)):
            #     z_temp.append(z_t_[i].view(-1))
            # z_temp = torch.cat(z_temp, dim=0)
            # if kernel == 'NFK':
            #     z_t = I_2 * (z_t - torch.sum(M)*z_theta)

            model.zero_grad()
            y = model(inputs)
            loss = torch.sum(y*M_current[num:num+inputs.shape[0]])
            loss.backward()
            z_temp = get_model_grad_vec_torch_2(model)
            
            z_t = z_t + z_temp
            num = num + inputs.shape[0]
        
        if kernel == 'NFK':
            z_t = I_2 * (z_t - torch.sum(M_current)*z_theta)
        M_update.append(z_t.unsqueeze(1).cpu())

    M_update = torch.cat(M_update, dim=1)
    print('M_update:', M_update.shape)
    
    return M_update


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
    # omiga = jvp_single(model, x, M[:,0], kernel, z_theta, I_2)

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


def jacobian_matrix_product_2(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None, device=None, world_size=2):
    ## M:(|theta|, k)，|theta|为参数维度
    omiga = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        omiga.append(jacobian_matrix_product(model, inputs, M, kernel=kernel, z_theta=z_theta, I_2=I_2).detach())
    omiga = torch.cat(omiga, dim=0)
    omiga = omiga.contiguous()
    torch.distributed.barrier()
    # print('omiga:', omiga)
    omiga_list = [torch.zeros(omiga.shape).to(device) for _ in range(world_size)]
    dist.all_gather(omiga_list, omiga)
    omiga = torch.cat(omiga_list, dim=0)
    # print('omiga.shape:', omiga.shape)
    # print('omiga:', omiga)
    # print(fsdfs)
    return omiga

def jacobian_matrix_product_3(model, x, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(|theta|, k)，|theta|为参数维度   
    if kernel == 'NFK':
        M = I_2*M
    
    fmodel, params, buffers = make_functional_with_buffers(model)
    def compute_loss_stateless_model(params, buffers, sample):
        predictions = fmodel(params, buffers, sample)
        return predictions
    
    function = partial(compute_loss_stateless_model, buffers=buffers, sample=x)
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
        
    if kernel == 'NFK':
        grad2 = grad2 - torch.dot(z_theta, M)
        
    return grad2


def jacobian_matrix_product_4(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None, device=None, world_size=2, rank=0):
    ## M:(|theta|, k)，|theta|为参数维度
    M_update = []
    for i in range(M.shape[1]):
        M_current = M[:,i].to(device)
        if rank==0:
            print('jmp, i:',i,'M_current:', M_current.shape)
        omiga = []
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            omiga.append(jacobian_matrix_product_3(model, inputs, M_current, kernel=kernel, z_theta=z_theta, I_2=I_2).detach())
        omiga = torch.cat(omiga, dim=0)
        torch.distributed.barrier()
        omiga_list = [torch.zeros(omiga.shape).to(device) for _ in range(world_size)]
        dist.all_gather(omiga_list, omiga)
        omiga = torch.cat(omiga_list, dim=0)

        M_update.append(omiga.cpu().unsqueeze(1))

    M_update = torch.cat(M_update, dim=1)
    return M_update

def jacobian_matrix_product_dataloader(model, dataloader, M, kernel='NTK', z_theta=None, I_2=None):
    ## M:(|theta|, k)，|theta|为参数维度
    M_update = []
    for i in range(M.shape[1]):
        M_current = M[:,i].cuda()
        print('jmp, i:', i, 'M_current:', M_current.shape)
        if kernel == 'NFK':
            M_current = I_2*M_current
        M_update_temp = []
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            # model.zero_grad()
            # bz = inputs.shape[0]
            # fmodel, params, buffers = make_functional_with_buffers(model)
            # def compute_loss_stateless_model_2(params, buffers, sample):
            #     batch = sample.unsqueeze(0)
            #     predictions = fmodel(params, buffers, batch).squeeze()
            #     # print(predictions.shape)
            #     return predictions
            # start = time.time()
            # ft_compute_grad = grad(compute_loss_stateless_model_2)
            # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
            # ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs)
            # ft_per_sample_grads_ = []
            # for i in range(len(ft_per_sample_grads)):
            #     ft_per_sample_grads_.append(ft_per_sample_grads[i].detach().view(bz, -1))
            # ft_per_sample_grads_ = torch.cat(ft_per_sample_grads_, dim=1)
            # grad2 = (ft_per_sample_grads_ @ M_current.unsqueeze(1)).squeeze()
            # end = time.time()
            # print(grad2.shape, 'time:', end-start)
        
            # model.zero_grad()
            # # bz = inputs.shape[0]
            fmodel, params, buffers = make_functional_with_buffers(model)
            def compute_loss_stateless_model(params, buffers, sample):
                # batch = sample.unsqueeze(0)
                predictions = fmodel(params, buffers, sample)#.squeeze()
                # print(predictions.shape)
                return predictions
            # start = time.time()
            function = partial(compute_loss_stateless_model, buffers=buffers, sample=inputs)
            _, M_temp, _ = make_functional_with_buffers(model)
            M_temp = list(M_temp)
            idx = 0
            for i in range(len(M_temp)):
                arr_shape = M_temp[i].shape
                size = arr_shape.numel()
                M_temp[i] = M_current[idx:idx+size].reshape(arr_shape).clone()
                idx += size
            M_temp = tuple(M_temp)     
            value, grad2 = jvp(function, (params,), (M_temp,))
            grad2 = grad2.detach()
            # end = time.time()
            # print(grad3.shape, 'time:', end-start)
            # print(torch.norm(grad2-grad3, p=np.inf))
            # print(fsdfs)

            if kernel == 'NFK':
                grad2 = grad2 - torch.dot(z_theta, M_current)

            # def jvp_sample(model, inputs, M_current, kernel, z_theta):
            #     model.zero_grad()
            #     fmodel, params, buffers = make_functional_with_buffers(model)
            #     _, M_temp, _ = make_functional_with_buffers(model)
            #     def compute_loss_stateless_model(params, buffers, sample):
            #         predictions = fmodel(params, buffers, sample)
            #         return predictions
            #     function = partial(compute_loss_stateless_model, buffers=buffers, sample=inputs)

            #     M_temp = list(M_temp)
            #     idx = 0
            #     for j in range(len(M_temp)):
            #         arr_shape = M_temp[j].shape
            #         size = arr_shape.numel()
            #         M_temp[j] = M_current[idx:idx+size].reshape(arr_shape).clone()
            #         idx += size
            #     M_temp = tuple(M_temp)     
                    
            #     _, grad3 = jvp(function, (params,), (M_temp,))
            
            #     if kernel == 'NFK':
            #         grad3 = grad3 - torch.dot(z_theta, M_current)
            #     return grad3
            # grad3 = jvp_sample(model, inputs, M_current, kernel, z_theta)
            # print('grad.shape:', grad3.shape, 'grad2.shape:', grad2.shape)
            # print('diff:', torch.norm(grad2.squeeze()-grad3, p=np.inf))
            # print(fdsfds)

            M_update_temp.append(grad2)
        
        M_update_temp = torch.cat(M_update_temp, dim=0)
        # print('M_update_temp: ', M_update_temp.shape)
        M_update.append(M_update_temp.unsqueeze(1).cpu())
        # torch.cuda.empty_cache()

    M_update = torch.cat(M_update, dim=1)
    print('M_update:', M_update.shape)
    return M_update


def pre_compute(model, dataloader, save_dir, device, rank):
    sum_ = 0
    nexample = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        nexample += inputs.shape[0]
        model.zero_grad()
        
        y = model(inputs)
        loss = torch.mean(y)
        loss.backward()
        a = get_model_grad_vec_torch_2(model)
        sum_ = sum_ + a*inputs.shape[0]
    
    nexample = torch.Tensor([nexample]).to(device)
    torch.distributed.barrier()
    sum_ = reduce_value(sum_)
    nexample = reduce_value(nexample)
    z_theta = sum_/nexample
    print('nexample:', nexample)
    if rank == 0:
        np.save(save_dir + 'z_theta.npy', z_theta.cpu().detach().numpy())

    L = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        for i in range(inputs.shape[0]):
            model.zero_grad()
            
            y_i = model(inputs[i].unsqueeze(0))
            y_i.backward()
            z_i = get_model_grad_vec_torch_2(model)
            u_i =  z_i - z_theta
            ### diagonal approximation
            L_i = torch.square(u_i)
            L = L + L_i
    torch.distributed.barrier()
    L = reduce_value(L)
    L = L / nexample
    # L^(-1/2)
    I_2 = 1/ torch.sqrt(L)
    #### count inf_value in I_2
    count = (I_2==np.inf).sum()
    ratio = 100*(count/len(I_2))
    print('inf-value, count:', count, 'ratio:', float(ratio), '%')
    #### deal with inf_value in I_2
    I_2_ = torch.where(torch.isinf(I_2), torch.full_like(I_2, 1), I_2)
    print('I_2_.max: ', float(I_2_.max()))
    I_2 = torch.where(I_2_==1, torch.full_like(I_2_, float(I_2_.max())), I_2_)

    # I_2 = torch.ones(z_theta.shape[0]).to(device)
    return z_theta, I_2


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


def truncated_svd_dataloader(model, dataloader, k=1, iter=10, kernel='NTK', z_theta=None, I_2=None, save_dir=None):
    # n = x.shape[0]  # 样本数
    n = 0
    for batch in dataloader:
        inputs, targets = batch
        n = n + inputs.shape[0]
    print('sample number:', n)

    omiga = torch.randn((n, k)).cuda()
    omiga, _ = torch.linalg.qr(omiga)
    omiga = matrix_jacobian_product_dataloader(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
    # print('omiga:', omiga.shape)
    # np.save(save_dir + 'omiga.npy', omiga.cpu().detach().numpy())
    # omiga = torch.from_numpy(np.load(save_dir + 'omiga.npy'))

    # omiga.shape: (M, k)
    # omiga = torch.randn((11173962, k))#.cuda()

    # for batch in dataloader:
    #     inputs, targets = batch
    #     inputs = inputs.cuda()
    #     b = matrix_jacobian_product(model, inputs, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
    #     print(torch.norm(a-b,p=np.inf))
    # print(fsdfsd)
    
    # print('omiga:', omiga.shape)
    for k in range(iter):
        print('k=', k)
        omiga = jacobian_matrix_product_dataloader(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # print('omiga:', omiga.shape)
        # np.save(save_dir + str(k) +'_jmp_omiga.npy', omiga.cpu().detach().numpy())
        print('k=', k)
        omiga = matrix_jacobian_product_dataloader(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # omiga = omiga / torch.norm(omiga, p=np.inf)
        omiga, _ = torch.linalg.qr(omiga)
        # np.save(save_dir + str(k) +'_mjp_omiga.npy', omiga.cpu().detach().numpy())

    # print('omiga:', omiga.shape)
    # np.save(save_dir + 'omiga.npy', omiga.cpu().detach().numpy())
    b = jacobian_matrix_product_dataloader(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
    print('b:', b.shape)
    # np.save(save_dir + 'b.npy', b.cpu().detach().numpy())
    p, sigma, q = torch.svd(b.T)
    p = omiga@p
    return p, sigma, q


def truncated_svd_2(model, dataloader, n=50000, k=1, iter=10, kernel='NTK', z_theta=None, I_2=None):
    # n = 0
    # for batch in dataloader:
    #     inputs, targets = batch
    #     n = n + inputs.shape[0]
    # print('sample number:', n)

    omiga = torch.randn((n, k)).cuda()
    omiga, _ = torch.linalg.qr(omiga)
    # print(omiga.shape, omiga[:,0].shape, torch.norm(omiga[:,0]))
    omiga = matrix_jacobian_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
    # print('omiga:', omiga.shape)
    for i in range(iter):
        print(i)
        omiga = jacobian_matrix_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # print('omiga:', omiga.shape)
        # print(fdsfsd)
        print(i)
        omiga = matrix_jacobian_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
        # omiga = omiga / torch.norm(omiga, p=np.inf)
        omiga, _ = torch.linalg.qr(omiga)

    b = jacobian_matrix_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2).detach()
    p, sigma, q = torch.svd(b.T)
    p = omiga@p
    return p, sigma, q


def orth(omiga, device, omiga_name=[]):
    for path in omiga_name:
        a = torch.from_numpy(np.load(path))
        # print('a:', a.shape)
        for i in range(a.shape[1]):
            b = a[:,i].to(device)
            # print('b:', b.shape)
            omiga = omiga - torch.dot(b, omiga.squeeze())*b.unsqueeze(1)
    omiga = omiga/(torch.norm(omiga)+1e-16)
    return omiga

def load_omiga(omiga_name=[]):
    his_omiga = []
    for path in omiga_name:
        a = torch.from_numpy(np.load(path))
        # a = np.load(path)
        his_omiga.append(a)
    his_omiga = torch.cat(his_omiga, 1)
    # his_omiga = np.concatenate(his_omiga, axis=1)
    return his_omiga

def orth_omiga(omiga, device, his_omiga):
    start_k = his_omiga.shape[1]
    k = omiga.shape[1]
    omiga = omiga.cpu()
    b = torch.cat([his_omiga, omiga],1)
    b, _ = torch.linalg.qr(b)
    # print(torch.norm(b[:,0]),torch.dot(b[:,0],b[:,1]))
    # print(fdsfsd)
    b = b[:,start_k:start_k+k]
    # b = b.to(device)
    return b

def qr_alter(omiga, his_omiga):
    for i in range(omiga.shape[1]):
        omiga[:,i] = omiga[:,i] - torch.sum(torch.mm(omiga[:,i].unsqueeze(0), his_omiga)*his_omiga, dim=1) - torch.sum(torch.mm(omiga[:,i].unsqueeze(0), omiga[:,0:i])*omiga[:,0:i], dim=1)
        omiga[:,i] = omiga[:,i]/torch.norm(omiga[:,i])
    # print(omiga, omiga.shape)
    return omiga

def orth_omiga_2(omiga, device, omiga_name=[]):
    omiga_new = []
    for k in range(omiga.shape[1]):
        omiga_cur = omiga[:,k]
        original_omiga_cur = omiga_cur.clone()
        ## history omiga
        for path in omiga_name:
            a = torch.from_numpy(np.load(path))
            for i in range(a.shape[1]):
                b = a[:,i].to(device)
                # print('b:', b.shape)
                omiga_cur = omiga_cur - torch.dot(b, original_omiga_cur)*b
        ## new omiga
        for j in range(len(omiga_new)):
            c = omiga_new[j].squeeze()
            omiga_cur = omiga_cur - torch.dot(c, original_omiga_cur)*c

        omiga_cur = omiga_cur/(torch.norm(omiga_cur))
        omiga_new.append(omiga_cur.unsqueeze(1))
    omiga_new = torch.cat(omiga_new, 1)
    # for path in omiga_name:
    #     a = np.load(path)
    #     for j in np.arange(0, a.shape[1], 10):
    #         b = a[:,j:min(j+10,a.shape[1])]
    #         print(b.shape)
    #         np.save('2023_05_26_cal_p_fast/resnet50_imagenet_NFK/sample_num=50001_k=512/'+ str(j) + '-' + str(min(j+9,a.shape[1]-1))+ '_p.npy', b)
    #     print(sfsf)
    # return
    return omiga_new

def truncated_svd_3(model, dataloader, omiga_name=[], n=50000, k=1, iter=10, kernel='NTK', z_theta=None, I_2=None, device=None, world_size=2, rank=0, his_omiga=None):
    # n = 0
    # for batch in dataloader:
    #     inputs, targets = batch
    #     n = n + inputs.shape[0]
    # print('sample number:', n)
    # if omiga_name!=[]:
    # #     if rank==0:
        # his_omiga = load_omiga(omiga_name)
    #         # his_omiga = mp.Value('f',his_omiga)
    #         his_omiga = multiprocessing.sharedctypes.Array('f', his_omiga, lock=False)
    #         # his_omiga = torch.Tensor.share_memory_(his_omiga)
    #         print('his_omiga:', his_omiga.shape)
    #         # print(torch.multiprocessing.get_all_sharing_strategies())
    #         # print(torch.multiprocessing.get_sharing_strategy())
    #         # print(torch.multiprocessing.set_sharing_strategy(new_strategy))
    #     elif rank!=0:
    #         sleep(10)
    #         print('finish wait')
    #         print('his_omiga:', his_omiga.shape)
    # torch.distributed.barrier()
        # if rank!=0:
        #     print('his_omiga:', his_omiga.shape)
    omiga = torch.randn((n, k)).to(device)
    # omiga = orth(omiga, omiga_name)
    # print(omiga[0:5,0])
    omiga, _ = torch.linalg.qr(omiga)
    # print(omiga[0:5,0])
    # print((omiga.T@omiga).diag(),_)
    # omiga = omiga/(torch.norm(omiga)+1e-16)
    # omiga = orth_omiga_2(omiga, device, omiga_name=[])

    # print(omiga.shape, omiga[:,0].shape, torch.norm(omiga[:,0]))
    omiga = matrix_jacobian_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=rank, num_example=n).detach()
    # print('omiga:', omiga.shape)
    for i in range(iter):
        # print(i)
        if rank==0:
            print('i:', i)
        omiga = jacobian_matrix_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size).detach()
        # print('omiga:', omiga.shape)
        if rank==0:
            print('i:', i)
        omiga = matrix_jacobian_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=rank, num_example=n).detach()
        # omiga = omiga / torch.norm(omiga, p=np.inf)
        # omiga, _ = torch.linalg.qr(omiga)
        
        # omiga = orth(omiga, device, omiga_name)
        if his_omiga!=None:
            omiga = orth_omiga(omiga, device, his_omiga)
        else:
            omiga, _ = torch.linalg.qr(omiga)
        # omiga = orth_omiga_2(omiga, device, omiga_name)
        # print('omiga:', (omiga.T@omiga))
        # print(omiga[0:5,0])
        # print(omiga.shape)

    b = jacobian_matrix_product_2(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size).detach()
    p, sigma, q = torch.svd(b.T)
    # print(p, sigma, q.shape)
    p = omiga@p
    return p, sigma, q

def truncated_svd_4(model, dataloader, omiga_name=[], n=50000, k=1, iter=10, kernel='NTK', z_theta=None, I_2=None, device=None, world_size=2, rank=0, his_omiga=None):
    omiga = torch.randn((n, k))#.to(device)
    omiga, _ = torch.linalg.qr(omiga)
   
    omiga = matrix_jacobian_product_4(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=rank, num_example=n).detach()
    for i in range(iter):
        if rank==0:
            print('iteration:', i)
        omiga = jacobian_matrix_product_4(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=rank).detach()
        omiga = matrix_jacobian_product_4(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=rank, num_example=n).detach()
        if his_omiga!=None:
            # omiga = orth_omiga(omiga, device, his_omiga)
            omiga = qr_alter(omiga, his_omiga)
        else:
            omiga, _ = torch.linalg.qr(omiga)  
        # np.save('omiga_'+str(i)+'.npy', omiga.cpu().detach().numpy())

    b = jacobian_matrix_product_4(model, dataloader, omiga, kernel=kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=rank).detach()
    # p, sigma, q = torch.linalg.svd((b.T).to(device), full_matrices=False, driver='gesvd')
    # p = p.cpu()
    # sigma = sigma.cpu()
    # q = q.cpu()
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
    

def cal_fisher_vector_dataloader(model, dataloader, kernel='NTK', z_theta=None, I_2=None, device=None, world_size=2):
    fisher_vector = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        bz = inputs.shape[0]
        fmodel, params, buffers = make_functional_with_buffers(model)
        def compute_loss_stateless_model(params, buffers, sample):
            predictions = fmodel(params, buffers, sample.unsqueeze(0)).squeeze()
            return predictions
        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs)
        ft_per_sample_grads_ = []
        for i in range(len(ft_per_sample_grads)):
            ft_per_sample_grads_.append(ft_per_sample_grads[i].detach().view(bz, -1))
        ft_per_sample_grads_ = torch.cat(ft_per_sample_grads_, dim=1)
        
        if kernel == 'NFK':
            ft_per_sample_grads_ =  ft_per_sample_grads_ - z_theta
            def dot_per_element(I_2, z_i):
                out = I_2 * z_i
                return out
            dot = vmap(dot_per_element, in_dims=(None, 0))
            ft_per_sample_grads_ = dot(I_2, ft_per_sample_grads_)

        # fisher_vector.append(ft_per_sample_grads_.cpu())
        fisher_vector.append(ft_per_sample_grads_)

    fisher_vector = torch.cat(fisher_vector, dim=0)

    fisher_vector = fisher_vector.contiguous()
    torch.distributed.barrier()
    fisher_vector_list = [torch.zeros(fisher_vector.shape).to(device) for _ in range(world_size)]
    dist.all_gather(fisher_vector_list, fisher_vector)
    fisher_vector = torch.cat(fisher_vector_list, dim=0)
    
    print('fisher_vector:', fisher_vector.shape)

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
    
    return fisher_vector


def obtain_feature_embedding(v_x, p):
    # v_x:(bz,M), p:(M,k), M为参数维度,k为降维维度
    feature = torch.mm(v_x, p)
    return feature

    # fea = []
    # for i in range(p.shape[0]):
    #     fea.append((torch.dot(v_x, p[i])/N).unsqueeze(0))
    # feature = torch.cat(fea, 0)
    # return feature


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
            P = I_2*P

        P_temp = list(P_temp)
        idx = 0
        for i in range(len(P_temp)):
            arr_shape = P_temp[i].shape
            size = arr_shape.numel()
            P_temp[i] = P[idx:idx+size].reshape(arr_shape).clone()
            idx += size
        P_temp = tuple(P_temp)     
            
        value, grad = jvp(function, (params,), (P_temp,))
        if kernel == 'NFK':
            grad = grad - torch.dot(z_theta, P)
        return grad
    
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


def test(data_loader, model, device):
    num = 0
    num_all = 0
    model.eval()
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
      
        predictions = model(inputs)

        correct = (torch.argmax(predictions, 1) == targets)
        num = num + (torch.nonzero(correct==True)).shape[0]
        num_all = num_all + targets.shape[0]
    acc = num/num_all
    print('test_acc:', acc)
    return acc


def reduce_value(value, op=dist.ReduceOp.SUM):
    world_size = dist.get_world_size()
    if world_size < 2:  # single GPU
        return value
 
    with torch.no_grad():
        dist.all_reduce(value, op)
        return value
    
def setup(global_rank, world_size, port='12345'):
    # 配置Master Node的信息
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # 初始化Process Group
    # 关于init_method, 参数详见https://pytorch.org/docs/stable/distributed.html#initialization
    dist.init_process_group("nccl", init_method='env://', rank=global_rank, world_size=world_size, timeout=timedelta(seconds=5))

def cleanup():
    dist.destroy_process_group()

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=1, type=int, help="Number of CPU threads for dataloaders.")
    #parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--data", default='cifar10', type=str, help="cifar10 or cifar100.")
    parser.add_argument("--model", default='resnet18', type=str, help="resnet18 or wideresnet or VGG16BN")
    parser.add_argument("--save_name", default='1', type=str, help="index of model")
    parser.add_argument("--kernel", default='NTK', type=str, help="NTK or NFK")
    parser.add_argument('--k', default=128, type=int, help='dimension reduce')
    parser.add_argument('--start_k', default=0, type=int, help='start dimension reduce')
    parser.add_argument('--interval_k', default=5, type=int, help='interval dimension reduce')
    parser.add_argument('--sample_num', default=1000, type=int, help='sample number of fisher kernel')
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
    ################ 多卡
    parser.add_argument('--seed', default=42, type=int, help='randomize seed')
    parser.add_argument('--nproc_per_node', type=int, default=2, help='number of process in each node, equal to number of gpus')
    parser.add_argument('--nnode', type=int, default=1, help='computer number')
    parser.add_argument('--node_rank', type=int, default=0, help='computer rank')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of gpus.")
    parser.add_argument("--port", default='12345', type=str, help="master port")
    args = parser.parse_args()
    return args

def main(local_rank, his_omiga, save_dir, args): 
    # 计算global_rank和world_size
    global_rank = local_rank + args.node_rank * args.nproc_per_node
    world_size = args.nnode * args.nproc_per_node
    print('global_rank:', global_rank, 'world_size:', world_size)
    setup(global_rank=global_rank, world_size=world_size, port=args.port)
    # 设置seed
    # torch.manual_seed(args.seed)
    initialize(args, seed=args.seed)
    # 输出记录log
    file_name=os.path.basename(__file__).split(".")[0]
    # save_path = file_name + '/' + args.model + '_' + args.data + '_best_rho=' + str(args.rho) + '_labsmooth='+str(args.label_smoothing) + '/' 
    save_path = save_dir
    os.makedirs(save_path, exist_ok=True)
    sys.stdout = Logger(save_path + 'output.log')

    # 设置数据集
    if args.data == 'mnist':
        dataset = MNIST(args.batch_size, args.threads, args.num_gpus)
    else:
        dataset = IN_DATA(args.data, args.batch_size, args.threads, args.num_gpus, args.model)

    # 创建模型
    if args.model == 'vgg16bn':
        model = VGG16BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'vgg19bn':
        model = VGG19BN(10 if args.data == 'cifar10' else 100)
    elif args.model == 'resnet18':
        model = resnet18(10 if args.data == 'cifar10' else 100)
        if args.data == 'cifar100':
            checkpoint = torch.load('2024_04_15_OE_detection/resnet18_cifar100_lr=0.07.pt')
            model.load_state_dict(checkpoint, strict=False)
        elif args.data == 'cifar10':
            checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar10_best_rho=0.05_labsmooth=0.1/seed_42.pt')
            print('epoch:', checkpoint['epoch'])
            model.load_state_dict(checkpoint['model'], strict=False)
            model.set_nor(False) 
    elif args.model == 'wideresnet':
        from model.wide_res_net import WideResNet
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10 if args.data == 'cifar10' else 100)
        checkpoint = torch.load('./weights/baseline/SGD_wideresnet-28-10_'+args.data+'_labsmooth=0.1/seed_111.pt')
        model.load_state_dict(checkpoint['model'], strict=False)
    elif args.model == 'lenet':
        model = LeNet()
    elif args.model == 'resnet50':
        import torchvision
        from torchvision import models
        model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    elif args.model == 'wrt':
        from model.wrn import WideResNet
        model = WideResNet(40, 100, 2, dropRate=0.3)
        checkpoint = torch.load('./DAL/models/'+args.data+'_baseline_oe_v21.pt')
        model.load_state_dict(checkpoint, strict=False)
    elif args.model == 'densenet':
        from model.densenet import densenet121
        model = densenet121(num_classes=10 if args.data == 'cifar10' else 100)
        # checkpoint = torch.load('./weights/baseline/SAM_densenet_'+args.data+'_best_rho=0.05_labsmooth=0.1/seed_42.pt') # ID ACC: 91.15
        checkpoint = torch.load('./weights/baseline/SGD_densenet_'+args.data+'_labsmooth=0.1/seed_111.pt') # ID ACC: 91.31
        model.load_state_dict(checkpoint['model'], strict=False)
        check = '_densenet'
    elif args.model == 'mobilenet':
        import torchvision
        from torchvision import models
        model = models.mobilenet_v2()
        model.classifier = nn.Linear(1280, 10)
        checkpoint = torch.load('weights/baseline/' + 'SGD_' + args.model + '_' + args.data + '_labsmooth=0.0/seed_42.pt')
        model.load_state_dict(checkpoint['model'], strict=True)
    
    # ##### load model
    # if args.data == 'cifar10':
    #     # checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar10_best_rho=0.05_labsmooth=0.1/seed_42.pt')
    #     # print('epoch:', checkpoint['epoch'])
    #     # model.load_state_dict(checkpoint['model'], strict=False)
    #     # model.set_nor(False)
    #     checkpoint = torch.load('2024_04_15_OE_detection/' + args.model +'_'+args.data+'_lr=0.07.pt')
    #     model.load_state_dict(checkpoint, strict=False)
    #     model.set_nor(False)

    # elif args.data == 'mnist':
    #     checkpoint = torch.load('2023_03_06_compare_sgd/lenet_mnist_best_rho=0.05_labsmooth=0.0/seed_42.pt')
    #     print('epoch:', checkpoint['epoch'])
    #     model.load_state_dict(checkpoint['model'], strict=False)

    # elif args.data == 'cifar100':
    #     # checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar100_best_rho=0.05.pt')
    #     # print('epoch:', checkpoint['epoch'])
    #     # model.load_state_dict(checkpoint['model'], strict=False)
    #     # model.set_nor(False)

    #     # checkpoint = torch.load('weights/SGD_resnet18_cifar100_labsmooth=0.1/epoch199.pt')
    #     # print('epoch:', checkpoint['epoch']) 
    #     # model.load_state_dict(checkpoint['model'], strict=False)

    #     checkpoint = torch.load('2024_04_15_OE_detection/' + args.model +'_'+args.data+'_lr=0.07.pt')
    #     model.load_state_dict(checkpoint, strict=False)
    #     model.set_nor(False)
    
    # checkpoint = torch.load('2023_03_06_compare_sgd/resnet18_cifar10_best_rho=0.05_labsmooth=0.1/seed_42.pt')
    # checkpoint = torch.load('2024_04_15_OE_detection/' + args.model +'_'+args.data+'_lr=0.07.pt')
    # model.load_state_dict(checkpoint, strict=False)
    # model.set_nor(False)


    # 移动模型到local_rank对应的GPU上
    model = model.to(local_rank)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # 测试模型准确率   
    device = torch.device('cuda:'+ str(local_rank))
    if args.start_k == 0:
        test(dataset.test, model, device)
    # print(fsdfsd)
    #######
    model = Energy_Model(model)
    model.eval()
    
    ######################## obtain z_theta, I_2, p
    sample_num = args.sample_num
    # if args.data == 'cifar10':
    #     kernel_dataset, _ = torch.utils.data.random_split(dataset.train_set, [sample_num, len(dataset.train_set)-sample_num], generator=torch.Generator().manual_seed(0))
    #     # kernel_dataloader = DataLoader(kernel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads) # shuffle=False is important
    #     kernel_sampler = torch.utils.data.distributed.DistributedSampler(kernel_dataset, num_replicas=world_size, rank=global_rank)
    #     kernel_dataloader = torch.utils.data.DataLoader(kernel_dataset, batch_size=args.batch_size, num_workers=args.threads, sampler=kernel_sampler)
    # else:
    if True:
        kernel_dataset, _ = torch.utils.data.random_split(dataset.train_set, [sample_num, len(dataset.train_set)-sample_num], generator=torch.Generator().manual_seed(0))
        sample_num_per_gpu = math.floor(sample_num/world_size)
        if local_rank != world_size-1:
            start_sample_num = local_rank*sample_num_per_gpu
            sub_sample_num = sample_num_per_gpu
            end_sample_num = start_sample_num + sub_sample_num
        else:
            start_sample_num = local_rank*sample_num_per_gpu
            sub_sample_num = sample_num-(world_size-1)*sample_num_per_gpu
            end_sample_num = start_sample_num + sub_sample_num
        indices = [i for i in range(start_sample_num, end_sample_num)]
        # print('local_rank:', local_rank, indices)
        sub_kernel_dataset = torch.utils.data.Subset(kernel_dataset, indices)
        # print('dataset_len:', len(sub_kernel_dataset))
        # torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        kernel_dataloader = torch.utils.data.DataLoader(sub_kernel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
        # print('imagenet or cifar100')
        

    # for batch in kernel_dataloader:
    #     inputs, targets = batch
    #     targets = targets.to(local_rank)
    #     print(targets.shape, 'data targets:', targets[0:min(10,targets.shape[0])])
    #     break
    

    # save_dir = file_name + '/' + args.model  + '_' + args.data + '_' + args.kernel + '/sample_num=' + str(args.sample_num) + '_k=' + str(args.k) + '/'
    # save_dir = file_name + '/' + args.model  + '_' + args.data + '_' + args.kernel + '/sample_num=50001' + '_k=' + str(args.k) + '/'
    # save_dir = file_name + '/' + 'lenet'  + '_' + 'mnist' + '_' + args.kernel + '/sample_num=' + str(args.sample_num) + '_k=' + str(args.k) + '/'
    os.makedirs(save_dir, exist_ok=True)
    if local_rank == 0:
        print('save_dir: ', save_dir)
    

    if os.path.exists(save_dir + 'I_2.npy'):
        z_theta = torch.from_numpy(np.load(save_dir + 'z_theta.npy')).to(device)
        I_2 = torch.from_numpy(np.load(save_dir + 'I_2.npy')).to(device)
    else:
        z_theta, I_2 = pre_compute(model, kernel_dataloader, save_dir, device, local_rank)
        if local_rank == 0:
            # np.save(save_dir + 'z_theta.npy', z_theta.cpu().detach().numpy())
            np.save(save_dir + 'I_2.npy', I_2.cpu().detach().numpy())

    print('z_theta:', z_theta.shape, 'I_2:', I_2.shape)
    print(z_theta, I_2)
    # p, sigma, q = truncated_svd_dataloader(model, kernel_dataloader, k=args.k, iter=10, kernel=args.kernel, z_theta=z_theta, I_2=I_2, save_dir=save_dir)
    # print('sigma:', sigma)
    #################################### 记录inf-value的地方，生成mask
    # z_theta, I_2 = pre_compute_mask(model, kernel_dataloader, save_dir, device, local_rank)
    ####################################

    start_k = args.start_k
    interval_k = args.interval_k
    # sigma_3 = []
    # p_3 = []
    # q_3 = []
    # for i in np.arange(start_k, args.k, interval_k):
    #     if local_rank==0:
    #         print('k=', i)
    #     p_name = []
    #     if args.data == 'imagenet':
    #         for j in np.arange(0, start_k, 10):
    #             p_name.append(save_dir + str(j) + '-' + str(min(j+9,start_k-1))+ '_p.npy')

    #     for j in np.arange(start_k, i, interval_k):
    #         path = save_dir + str(j) + '-' + str(min(j+interval_k-1,args.k-1))+ '_p.npy'
    #         p_name.append(path)
    #     start = time.time()
    #     p, sigma, q = truncated_svd_3(model, kernel_dataloader, n=sample_num, omiga_name=p_name, k=interval_k, iter=2, kernel=args.kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=local_rank)
    #     end = time.time()
    #     if local_rank==0:
    #         print('time:', end-start)
    #     p_3.append(p)
    #     q_3.append(q)
    #     sigma_3.append(sigma)
    #     if local_rank == 0:
    #         print(sigma, p.shape, q.shape)
    #         np.save(save_dir + str(i) + '-' + str(min(i+interval_k-1,args.k-1))+ '_p.npy', p.cpu().detach().numpy())
    #     torch.distributed.barrier()

    if local_rank==0:
        print('k=', start_k)
    
    start = time.time()
    # p, sigma, q = truncated_svd_3(model, kernel_dataloader, n=sample_num, omiga_name=[], k=interval_k, iter=10, kernel=args.kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=local_rank, his_omiga=his_omiga)
    p, sigma, q = truncated_svd_4(model, kernel_dataloader, n=sample_num, omiga_name=[], k=interval_k, iter=10, kernel=args.kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size, rank=local_rank, his_omiga=his_omiga)
    end = time.time()
    if local_rank==0:
        print('time:', end-start)
    # p_3.append(p)
    # q_3.append(q)
    # sigma_3.append(sigma)
    if local_rank == 0:
        torch.set_printoptions(sci_mode=True)
        print(start_k, '-', start_k+interval_k-1, ', sigma:', sigma)
        print(p.shape, q.shape)
        np.save(save_dir + str(start_k) + '-' + str(start_k+interval_k-1) + '_p.npy', p.cpu().detach().numpy())
        np.save(save_dir + str(start_k) + '-' + str(start_k+interval_k-1) + '_sigma.npy', sigma.cpu().detach().numpy())
    # print(fdsfsf)
    # torch.distributed.barrier()

    # p_3 = torch.cat(p_3, 1)
    # q_3 = torch.cat(q_3, 1)
    # sigma_3 = torch.cat(sigma_3, 0)
    # sim_v = (p_3 @ torch.diag_embed(sigma_3) @ q_3.T)

    # ######## naive truncated svd 
    # v = cal_fisher_vector_dataloader(model, kernel_dataloader, kernel=args.kernel, z_theta=z_theta, I_2=I_2, device=device, world_size=world_size)
    # print('error:', torch.norm(sim_v.T-v.to(device))/torch.norm(v.to(device)))
    # v = v.cpu()
    # svd = TruncatedSVD(n_components=args.k, n_iter=10, random_state=42)
    # svd.fit(v)
    # print('singular_values_:', svd.singular_values_)

    return

if __name__ == "__main__":
    args = set_args()
    # mp.spawn(main, args=(args,), nprocs=args.nproc_per_node)
    mp.set_start_method("spawn")

    p_name = []
    start_k = args.start_k
    interval_k = args.interval_k
    file_name=os.path.basename(__file__).split(".")[0]
    # save_dir = file_name + '/' + args.model  + '_' + args.data + '_' + args.kernel + '/sample_num=' + str(args.sample_num) + '_k=' + str(args.k) + '/'
    # save_dir = file_name + '/2024_05_23/resnet18_cifar10_gpu4_bz500/'
    # save_dir = file_name + '/2024_05_23/resnet18_cifar100_OE_gpu4_bz500/'
    # save_dir = file_name + '/2024_05_23/densenet_cifar10_I2=1_gpu4_bz500/'
    # save_dir = file_name + '/2024_05_23/'+args.model+'_'+args.data+'_I2=1_gpu4_bz500/'
    # save_dir = file_name + '/2024_05_23/'+args.model+'_'+args.data+'_gpu4_bz500/'
    # save_dir = file_name + '/2024_05_23/'+args.model+'_'+args.data+'_reproduce_2/'
    save_dir = './2023_05_26_cal_p_fast/resnet18_cifar10_NFK/sample_num=49998_k=512/'
    os.makedirs(save_dir, exist_ok=True)

    if args.data == 'imagenet':
        ##### start_k=128
        p_name.append(save_dir + 'p.npy')
        for j in np.arange(128, start_k, interval_k):
            path = save_dir + str(j) + '-' + str(min(j+interval_k-1,start_k-1))+ '_p.npy'
            p_name.append(path)
    elif args.data == 'cifar10':
        # # p_name.append(save_dir + 'p.npy')
        # # for j in np.arange(0, start_k, interval_k):
        # #     path = save_dir + str(j) + '-' + str(min(j+interval_k-1,start_k-1))+ '_p.npy'
        # #     p_name.append(path)
        # for j in np.arange(0, 182, 2):
        #     path = save_dir + str(j) + '-' + str(min(j+2-1,start_k-1))+ '_p.npy'
        #     p_name.append(path)
        # for j in np.arange(182, 214, 32):
        #     path = save_dir + str(j) + '-' + str(min(j+32-1,start_k-1))+ '_p.npy'
        #     p_name.append(path)
        # for j in np.arange(214, 216, 2):
        #     path = save_dir + str(j) + '-' + str(min(j+2-1,start_k-1))+ '_p.npy'
        #     p_name.append(path)
        #     # print(path)
        # for j in np.arange(216, start_k, interval_k):
        #     path = save_dir + str(j) + '-' + str(min(j+interval_k-1,start_k-1))+ '_p.npy'
        #     p_name.append(path)
        #     # print(path)
        p_name.append(save_dir + '0-128.npy')
        p_name.append(save_dir + '128-256.npy')
        p_name.append(save_dir + '256-263.npy')
        # for j in np.arange(0, start_k, interval_k):
        for j in np.arange(264, start_k, interval_k):
            path = save_dir + str(j) + '-' + str(min(j+interval_k-1,start_k-1))+ '_p.npy'
            p_name.append(path)
    elif args.data == 'mnist':
        for j in np.arange(0, start_k, interval_k):
            path = save_dir + str(j) + '-' + str(min(j+interval_k-1,start_k-1))+ '_p.npy'
            p_name.append(path)
    elif args.data == 'cifar100':
        # for j in np.arange(0, start_k, interval_k):
        #     path = save_dir + str(j) + '-' + str(min(j+interval_k-1,start_k-1))+ '_p.npy'
        #     p_name.append(path)
        for j in np.arange(0, start_k, interval_k):
            path = save_dir + str(j) + '-' + str(min(j+interval_k-1,start_k-1))+ '_p.npy'
            p_name.append(path)

    if p_name!=[]:
        his_omiga = load_omiga(p_name)
        his_omiga = his_omiga.share_memory_()
        print('his_omiga:', his_omiga.shape)
    else:
        his_omiga=None
        print('his_omiga=None')

    processes = []
    try:
        for rank in range(args.nproc_per_node):
            p = mp.Process(target=main, args=(rank, his_omiga, save_dir, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print('catch keyboardinterupterror')
        pid=os.getpid()
        os.popen('taskkill.exe /f /pid:%d'%pid) #在unix下无需此命令，但在windows下需要此命令强制退出当前进程
    except Exception as e:
        print(e)
    else:
        print('quit normally')


    


    
