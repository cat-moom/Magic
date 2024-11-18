import datetime
import json
import os

import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from models.stage_two import Mymodel_two
from utils.callbacks import EvalCallback, LossHistory
from utils.decoder_dataset import MyDataset, dataset_collate
from utils.utils import (get_configs, get_lr_scheduler, set_optimizer_lr,
                         show_config)
from utils.utils_train_two import fit_one_epoch


if __name__ == "__main__":
    Cuda                = True
    distributed         = False
    fp16                = False
    batch_size      = 32
    Init_Epoch      = 0
    Epoch           = 100
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adamw"
    momentum            = 0.9
    weight_decay        = 1e-2
    lr_decay_type       = 'cos'
    save_period         = 1

    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 1
    num_workers         = 4
    datasets_path               = "datasets/"
    datasets_train_json_path    = "datasets/train_C_0.7.json"
    datasets_val_json_path      = "datasets/val_C_0.7.json"
    datasets_random             = True
    

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank      = 0
    rank            = 0

    model   = Mymodel_two()
    
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, None)
    else:
        loss_history = None

    scaler = None

    model_train     = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
        
    train_lines = json.load(open(datasets_train_json_path, mode = 'r', encoding = 'utf-8'))
    val_lines   = json.load(open(datasets_val_json_path, mode = 'r', encoding = 'utf-8'))
    
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        
    if True:
        #   判断当前batch_size，自适应调整学习率
        nbs             = 64
        lr_limit_max    = 1e-4
        lr_limit_min    = 3e-5
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #   根据optimizer_type选择优化器
        optimizer = {
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
        }[optimizer_type]

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        #   判断每一个世代的长度
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        #   构建数据集加载器
        train_dataset   = MyDataset([224, 224], train_lines, datasets_path, random = datasets_random)
        val_dataset     = MyDataset([224, 224], val_lines, datasets_path, random = False)
        

        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler, prefetch_factor=5)
        gen_val         = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=False, collate_fn=dataset_collate, sampler=val_sampler, prefetch_factor=5)

        #   记录eval的map曲线
        if local_rank == 0:
            eval_dataset    = MyDataset([224, 224], val_lines, datasets_path, random = False)
            gen_eval        = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=False, collate_fn=dataset_collate, sampler=None)
            eval_callback   = EvalCallback(model, gen_eval, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, loss_history,  optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, \
                          fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
