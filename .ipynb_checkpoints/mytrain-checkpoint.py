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

from nets.clip import CLIP
from utils.callbacks import EvalCallback, LossHistory
from utils.mydataloader import MyDataset, dataset_collate
from utils.utils import (get_configs, get_lr_scheduler, set_optimizer_lr,
                         show_config)
from utils.utils_fit import fit_one_epoch
from torch import nn

if __name__ == "__main__":
    Cuda = True
    distributed = False
    fp16 = False
    model_path = "model_data/ViT-B-32-OpenAI.pth"
    phi = "openai/VIT-B-32"
    batch_size = 32
    Init_Epoch = 0
    Epoch = 100
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2
    lr_decay_type = 'cos'
    save_period = 1
    save_dir = 'logs'
    gene_number = 2000

    eval_flag = True
    eval_period = 1

    num_workers = 4

    datasets_train_json_path = "datasets/train_0.7.json"
    datasets_val_json_path = "datasets/val_0.7.json"
    datasets_random = True

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    config = get_configs(phi)
    model = CLIP(**config)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict, strict=False)
        model.gene.load_state_dict(torch.load('./panglao_pretrain.pth', map_location=device), strict=False)
        for param in model.gene.parameters():
            param.requires_grad = False
        for param in model.gene.norm.parameters():
            param.requires_grad = True
        for param in model.gene.performer.net.layers[-2].parameters():
            param.requires_grad = True
        for param in model.gene.output.parameters():
            param.requires_grad = True
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, None)
    else:
        loss_history = None

    if fp16:
        # ------------------------------------------------------------------#
        #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
        #   因此torch1.2这里显示"could not be resolve"
        # ------------------------------------------------------------------#
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    train_lines = json.load(open(datasets_train_json_path, mode='r', encoding='utf-8'))
    val_lines = json.load(open(datasets_val_json_path, mode='r', encoding='utf-8'))

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            model_path=model_path, phi=phi, \
            Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    if True:
        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-4
        lr_limit_min = 3e-5
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # ---------------------------------------#
        #   构建数据集加载器
        # ---------------------------------------#
        train_dataset = MyDataset([config['input_resolution'], config['input_resolution']], train_lines,
                                  gene_number=gene_number, random=datasets_random)
        val_dataset = MyDataset([config['input_resolution'], config['input_resolution']], val_lines,
                                gene_number=gene_number,
                                random=False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate, sampler=train_sampler, prefetch_factor=5) # prefetch_factor=5
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False, collate_fn=dataset_collate, sampler=val_sampler, prefetch_factor=5)

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_dataset = MyDataset([config['input_resolution'], config['input_resolution']], val_lines,
                                     gene_number=gene_number, random=False)
            gen_eval = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False, collate_fn=dataset_collate, sampler=None)
            eval_callback = EvalCallback(model, gen_eval, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, Epoch, Cuda, \
                          fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
