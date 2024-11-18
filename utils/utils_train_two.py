import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .callbacks import de_parallel
from .utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    val_total_loss  = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train = model_train.cuda(0)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, genes = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                genes = genes.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            output_decorder, loss= model_train(images, genes)


            loss.backward()
            optimizer.step()

        else:
            from torch.cuda.amp import autocast
            with autocast():
                output_decorder, loss = model_train(images, genes)
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()


        if local_rank == 0:
            pbar.set_postfix(**{'total_loss'            : total_loss / (iteration + 1),
                                'lr'                    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, genes = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                genes = genes.cuda(local_rank)

            optimizer.zero_grad()

            output_decorder, loss = model_train(images, genes)

            val_total_loss += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'              : val_total_loss / (iteration + 1),
                                'lr'                    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch, total_loss / epoch_step, val_total_loss / epoch_step_val)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_total_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_total_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
