import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


from tensorboardX import SummaryWriter

from src.utils.logger import LOGGER
from src.utils.dist import concat_all_gather
from src.utils.dist import master_process
from src.utils.metrics import compute_rt_metrics

from src.utils.misc import AverageMeter


class Trainer_Pretrain():
    def __init__(self, args, config, model, optimizer, scheduler,
                    dataloader_train, task_dataloader_val=None, pre_dataloader_val=None):
        self.config = config
        self.local_rank = model.local_rank
        self.global_step = 0
        self.start_epoch = 0
        self.total_epochs = config.TRAINING.EPOCHS
        self.dataloader_train = dataloader_train
        self.task_dataloader_val = task_dataloader_val
        self.pre_dataloader_val = pre_dataloader_val
        
        self.args = args

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer

        if master_process(self.args) and config.log_tb:
            self.summary_writer = SummaryWriter(log_dir=os.path.join(args.blob_mount_dir,config.TRAINING.save_dir,'tb_log'))

    def _checkpoint(self,PATH, ckpt_id, epoch, global_step):

        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        checkpoint_state_dict = {
            'epoch': epoch,
            'global_step': global_step,
        }
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'checkpoint')
        
        save_trial = 0
        while save_trial < 10:
            try:
                LOGGER.info(f"checkpointing trial NO. {save_trial}")
                success = self.model.save_checkpoint(save_dir, ckpt_id, checkpoint_state_dict)
                status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(save_dir, ckpt_id)
                if success:
                    LOGGER.info(f"Success {status_msg}")
                    break
            except Exception as e:
                save_trial += 1
                LOGGER.warning(f"Failure {status_msg}")
        dist.barrier()


    def _save_model(self,PATH, epoch, step):
        
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'saved_model', 'epoch_{0:03d}_step_{1:05d}'.format(epoch, step))
        save_trial = 0
        while save_trial < 10:
            try:
                sucess = self.model.save_fp16_model(save_dir)
                break
            except Exception as e:
                save_trial += 1
                LOGGER.warning(f"Failure save model")

    def _resume(self,PATH, tag=None):
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'checkpoint')
        LOGGER.info(f"resume from {save_dir}")
        _, checkpoint_state_dict = self.model.load_checkpoint(save_dir)
        self.start_epoch = checkpoint_state_dict['epoch']
        self.global_step = checkpoint_state_dict['global_step']
        del checkpoint_state_dict
  
    def train(self, resume):
        self.model.train()
        if resume:
            self._resume(self.config.TRAINING.save_dir)
            LOGGER.info(f'resume from {self.start_epoch}, global step {self.global_step}')

        LOGGER.info(f'begin training from {self.start_epoch}')
        for epoch in range(self.start_epoch, self.total_epochs):

            if self.args.distributed:
                self.dataloader_train.sampler.set_epoch(epoch)

            for step, batch in enumerate(self.dataloader_train):

                video_frames = batch['video_frames'].to(self.local_rank)
                text_ids = batch['text_ids'].to(self.local_rank)
                attention_mask = batch['attention_mask'].to(self.local_rank)
  
                if self.config.stage == 2 and self.config.TRAINING.use_mlm:
                    mlm_labels = batch['mlm_labels'].to(self.local_rank)
                else:
                    mlm_labels=None


                if self.args.fp16:
                    video_frames = video_frames.half()
                    attention_mask = attention_mask.half()

                total_loss = self.model(video_frames,  
                                    text_ids, 
                                    attention_mask,
                                    mlm_labels = mlm_labels,
                                    stage=self.config.stage)


                self.model.backward(total_loss)
                self.model.step()

                self.global_step += 1
                self.scheduler.step_update(self.global_step)
                lr = self.scheduler._get_lr(self.global_step)[0]

                if self.global_step % self.config.TRAINING.eval_step == 0:
                    if self.config.stage==1:
                        self.evaluate(self.task_dataloader_val, stage=1)
                        self.evaluate(self.pre_dataloader_val, stage=1, pretrain_val=True)

                    if self.config.stage==2:
                        self.evaluate(self.pre_dataloader_val, stage=2, pretrain_val=True)

                if self.global_step % self.config.TRAINING.checkpoint_step == 0:
                    self._checkpoint(self.config.TRAINING.save_dir, self.global_step, epoch, self.global_step)

                if self.global_step % self.config.TRAINING.save_step == 0:
                    self._save_model(self.config.TRAINING.save_dir, epoch, step)
                
                if self.global_step > self.config.TRAINING.BREAK_STEP:
                    LOGGER.info(f"Job finished")
                    break
                    
            self.start_epoch = epoch
            LOGGER.info(epoch)
            if self.global_step > self.config.TRAINING.BREAK_STEP:
                LOGGER.info(f"Job finished")
                break

