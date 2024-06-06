import os
import datetime
import sys
sys.path.append('/home/pylib/lib/python3.7/site-packages') 
import logging
import torch
import torch.distributed as dist
import deepspeed
import argparse

from mmcv import Config
from models import TCP_Pretrain

from tools import Trainer_Pretrain
from datasets.dataloader import build_dataloader
from optimization.lr_scheduler import build_scheduler
from optimization.optimizer import build_optimizer_parameters

from utils.logger import LOGGER, add_log_to_file
from utils.dist import master_process
from utils.misc import mkdirp, set_random_seed
from utils.load import load_model_weights_with_mismatch


# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='TCP parameters')
    parser.add_argument('--data', dest='dataset',default='CUB', type=str)
    parser.add_argument('--save', dest='resume',default=None,type=str)

    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch', default=360, type=int)
    parser.add_argument('--batch_size', dest='batch_size',default=16, type=int)

    parser.add_argument('--weight_decay', dest='weight_decay',
                        default=0.05, type=float)
    parser.add_argument('--optim', dest='optimizer',default='AdamW', type=str)

    parser.add_argument('--lr', dest='learn_rate',default=0.00001, type=float)
    ##
    parser.add_argument('--video_weight', dest='video_weight',default='',type=str)
    parser.add_argument('--text_weight', dest='text_weight',default='',type=str)

    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--nw', dest='num_workers', default=32, type=int)

    parser.add_argument('--use_amp', dest='use_amp', type=str, default='')
    parser.add_argument('--warmup_epch', type=int, default=1)
    parser.add_argument('--saved_dir', dest='saved_dir', type=str, default=os.getcwd())

    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--frequent', dest='frequent', type=int, default=100)
    parser.add_argument('--total_step', dest='total_step', type=int, default=200000)


    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)

    os.makedirs(save_dir, exist_ok=True)
    Config.automix_cfg['save_dir'] = os.path.join(save_dir, Config.automix_cfg['save_dir'])
    os.makedirs(Config.automix_cfg['save_dir'], exist_ok=True)

    args.world_size = world_size = torch.cuda.device_count()
    args.total_batch_size = args.train_batch * world_size

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    set_random_seed(args.seed)
    config = Config.fromfile(args.config)

    logging.info(config)
    logging.info(args)

    if not master_process(args):
        LOGGER.disabled = True
    if master_process(args):
        mkdirp(os.path.join(args.blob_mount_dir, config.TRAINING.save_dir,"log"))
        add_log_to_file(os.path.join(args.blob_mount_dir, config.TRAINING.save_dir,"log/log.txt"))

    model = TCP_Pretrain(args, config)

    if args.video_weight != '':
        logging.info(f"Loading video weights from {args.video_weight}")
        load_model_weights_with_mismatch(model.video_encoder, 
                                        os.path.join(args.blob_mount_dir, config.WEIGHTS.swin_weight),
                                        load_swin=True,
                                        pretrained2d=config.WEIGHTS.pretrained_2d)
    if args.text_weight != '':
        logging.info(f"Loading BERT weights from {config.WEIGHTS.bert_weight}")
        load_model_weights_with_mismatch(model.text_encoder, os.path.join(args.blob_mount_dir, config.WEIGHTS.bert_weight),load_bert=True)
        logging.info(f"Init sentence position embedding")
        model._init_sent_embedding()


    parameter_group = build_optimizer_parameters(config, model)

    # init deepspeed
    if args.distributed:
        model_engine, optimizer, _, _ = deepspeed.initialize(args = args,
                                                            model=model,
                                                            model_parameters=parameter_group,
                                                            config=config.deepspeed_config
                                                        )
        print(dist.get_rank())
    

    logging.info(f'Training with {dist.get_world_size()} gpus')
    

    dataset_trains, dataset_vals, dataloader_trains, dataloader_vals = build_dataloader(args, config)

    dataloader_train = dataloader_trains['PreTrainDataset-train']
    steps_per_epoch = len(dataloader_train)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)

    args.fp16 = model_engine.fp16_enabled()
    if args.fp16:
        logging.info('Enable fp16 Training')

    trainer = Trainer_Pretrain(args, config, model_engine, optimizer, scheduler, dataloader_train, dataloader_vals['RetrievalDataset-val'],dataloader_vals['PreTrainDataset-val'])

    #train
    trainer.train(args.resume)

if __name__ == '__main__':
    deepspeed.init_distributed()
    main()
