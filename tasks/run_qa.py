import torch.distributed as dist
import deepspeed
import argparse
import os
from mmcv import Config
from models import TCP_QA_Multichoice, TCP_QA_Classification
from datasets.dataloader import build_dataloader
from optimization.lr_scheduler import build_scheduler
from optimization.optimizer import build_optimizer_parameters

from utils.logger import LOGGER, add_log_to_file
from utils.dist import master_process
from utils.misc import mkdirp, set_random_seed
from utils.load import load_model_weights_with_mismatch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./src/configs/pretrain.yaml')
    parser.add_argument('--blob_mount_dir', default="/blob_mount")
    parser.add_argument('--deepspeed_sparse_attention',action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--fp16', action='store_true', help='enable fp16')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--distributed',action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    set_random_seed(args.seed)

    config = Config.fromfile(args.config)

    LOGGER.info(config)
    LOGGER.info(args)


    if not master_process(args):
        LOGGER.disabled = True
    if master_process(args):
        mkdirp(os.path.join(args.blob_mount_dir, config.TRAINING.save_dir,"log"))
        add_log_to_file(os.path.join(args.blob_mount_dir, config.TRAINING.save_dir,"log/log.txt"))

    if config.qa_type == 'classification':
        model =  LFVILA_QA_Classification(args, config)
    else:
        model =  LFVILA_QA_Multichoice(args, config)

    if config.WEIGHTS.model_weight != '':
        LOGGER.info(f"Loading model weights from {config.WEIGHTS.model_weight}")
        load_model_weights_with_mismatch(model, os.path.join(args.blob_mount_dir, config.WEIGHTS.model_weight))

    elif config.WEIGHTS.stage1_model_weight != '':

        LOGGER.info(f"Loading bert part2 weights from {config.WEIGHTS.bert_weight}")
        load_model_weights_with_mismatch(model.text_encoder, os.path.join(args.blob_mount_dir, config.WEIGHTS.bert_weight),load_bert=True)

        LOGGER.info(f"Loading stage1 model weights (video encoder, bert part1) from {config.WEIGHTS.stage1_model_weight}")
        load_model_weights_with_mismatch(model, os.path.join(args.blob_mount_dir, config.WEIGHTS.stage1_model_weight))

    else:
        if config.WEIGHTS.swin_weight != '':
            LOGGER.info(f"Loading video encoder weights from {config.WEIGHTS.swin_weight}")
            
            load_model_weights_with_mismatch(model.video_encoder, 
                                            os.path.join(args.blob_mount_dir, config.WEIGHTS.swin_weight),
                                            load_swin=True,
                                            pretrained2d=config.WEIGHTS.pretrained_2d)

        if config.WEIGHTS.bert_weight != '':
            LOGGER.info(f"Loading bert weights from {config.WEIGHTS.bert_weight}")
            load_model_weights_with_mismatch(model.text_encoder, os.path.join(args.blob_mount_dir, config.WEIGHTS.bert_weight),load_bert=True)
            LOGGER.info(f"Init sentence position embedding")
            model._init_sent_embedding()
    

    parameter_group = build_optimizer_parameters(config, model)

    # init deepspeed
    
    if args.distributed:
        model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                        model=model,
                                                        model_parameters=parameter_group,
                                                        config=config.deepspeed_config)
    

    LOGGER.info(f'Training with {dist.get_world_size()} gpus')
    

    dataset_trains, dataset_vals, dataloader_trains, dataloader_vals = build_dataloader(args, config)

    dataloader_train = dataloader_trains['QADataset-train']

    steps_per_epoch = len(dataloader_train)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)

    args.fp16 = model_engine.fp16_enabled()
    if args.fp16:
        LOGGER.info('Enable fp16 Training')

    if config.qa_type == 'classification':
        trainer = Trainer_QA_Classification(args, config, model_engine, optimizer, scheduler, dataloader_train, dataloader_vals['QADataset-val'])
    else:
        trainer = Trainer_QA_Multichoice(args, config, model_engine, optimizer, scheduler, dataloader_train, dataloader_vals['QADataset-val'])

    LOGGER.info('start first evaluate')

    trainer.evaluate(dataloader_vals['QADataset-val'], stage=2)


    trainer.train(args.resume)

if __name__ == '__main__':
    deepspeed.init_distributed()
    main()


