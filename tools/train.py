import sys
sys.path.append('.')
import argparse
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from collections import Counter

import random
import numpy as np
from network import NETWORK_REGISTRY
from mmengine import Config,DictAction
from core.trainer import Trainer
from dataset import DATALOADER_REGISTRY
from utils.logging import create_checkpoint, setup_logger
from utils.utils import OPTIMIZER_REGISTRY, save_checkpoint, SCHEDULER_REGISTRY,farthest_point_sample,index_points

def set_random_seed(seed, deterministic=False): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    if deterministic: 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')
    # distributed training
    parser.add_argument('--gpus',
                        help='gpu ids for ddp training',
                        type=str)
    parser.add_argument('--port',
                        default='23459',
                        type=str,
                        help='port used to set up distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1',
                        type=str,
                        help='url used to set up distributed training')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        cfg.MODEL.vis=cfg.VIS
    assert cfg.train
    print(cfg.pretty_text)

    final_output_dir = create_checkpoint(cfg, 'train')

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    dist_url = args.dist_url + ':{}'.format(args.port)
    # save config file
    if not cfg.VIS:
        print('save cfg and source')
        src_folder = os.path.join(final_output_dir, 'src')
        if os.path.exists(os.path.join(src_folder, 'network')):
            shutil.rmtree(os.path.join(src_folder, 'network'))
        shutil.copytree('network', os.path.join(src_folder, 'network'))
        if os.path.exists(os.path.join(src_folder, 'tools')):
            shutil.rmtree(os.path.join(src_folder, 'tools'))
        shutil.copytree('tools', os.path.join(src_folder, 'tools'))
        if os.path.exists(os.path.join(src_folder, 'cfg.py')):
            os.remove(os.path.join(src_folder, 'cfg.py'))
        cfg.dump(os.path.join(src_folder, 'cfg.py'))

    ngpus_per_node = torch.cuda.device_count()

    set_random_seed(123)

    if cfg.DDP:
        world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(world_size, dist_url, final_output_dir, cfg))
    else:
        main_worker(0, 1, dist_url, final_output_dir, cfg)

def main_worker(rank, world_size, dist_url, final_output_dir, cfg):
    set_random_seed(42)
    if rank==0:
        logger, _ = setup_logger(final_output_dir, rank, 'train',cfg.VIS)
    else:
        logger=None

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    print("Use GPU: {} for training".format(rank))
    if cfg.DDP:
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.format(dist_url, world_size, rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=dist_url,
            world_size=world_size,
            rank=rank
        )

    # Data loading code
    train_loader = DATALOADER_REGISTRY.build(cfg)

    model = NETWORK_REGISTRY.build(cfg.MODEL)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank==0:
        logger.info("Total params: {num:.3f}M".format(num=count_parameters(model)/1e6))
    if cfg.DDP:
        print(rank)
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],find_unused_parameters=cfg.find_unused_parameters)
    else:
        model = torch.nn.DataParallel(model).cuda()


    
    best_perf = -1
    last_epoch = -1
    optimizer = OPTIMIZER_REGISTRY.build(cfg=cfg.OPTIMIZER, parameters=model.parameters())
    lr_scheduler = SCHEDULER_REGISTRY.build(cfg=cfg.SCHEDULER, optimizer=optimizer)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    if cfg.AUTO_RESUME:
        if cfg.RESUME_FILE != '':
            checkpoint_file = os.path.join(
            final_output_dir, 'model', cfg.RESUME_FILE)
        else:
            checkpoint_file = os.path.join(
            final_output_dir, 'model', 'checkpoint.pth.tar')
        print(checkpoint_file)
        if os.path.exists(checkpoint_file):
            if rank==0:
                logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            if not cfg.ONLY_MODEL:
                begin_epoch = checkpoint['epoch']
                best_perf = checkpoint['perf']
                last_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                if cfg.CHANGE_SCHEDULE:
                    milestones=cfg.SCHEDULER.milestones
                    new_counter={stones:1 for stones in milestones}
                    new_counter=Counter(new_counter)
                    checkpoint['scheduler']['milestones']=new_counter
                    checkpoint['scheduler']['gamma']=cfg.SCHEDULER.gamma
                if 'scheduler' in checkpoint.keys():
                    lr_scheduler.load_state_dict(checkpoint['scheduler'])

                # if cfg.CHANGE_SCHEDULE:
                #     lr_scheduler.step()
                #     print('lr',optimizer.state_dict()['param_groups'][0]['lr'])
            if rank==0:
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))


    is_iter=cfg.is_iter
    if is_iter:
        trainer = Trainer(cfg, model, rank, final_output_dir,logger=logger,lr_scheduler=lr_scheduler)
    else:
        trainer = Trainer(cfg, model, rank, final_output_dir,logger=logger)
    
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        train_loader.dataset.set_epoch(epoch)
        if cfg.DDP:
            train_loader.sampler.set_epoch(epoch)
        

        trainer.train(epoch, train_loader, optimizer)

        if not is_iter:
            lr_scheduler.step()

        perf_indicator = epoch
        if perf_indicator >= best_perf:
            
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.DDP or (cfg.DDP and rank == 0 and epoch%(cfg.TRAIN.SAVE_EPOCH_STEP)==0) and not cfg.VIS:
            file_name='checkpoint_epoch_'+str(epoch)+'.tar.pth'
            if rank==0:
                logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.type,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'scheduler':lr_scheduler.state_dict()
            }, best_model,final_output_dir, filename=file_name)
        if best_model:
            torch.save(
                model.module.state_dict(),
                os.path.join(final_output_dir, 'model_best.pth.tar')
            )

    final_model_state_file = os.path.join(
        final_output_dir, 'model', 'final_state{}.pth.tar'.format(rank)
    )
    if rank==0:
        logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

if __name__ == '__main__':
    
    main()