# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
import os.path as osp
import sys

import cv2
from loguru import logger
from yacs.config import CfgNode

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import random
import os
import numpy as np

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.optim import builder as optim_builder
from videoanalyst.utils import Timer, complete_path_wt_root_in_cfg, ensure_dir, dist_utils
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.engine.monitor.monitor_impl.tensorboard_logger import TensorboardLogger

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='path to experiment configuration')
    parser.add_argument(
        '-r',
        '--resume',
        default="",
        help=r"completed epoch's number, latest or one model path")
    parser.add_argument(
        '-v',
        '--validation',
        default="15",
        help=r"Epoch's number to start to evaluate the model on the validation set")
    parser.add_argument('-ad',
                        '--auto-dist',
                        default=True,
                        help="whether use auto distributed training")
    parser.add_argument('-du',
                        '--dist_url',
                        default="tcp://127.0.0.1:12345",
                        help="the url port of master machine")

    return parser

def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def steup(rank: int, world_size: int, dist_url: str):
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        init_method=dist_url
    )

def cleanup():
    dist.destroy_process_group()

def run_dist_training(rank_id: int, world_size: int, task: str,
                      task_cfg: CfgNode, parsed_args, model, dist_url):

    devs = ["cuda:{}".format(rank_id)]

    steup(rank_id, world_size, dist_url)
    dist_utils.synchronize()

    model.set_device(devs[0])

    optimizer = optim_builder.build(task, task_cfg.optim, model)

    with Timer(name="Dataloader building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data, seed=rank_id)

    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer, dataloader)

    trainer.set_device(devs)

    trainer.resume(parsed_args.resume)

    logger.info("Start Training")
    while not trainer.is_completed():
        trainer.train()
        if rank_id == 0:
            trainer.save_snapshot()
        dist_utils.synchronize()

    if rank_id == 0:
        trainer.save_snapshot()

    cleanup()


if __name__ == '__main__':
    set_seed(1000000007)
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()
    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    # root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg.train)
    task_cfg.freeze()
    # log config
    log_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")
    ensure_dir(log_dir)
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, level="INFO"),
            dict(sink=osp.join(log_dir, "train_log.txt"),
                 enqueue=True,
                 serialize=True,
                 diagnose=True,
                 backtrace=True,
                 level="INFO")
        ],
        extra={"common_to_all": "default"},
    )
    # backup config
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)
    logger.info(
        "Merged with root_cfg imported from videoanalyst.config.config.cfg")
    cfg_bak_file = osp.join(log_dir, "%s_bak.yaml" % task_cfg.exp_name)
    with open(cfg_bak_file, "w") as f:
        f.write(task_cfg.dump())
    logger.info("Task configuration backed up at %s" % cfg_bak_file)
    # build dummy dataloader
    with Timer(name="Dummy dataloder building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data)
    del dataloader
    logger.info("Dummy dataloader destroyed.")
    # device config
    world_size = task_cfg.num_processes
    assert torch.cuda.is_available(), "please check your devices"
    assert torch.cuda.device_count(
    ) >= world_size, "cuda device {} is less than {}".format(
        torch.cuda.device_count(), world_size)

    # build model
    model = model_builder.build(task, task_cfg.model)
    # get dist url
    if parsed_args.auto_dist:
        port = _find_free_port()
        dist_url = "tcp://127.0.0.1:{}".format(port)
    else:
        dist_url = parsed_args.dist_url
    # prepare to spawn
    torch.multiprocessing.set_start_method('spawn', force=True)
    # spawn trainer prcess
    mp.spawn(run_dist_training,
             args=(world_size, task, task_cfg, parsed_args, model, dist_url),
             nprocs=world_size,
             join=True)
    logger.info("Distrubuted training completed.")