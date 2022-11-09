# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.rnn.rnn_base import TASK_RNN
from videoanalyst.utils import merge_cfg_into_hps


def build(task: str, cfg: CfgNode):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task in TASK_RNN:
        rnn_modules = TASK_RNN[task]
    else:
        logger.error("no task model for task {}".format(task))
        exit(-1)

    name = cfg.name
    rnn_module = rnn_modules[name]()
    hps = rnn_module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    rnn_module.set_hps(hps)
    rnn_module.update_params()

    return rnn_module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}

    for cfg_name, module in TASK_RNN.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            task_model = module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
