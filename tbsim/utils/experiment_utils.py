import json
import os
from typing import List
from glob import glob
import subprocess
import shutil
from pathlib import Path

import tbsim
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.configs.config import Dict
from tbsim.configs.eval_config import EvaluationConfig

def read_configs(config_dir):
    configs = []
    config_fns = []
    for cfn in glob(config_dir + "/*.json"):
        print(cfn)
        config_fns.append(cfn)
        ext_cfg = json.load(open(cfn, "r"))
        c = get_registered_experiment_config(ext_cfg["registered_name"])
        c.update(**ext_cfg)
        configs.append(c)
    return configs, config_fns


def read_evaluation_configs(config_dir, eval_class=EvaluationConfig):
    configs = []
    config_fns = []
    for cfn in glob(config_dir + "/*.json"):
        print(cfn)
        config_fns.append(cfn)
        c = eval_class()
        ext_cfg = json.load(open(cfn, "r"))
        c.update(**ext_cfg)
        configs.append(c)
    return configs, config_fns

def get_checkpoint(
    ckpt_key, ckpt_dir=None, ckpt_root_dir="checkpoints/", download_tmp_dir="/tmp"
):
    """
    Get checkpoint and config path from local dir.

    If a @ckpt_dir is specified, the function will look for the directory locally and return the ckpt that contains
    @ckpt_key, as well as its config.json.

    Args:
        ckpt_key (str): a string that uniquely identifies a checkpoint file with a directory, e.g., `iter50000.ckpt`
        ckpt_dir (str): (Optional) a local directory that contains the specified checkpoint
        ckpt_root_dir (str): (Optional) a directory that the function will look for checkpoints
        download_tmp_dir (str): a temporary storage for the checkpoint.

    Returns:
        ckpt_path (str): path to a checkpoint file
        cfg_path (str): path to a config.json file
    """
    def ckpt_path_func(paths): return [p for p in paths if str(ckpt_key) in p]
    local_dir = ckpt_dir
    assert ckpt_dir is not None

    ckpt_paths = glob(local_dir + "/**/*.ckpt", recursive=True)
    if len(ckpt_path_func(ckpt_paths)) == 0:
        raise FileNotFoundError("Cannot find checkpoint in {} with key {}".format(local_dir, ckpt_key))
    else:
        ckpt_dir = local_dir

    ckpt_paths = ckpt_path_func(glob(ckpt_dir + "/**/*.ckpt", recursive=True))
    assert len(ckpt_paths) > 0, "Could not find a checkpoint that has key {}".format(
        ckpt_key
    )
    assert len(ckpt_paths) == 1, "More than one checkpoint found {}".format(ckpt_paths)
    cfg_path = glob(ckpt_dir + "/**/config.json", recursive=True)
    if len(cfg_path) == 0:
        # look in parent
        cfg_path = glob(ckpt_dir + "/../config.json")
    cfg_path = cfg_path[0]
    print("Checkpoint path: {}".format(ckpt_paths[0]))
    print("Config path: {}".format(cfg_path))
    return ckpt_paths[0], cfg_path

