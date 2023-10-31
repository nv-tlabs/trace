"""Factory methods for creating models"""
from tbsim.configs.base import ExperimentConfig

from tbsim.algos.algos import (
    DiffuserTrafficModel,
)

def algo_factory(config: ExperimentConfig, modality_shapes: dict):
    """
    A factory for creating training algos

    Args:
        config (ExperimentConfig): an ExperimentConfig object,
        modality_shapes (dict): a dictionary that maps observation modality names to shapes

    Returns:
        algo: pl.LightningModule
    """
    algo_config = config.algo
    algo_name = algo_config.name

    if algo_name == "diffuser":
        algo = DiffuserTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
