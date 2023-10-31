"""A global registry for looking up named experiment configs"""
from tbsim.configs.base import ExperimentConfig

from tbsim.configs.trajdata_eupeds_config import (
    EupedsTrainConfig,
    EupedsEnvConfig
)
from tbsim.configs.trajdata_nusc_config import (
    NuscTrajdataTrainConfig,
    NuscTrajdataEnvConfig
)
from tbsim.configs.trajdata_nusc_ped_config import (
    NuscTrajdataPedTrainConfig,
    NuscTrajdataPedEnvConfig
)

from tbsim.configs.orca_config import (
    OrcaTrainConfig,
    OrcaEnvConfig
)

from tbsim.configs.trajdata_mixed_ped_config import (
    MixedPedTrainConfig,
    MixedPedEnvConfig
)

from tbsim.configs.algo_config import (
    DiffuserConfig,
)


EXP_CONFIG_REGISTRY = dict()

EXP_CONFIG_REGISTRY["trajdata_nusc_diff"] = ExperimentConfig(
    train_config=NuscTrajdataTrainConfig(),
    env_config=NuscTrajdataEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_nusc_diff"
)

EXP_CONFIG_REGISTRY["orca_diff"] = ExperimentConfig(
    train_config=OrcaTrainConfig(),
    env_config=OrcaEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="orca_diff"
)

EXP_CONFIG_REGISTRY["nusc_ped_diff"] = ExperimentConfig(
    train_config=NuscTrajdataPedTrainConfig(),
    env_config=NuscTrajdataPedEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="nusc_ped_diff"
)

EXP_CONFIG_REGISTRY["eupeds_diff"] = ExperimentConfig(
    train_config=EupedsTrainConfig(),
    env_config=EupedsEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="eupeds_diff"
)

EXP_CONFIG_REGISTRY["mixed_ped_diff"] = ExperimentConfig(
    train_config=MixedPedTrainConfig(),
    env_config=MixedPedEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="mixed_ped_diff"
)

def get_registered_experiment_config(registered_name):
    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()