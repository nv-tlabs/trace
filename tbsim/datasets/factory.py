"""DataModule / Dataset factory"""
from tbsim.utils.config_utils import translate_pass_trajdata_cfg
from tbsim.datasets.trajdata_datamodules import PassUnifiedDataModule

def datamodule_factory(cls_name: str, config):
    """
    A factory for creating pl.DataModule.

    Args:
        cls_name (str): name of the datamodule class
        config (Config): an Experiment config object
        **kwargs: any other kwargs needed by the datamodule

    Returns:
        A DataModule
    """
    if cls_name.startswith("PassUnified"):
        trajdata_config = translate_pass_trajdata_cfg(config)
        datamodule = eval(cls_name)(data_config=trajdata_config, train_config=config.train)
    else:
        raise NotImplementedError("{} is not a supported datamodule type".format(cls_name))
    return datamodule