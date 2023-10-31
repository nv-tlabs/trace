"""
Helpful script to generate example config files for each algorithm. These should be re-generated
when new config options are added, or when default settings in the config classes are modified.
"""
import os

import tbsim
from tbsim.configs.registry import EXP_CONFIG_REGISTRY
from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.scene_edit_config import SceneEditingConfig


def main():
    # store template config jsons in this directory
    target_dir = os.path.join(tbsim.__path__[0], "../configs/templates/")

    for name, cfg in EXP_CONFIG_REGISTRY.items():
        cfg.dump(filename=os.path.join(target_dir, name + ".json"))

    EvaluationConfig().dump(filename=os.path.join(target_dir, "eval.json"))
    SceneEditingConfig().dump(filename=os.path.join(target_dir, "scene_edit.json"))


if __name__ == "__main__":
    main()
