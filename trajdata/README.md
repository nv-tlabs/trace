# Fork of Unified Trajectory Data Loader (trajdata) for TRACE

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/488789438.svg)](https://zenodo.org/badge/latestdoi/488789438)

## Introduction
This is a fork of [trajdata](https://github.com/NVlabs/trajdata) used for TRACE. Main changes include:
* Added support for the synthetic ORCA crowd simulation dataset (see [`orca_dataset.py`](./src/trajdata/dataset_specific/orca_peds/orca_dataset.py))
* Added script to compute dataset statistics to be used for normalization (see [`compute_norm_info.py`](./examples/compute_norm_info.py))
* Small updates to how stats are computed for simulations (see [`sim_stats.py`](./src/trajdata/simulation/sim_stats.py))

Please follow the instruction in the [main TRACE README](../README.md) for installing dependencies and downloading datasets. 
The remainder of this README is copied from the original codebase.

Note that this is now a fairly old version of `trajdata`; please see their [current repo](https://github.com/NVlabs/trajdata) if you're interested in other datasets or an updated version.

## Installation

The easiest way to install trajdata is through PyPI with
```sh
pip install trajdata
```

In case you would also like to use datasets such as nuScenes and Lyft Level 5 (which require their own devkits to access raw data), the following will also install the respective devkits.
```sh
# For nuScenes
pip install "trajdata[nusc]"

# For Lyft
pip install "trajdata[lyft]"

# Both
pip install "trajdata[nusc,lyft]"
```
Then, download the raw datasets (nuScenes, Lyft Level 5, ETH/UCY, etc) in case you do not already have them. For more information about how to structure dataset folders/files, please see [`DATASETS.md`](./DATASETS.md).

### Package Developer Installation

First, in whichever environment you would like to use (conda, venv, ...), make sure to install all required dependencies with
```
pip install -r requirements.txt
```
Then, install trajdata itself in editable mode with
```
pip install -e .
```

## Data Preprocessing [Optional]
The dataloader operates via a two-stage process, visualized below.
![architecture](./img/architecture.png)
While optional, we recommend first preprocessing data into a canonical format. Take a look at the `examples/preprocess_data.py` script for an example script that does this. Data preprocessing will execute the first part of the diagram above and create data caches for each specified dataset.

**Note**: Explicitly preprocessing datasets like this is not necessary; the dataloader will always internally check if there exists a cache for any requested data and will create one if not.

## Data Loading

At a minimum, batches of data for training/evaluation/etc can be loaded the following way:
```py
import os
from torch.utils.data import DataLoader
from trajdata import AgentBatch, UnifiedDataset

# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["nusc_mini"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nusc_mini": "~/datasets/nuScenes"
    },
)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
)

batch: AgentBatch
for batch in dataloader:
    # Train/evaluate/etc.
    pass
```

For a more comprehensive example, please see `examples/batch_example.py`.

For more information on all of the possible `UnifiedDataset` constructor arguments, please see `src/trajdata/dataset.py`.

## Supported Datasets
Currently, the dataloader supports interfacing with the following datasets:

| Dataset | ID | Splits | Add'l Tags | Description | dt | Maps |
|---------|----|--------|------------|-------------|----|------|
| nuScenes Train/Val | `nusc_trainval` | `train`, `val` | `boston`, `singapore` | nuScenes' training/validation splits (700/150 scenes) | 0.5s (2Hz) | :white_check_mark: |
| nuScenes Test | `nusc_test` | `test` | `boston`, `singapore` | nuScenes' test split (150 scenes) | 0.5s (2Hz) | :white_check_mark: |
| nuScenes Mini | `nusc_mini` | `mini_train`, `mini_val` | `boston`, `singapore` | nuScenes mini training/validation splits (8/2 scenes) | 0.5s (2Hz) | :white_check_mark: |
| Lyft Level 5 Train | `lyft_train` | `train` | `palo_alto` | Lyft Level 5 training data - part 1/2 (8.4 GB) | 0.1s (10Hz) | :white_check_mark: |
| Lyft Level 5 Train Full | `lyft_train_full` | `train` | `palo_alto` | Lyft Level 5 training data - part 2/2 (70 GB) | 0.1s (10Hz) | :white_check_mark: |
| Lyft Level 5 Validation | `lyft_val` | `val` | `palo_alto` | Lyft Level 5 validation data (8.2 GB) | 0.1s (10Hz) | :white_check_mark: |
| Lyft Level 5 Sample | `lyft_sample` | `mini_train`, `mini_val` | `palo_alto` | Lyft Level 5 sample data (100 scenes, randomly split 80/20 for training/validation) | 0.1s (10Hz) | :white_check_mark: |
| ETH - Univ | `eupeds_eth` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `zurich` | The ETH (University) scene from the ETH BIWI Walking Pedestrians dataset | 0.4s (2.5Hz) | |
| ETH - Hotel | `eupeds_hotel` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `zurich` | The Hotel scene from the ETH BIWI Walking Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Univ | `eupeds_univ` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The University scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Zara1 | `eupeds_zara1` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The Zara1 scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Zara2 | `eupeds_zara2` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The Zara2 scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |

## Examples

### Multiple Datasets
The following will load data from both the nuScenes mini dataset as well as the ETH - University scene from the ETH BIWI Walking Pedestrians dataset.

```py
dataset = UnifiedDataset(
    desired_data=["nusc_mini", "eupeds_eth"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nusc_mini": "~/datasets/nuScenes",
        "eupeds_eth": "~/datasets/eth_ucy_peds"
    },
    desired_dt=0.1, # Please see the note below about common dt!
)
```

**Note**: Be careful about loading multiple datasets without an associated `desired_dt` argument; many datasets do not share the same underlying data annotation frequency. To address this, we've implemented timestep interpolation to a common frequency which will ensure that all batched data shares the same dt. Interpolation can only be performed to integer multiples of the original data annotation frequency. For example, nuScenes' `dt=0.5` and the ETH BIWI dataset's `dt=0.4` can be interpolated to a common `desired_dt=0.1`.

## Adding New Datasets
The code that interfaces raw datasets can be found in `src/trajdata/dataset_specific`.

To add a new dataset, ...

## Simulation Interface
One additional feature of trajdata is that it can be used to initialize simulations from real data and track resulting agent motion, metrics, etc. 

At a minimum, a simulation can be initialized and stepped through as follows (also present in `examples/simple_sim_example.py`):
```py
from typing import Dict # Just for type annotations

import numpy as np

from trajdata import AgentBatch, UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene # Just for type annotations
from trajdata.simulation import SimulationScene

# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["nusc_mini"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nusc_mini": "~/datasets/nuScenes",
    },
)

desired_scene: Scene = dataset.get_scene(scene_idx=0)
sim_scene = SimulationScene(
    env_name="nusc_mini_sim",
    scene_name="sim_scene",
    scene=desired_scene,
    dataset=dataset,
    init_timestep=0,
    freeze_agents=True,
)

obs: AgentBatch = sim_scene.reset()
for t in range(1, sim_scene.scene_info.length_timesteps):
    new_xyh_dict: Dict[str, np.ndarray] = dict()

    # Everything inside the forloop just sets
    # agents' next states to their current ones.
    for idx, agent_name in enumerate(obs.agent_name):
        curr_yaw = obs.curr_agent_state[idx, -1]
        curr_pos = obs.curr_agent_state[idx, :2]

        next_state = np.zeros((3,))
        next_state[:2] = curr_pos
        next_state[2] = curr_yaw
        new_xyh_dict[agent_name] = next_state

    obs = sim_scene.step(new_xyh_dict)
```

`examples/sim_example.py` contains a more comprehensive example which initializes a simulation from a scene in the nuScenes mini dataset, steps through it by replaying agents' GT motions, and computes metrics based on scene statistics (e.g., displacement error from the original GT data, velocity/acceleration/jerk histograms).

## TODO
- Create a method like finalize() which writes all the batch information to a TFRecord/WebDataset/some other format which is (very) fast to read from for higher epoch training.
- Add more examples to the README.
- Finish README section about how to add a new dataset.
