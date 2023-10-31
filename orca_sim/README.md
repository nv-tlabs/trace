# ORCA Sim

Generate synthetic pedestrian datasets using ORCA (optimal reciprocal collision avoidance).

## Introduction

This directory contains the code used to generate our synthetic dataset for training and evaluating TRACE.
It makes use of [Python bindings](https://github.com/sybrenstuvel/Python-RVO2/tree/main) for the [original ORCA simulator](https://github.com/snape/RVO2).

## Installation
Some extra set up is required to install the ORCA simulator.

First, install a few extra requirements
```
pip install -r requirements.txt
```

Then, clone the ORCA sub-repo using:
```
cd Python-RVO2
git submodule init
git submodule update
```

And install it:
```
python setup.py build
python setup.py install
cd ..
```

## Data Generation
Here are the commands used to generate the two subsets of our synthetic ORCA dataset:
```
# ORCA Maps data generation
python gen_dataset.py --num_scenes 1000 --scene_len 10 --max_agents 10 --max_obstacles 20 --viz --out datagen_out/orca_sim/maps --resample_goal

# ORCA no maps data generation
python gen_dataset.py --num_scenes 1000 --scene_len 10 --max_agents 20 --max_obstacles 0 --viz --out datagen_out/orca_sim/no_maps --resample_goal
```

You can alternatively use the `--viz_vid` flag to render videos of the simulations.