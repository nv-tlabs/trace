# TRACE: Trajectory Diffusion Model for Controllable Pedestrians

Official implementation of TRACE, the TRAjectory Diffusion Model for Controllable PEdestrians, from the CVPR 2023 paper: "Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion".

**Note: this repo only contains the TRACE component of this paper** (_i.e._, the trajectory diffusion model). For PACER (the pedestrian animation controller), please see [this other repository](https://github.com/nv-tlabs/pacer).

[[Paper]](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf) [[Website]](https://research.nvidia.com/labs/toronto-ai/trace-pace/) [[Video]](https://www.youtube.com/watch?v=225c52QDkzg)

<div float="center">
    <img src="assets/gif/trace_pace.gif" />
</div>

## Overview
This repo is built upon [tbim](https://github.com/NVlabs/traffic-behavior-simulation) for training and simulation and [trajdata](https://github.com/NVlabs/trajdata) for data loading. 
For ease of use, this repo included minimal versions of these libraries along with heavy modifications to support the model and data of TRACE. 
We additionally include the code used to create our synthetic ORCA dataset. 

These components are located in the following top-level directories:
* [tbsim](./tbsim)
* [trajdata](./trajdata)
* [orca_sim](./orca_sim)

This README focuses on setting up and running the main TRACE model. To generate synthetic data with ORCA, please see [this separate README](./orca_sim/README.md).

## Dependencies
> This codebase was developed using Python 3.8, PyTorch 1.10.2, and CUDA 11.1.

To evaluate and train TRACE, first install a version of [PyTorch](https://pytorch.org/get-started/previous-versions/) and `torchvision` that works with your CUDA. 

Then install the requirements for both `tbsim` and `trajdata` contained in the single requirements file:
```
pip install -r requirements.txt
```

Finally, install `tbsim` and `trajdata`:
```
pip install -e .;
cd trajdata;
pip install -e .;
cd ..
```

## Downloads
### Datasets
Please see the [datasets README](./datasets/) for how all downloaded data should be structured in the `datasets` directory.

The nuScenes dataset should be downloaded from the [webpage](https://www.nuscenes.org/nuscenes#download) and placed in [`datasets/nuscenes`](./datasets/). Note, only the maps and full dataset metadata are needed.

The full ETH/UCY dataset can be downloaded as pre-processed text files from [this repo](https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw/raw/all_data). They should be placed in [`datasets/eth_ucy`](./datasets/).

The dataset generated with the [ORCA crowd simulator](./orca_sim) is available to download [on Google Drive](https://drive.google.com/file/d/17ANk-ZKZT0VcXJLM3oAxqdRyU6bIVVqg/view?usp=sharing). It should be unzipped and placed in the `datasets` directory.

### Pre-trained Models
The pre-trained TRACE models evaluated in the paper are available for download [from this link](https://drive.google.com/file/d/1pTlZDUQIqmMRDpNssdfOUGDFE3sQpKmF/view?usp=drive_link). They should be unzipped in the `ckpt` directory to use the commands detailed below.

## Running Test-Time Guidance
First, let's see how to run the pre-trained models and use guidance at test time. 

To run the ORCA-trained model on the whole ORCA-Maps test set for a single guidance configuration (e.g., going to a target waypoint or `target_pos`):
```
python scripts/scene_editor.py --results_root_dir ./out/orca_mixed_out --num_scenes_per_batch 1 --policy_ckpt_dir ./ckpt/trace/orca_mixed --policy_ckpt_key iter40000 --eval_class Diffuser --render_img --config_file ./configs/eval/orca/target_pos.json
```
This script uses TRACE to control and simulate all agents in each test-set scene. The configuration for evaluation in this example is in `./configs/eval/orca/target_pos.json` and can be changed to modify test-time operation. For example, `class_free_guide_w` determines the strength of classifier-free guidance, `guide_as_filter_only` will only use filtering (i.e. choosing the best sample) and not full test-time guidance, and `guide_clean` determines whether our proposed _clean_ guidance is used or _noisy_ guidance from previous work. Finally, the `heuristic_config` field determines which guidance objectives are used. All guidance objectives are implemented in [`tbsim/utils/guidance_loss.py`](./tbsim/utils/guidance_loss.py).

`./configs/eval/orca` includes additional configurations for the different kinds of guidance presented in the paper.

To run the model trained on a mixture of nuScenes and ETH/UCY data on held out nuScenes data:
```
python scripts/scene_editor.py --results_root_dir ./out/nusc_eth_ucy_mixed_out --num_scenes_per_batch 1 --policy_ckpt_dir ./ckpt/trace/nusc_eth_ucy_mixed --policy_ckpt_key iter40000 --eval_class Diffuser --render --config_file ./configs/eval/nusc/target_pos_perturb.json
```
In this case, we use `--render` instead of `--render_img` to create a video of each simulation since the model is used in a closed loop rather than open-loop as with the ORCA data.

### Computing Metrics
To compute metrics after running a single guidance configuration, use:
```
python scripts/parse_scene_edit_results.py --results_dir ./out/orca_mixed_out/orca_map_open_loop_target_pos --gt_hist ./out/ground_truth/orca_map_gt/hist_stats.json
```
This outputs a directory called `eval_out` in the results directory that contains metrics and plots. The `--gt_hist` argument passes in histogram statistics from the ground truth data needed to compute metrics like EMD. Note that metrics that are not applicable for a certain guidance configuration are simply output as `nans` in the csv files.

If you run multiple guidance configurations and write the output results to the same directory (e.g. `./out/orca_mixed_out`), you can then compute metrics for all of them at once and easily compare with:
```
python scripts/parse_scene_edit_results.py --results_set ./out/orca_mixed_out --gt_hist ./out/ground_truth/orca_map_gt/hist_stats.json
```
This will output a file called `results_agg_set.csv` in the specified `--results_set` directory that compares performance across the different configurations.

These examples are shown for the ORCA data, but similar ones work for nuScenes as well (make sure to change `--gt_hist` to `./out/ground_truth/nusc_gt/hist_stats.json`).

### Custom Configurations
The easiest way to experiment with guidance configs is to copy and modify the most relevant ones from `configs/eval`.

However, if you prefer to start from "scratch" you can see the default template guidance config in `configs/template/scene_edit.json`. These templates are generated by running `python scripts/generate_config_templates.py`, which creates template json files from the config classes in `tbsim/configs`. If you change the default config class for guidance in `tbsim/configs/scene_edit_config.py`, you can run the `generate_config_templates` script to get a usable json version of it.

## Training
To train TRACE on mixed ORCA data, use:
```
python scripts/train.py --output_dir ./out/train_orca_mixed --config_file ./configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json --wandb_project_name trace 
```
To use wandb for logging, make sure to set your API key beforehand with `export WANDB_APIKEY=your_api_key` and customize `--wandb_project_name`. Alternatively, adding the `--log_local` flag will use tensorboard locally to log instead of wandb. 

To train on a mix of nuScenes and ETH/UCY data:
```
python scripts/train.py --output_dir ./out/train_eth_ucy_nusc_mixed --config_file ./configs/train/diffuser_mixed_nusc_mlp_hist_grid_map_drop_10.json --wandb_project_name trace
```
Instead of using a config file, you can also use the `--config_name` argument to use a default data and model configuration (e.g. `orca_diff` for mixed ORCA data). For a full list of possible configs, see [`tbsim/configs/registry.py`](./tbsim/configs/registry.py). 

If you want to modify settings of the model, it's easiest to update the json config file. However, you can also modify the default configuration in [`tbsim/configs/algo_config.py`](./tbsim/configs/algo_config.py) (under `DiffuserConfig`) and then use `python scripts/generate_config_templates.py` to get a resulting json file. Importantly, make sure the `norm_info` settings for the model config are for the dataset you intend to use before exporting to a json file.

## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}            
```

## Acknowledgments
This codebase is built upon several awesome prior works. Please see the associated licenses for each of these codebases which are included [in this repo](./assets/licenses/) and adhere to them when using this codebase.
* [Traffic Behavior Simulator (tbsim)](https://github.com/NVlabs/traffic-behavior-simulation) [[License]](./LICENSE) is the basis of our training and simulation pipeline contained in the [`tbsim` directory](./tbsim/). We heavily modified it to support the necessary datasets from `trajdata` and implemented our new models and guidance.
* [Unified Trajectory Data Loader (trajdata)](https://github.com/NVlabs/trajdata) [[License]](./assets/licenses/trajdata) is used for data loading and is in the [`trajdata` directory](./trajdata/). We extended the library to support our new synthetic ORCA dataset.
* [Diffuser](https://github.com/jannerm/diffuser/) [[License]](./assets/licenses/diffuser) is the basis of our [TRACE model code](./tbsim/models/trace.py). It was heavily modified as described in our paper to support action denoising, conditioning, and test-time guidance.
* [Optimal Reciprocal Collision Avoidance (ORCA)](https://github.com/sybrenstuvel/Python-RVO2) [[License]](./assets/licenses/orca) was used to generate our synthetic pedestrian dataset.

## License
Please see [the license](./LICENSE) for using the code in this repo. This does not apply to the `trajdata` code, which contains [its own license](./trajdata/LICENSE).

The synethetic ORCA dataset and all pre-trained models provided in this repo are separately licensed under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
