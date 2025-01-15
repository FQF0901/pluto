# PLUTO

This is the official repository of

**PLUTO: Push the Limit of Imitation Learning-based Planning for Autonomous Driving**,

[Jie Cheng](https://jchengai.github.io/), [Yingbing Chen](https://sites.google.com/view/chenyingbing-homepage), and [Qifeng Chen](https://cqf.io/)


<p align="left">
<a href="https://jchengai.github.io/pluto">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/abs/2404.14327' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

## Setup Environment

### Setup dataset

Setup the nuPlan dataset following the [offiical-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

### Setup conda environment

```
conda create -n pluto python=3.9
conda activate pluto

# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt【另需安装imageio作可视化，注意numpy可能需要降级为1.23.5】

# setup pluto
cd ..
git clone https://github.com/jchengai/pluto.git && cd pluto
sh ./script/setup_env.sh

【服务器也需要改下setup_env.sh：】
#!/bin/sh

# 安装PyTorch和torchvision，同时指定可信主机
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --trusted-host download.pytorch.org

# 安装natten并指定可信主机
pip3 install natten==0.14.6 -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html --trusted-host shi-labs.com

# 安装其他依赖项，并传递任何需要的可信主机（如果requirements.txt中有外部链接）
pip install -r ./requirements.txt --trusted-host download.pytorch.org --trusted-host data.pyg.org --trusted-host shi-labs.com
```

## Feature Cache

Preprocess the dataset to accelerate training. It is recommended to run a small sanity check to make sure everything is correctly setup.

```
 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan_mini \
    cache.cache_path=/nuplan/exp/sanity_check \ # 【本代码产物存放地址：/home/fqf/nuplan/exp/sanity_check】
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_tiny \
    worker=sequential

【正确应该：】
python run_training.py py_func=cache +training=train_pluto scenario_builder=nuplan cache.cache_path=/home/fqf/nuplan/exp/sanity_check cache.cleanup_cache=true scenario_filter=training_scenarios_tiny worker=sequential
```

nuplan的一些配置
```
【scenario_builder】
1. nuplan: 默认的场景构建器。支持从日志文件中提取场景，并根据配置文件或过滤器生成仿真场景
2. nuplan_challenge: 专门用于 nuPlan 挑战赛的场景构建器。包含一些特定的预处理步骤或过滤器，以适应挑战赛的需求
3. nuplan_test: 用于测试集的场景构建器。
4. nuplan_mini: 用于小型数据集或演示的场景构建器。通常只包含少量场景，用于快速测试和演示

【scenario_filter】
1. 预定义的场景过滤器: mini_demo_scenario, training_scenarios, validation_scenarios, test_scenarios, all_scenarios
2. 基于场景类型的过滤器: intersection, highway, roundabout, pedestrian_crossing, lane_change, stop_sign, traffic_light
3. 难度级别: easy, medium, hard
4. 天气条件: clear, rain, snow
5. 时间条件: day, night
6. 地理位置: boston, singapore, pittsburgh
eg: 
python run_simulation.py \
    scenario_builder=nuplan \
    scenario_filter=mini_demo_scenario \
    scenario_filter.difficulty=hard \
    scenario_filter.weather=clear \
    scenario_filter.time_of_day=day \
    ...
```

Then preprocess the whole nuPlan training set (this will take some time). You may need to change `cache.cache_path` to suit your condition

```
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

## Training

(The training part it not fully tested)

Same, it is recommended to run a sanity check first:

```
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=4 \
  scenario_builder=nuplan cache.cache_path=/nuplan/exp/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1

【正确应该：】
CUDA_VISIBLE_DEVICES=0 python run_training.py py_func=train +training=train_pluto worker=single_machine_thread_pool worker.max_workers=4 scenario_builder=nuplan cache.cache_path=/home/fqf/nuplan/exp/sanity_check cache.use_cache_without_dataset=true data_loader.params.batch_size=4 data_loader.params.num_workers=1
```

Training on the full dataset (without CIL):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/nuplan/exp/cache_pluto_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto
```

- add option `model.use_hidden_proj=true +custom_trainer.use_contrast_loss=true` to enable CIL.

- you can remove wandb related configurations if your prefer tensorboard.


## Checkpoint

Download and place the checkpoint in the `pluto/checkpoints` folder.

| Model            | Download |
| ---------------- | -------- |
| Pluto-1M-aux-cil | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EaFpLwwHFYVKsPVLH2nW5nEBNbPS7gqqu_Rv2V1dzODO-Q?e=LAZQcI)    |


## Run Pluto-planner simulation

Run simulation for a random scenario in the nuPlan-mini split

```
sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario pluto_1M_aux_cil.ckpt /dir_to_save_the_simulation_result_video

【正确应该：】
sh ./script/run_pluto_planner.sh pluto_planner nuplan mini_demo_scenario pluto_1M_aux_cil.ckpt /home/fqf/fqf_folder/01_Git/pluto/simulation_result_video
```

The rendered simulation video will be saved to the specified directory (need change `/dir_to_save_the_simulation_result_video`).

## To Do

The code is under cleaning and will be released gradually.

- [ ] improve docs
- [x] training code
- [x] visualization
- [x] pluto-planner & checkpoint
- [x] feature builder & model
- [x] initial repo & paper

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@article{cheng2024pluto,
  title={PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving},
  author={Cheng, Jie and Chen, Yingbing and Chen, Qifeng},
  journal={arXiv preprint arXiv:2404.14327},
  year={2024}
}
```
