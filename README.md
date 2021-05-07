# MineRL2021-Intro-baselines
MineRL 2021 Intro track has three baseline agents:
1. Fully scripted
2. Behavioural cloning (BC) plus scripted
3. Reinforcement Learning (RL) plus scripted

The agents can be run in three different ways:
1. Colab notebook
2. Standalone file
3. Submittable repository (Under construction)

## Colab notebooks
This is the same code as in standalone files, but with extra documentation and fully running in Colab. The RL_plus_script is not included as it takes too long to run fully in Colab. The notebooks are:

[MineRL fully scripted on a fixed seed](https://colab.research.google.com/drive/1laXCpyf0k6O8Oo1AvUK4UrnywK7IcEh3?usp=sharing)

[MineRL fully scripted](https://colab.research.google.com/drive/1ipj34U_Ub8IsTO0I80o4bUTtltERMErm?usp=sharing)

[MineRL BC+scripted](https://colab.research.google.com/drive/1qfjHCQkukFcR9w1aPvGJyQxOa-Gv7Gt_?usp=sharing)

## Standalone files
These agents are located in the [standalone](https://github.com/KarolisRam/MineRL2021-Intro-baselines/tree/main/standalone) directory.  
MineRL requires [JDK8](https://www.minerl.io/docs/tutorials/index.html) to be installed first.
After that, run:  
```
pip install --upgrade minerl
pip install pytorch
pip install stable_baselines3
```
The agents can be run with:  
```
python fully_scripted_fixed_seed.py
```
```
python fully_scripted.py
```
```
python BC_plus_script.py
```
```
python RL_plus_script.py
```
The BC and RL ones come with pretrained models. If you want to train them yourself, you will have to uncomment the `train()` line.

## Submittable repository
Under construction.
