This code is modified from MADDPG and M3DDPG.

# Robust Multi-Agent Reinforcement Learning with State Uncertainty

This is the code for implementing the Robust Multi-Agent Actor-Critic (RMAAC) algorithm presented in the paper:
[Robust Multi-Agent Reinforcement Learning with State Uncertainty](https://openreview.net/forum?id=CqTkapZ6H9).
It is configured to be run in conjunction with environments from the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).


## Installation

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

You can use the following commands to configure the environment.

`conda create -n rmaac_env python=3.5.4`

`conda activate rmaac_env`

`conda install numpy=1.14.5`

`# conda install -c anaconda tensorflow-gpu`

`conda install tensorflow`

`# conda install gym=0.10.5`

`pip install gym==0.10.5`

## Multi-Agent Particle Environments

We demonstrate here how the code can be used in conjunction with the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

- Download and install the MPE code [here](https://github.com/openai/multiagent-particle-envs)
by following the `README`.

- Ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`).

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python train.py --scenario simple``

- You can replace `simple` with any environment in the MPE you'd like to run.

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries` number of adversaries in the game (default: `0`)


### Core training parameters

- `--lr`: learning rate for agents (default: `1e-2`)

- `--lr-adv`: learning rate for state perturbation adversaries(default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--noise-type`: noise format (default: `Linear`)

- `--noise-variance`: variance of gaussian noise (default: `1`)

- `--constraint-epsilon`: the constraint parameter (default: `0.5`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)


## Paper citation

If you used this code for your experiments or found it helpful, consider citing the following paper:

<pre>
@article{
he2023robust,
title={Robust Multi-Agent Reinforcement Learning with State Uncertainty},
author={Sihong He, Songyang Han, Sanbao Su, Shuo Han, Shaofeng Zou, and Fei Miao},
journal={Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=CqTkapZ6H9}
}
</pre>
