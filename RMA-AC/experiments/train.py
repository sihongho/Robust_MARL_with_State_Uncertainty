import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import os
import gym

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


def parse_args():
    parser = argparse.ArgumentParser("robust multi-agent actor-critic experiments for multiagent environments")
    parser.add_argument("--server-name", type=str, default="MyServer", help="type in what host machine the experiments run on")
    parser.add_argument("--do-not-use-send-email", action="store_true", default=True)
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries in the game")
    parser.add_argument("--good-policy", type=str, default="rmaac", help="policy for good agents in the game")
    parser.add_argument("--adv-policy", type=str, default="rmaac", help="policy of adversaries in the game")
    parser.add_argument("--noise-policy", type=str, default="rmaac", help="policy of state perturbation adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Agent Adam optimizer")
    parser.add_argument("--lr-adv", type=float, default=1e-2, help="learning rate for State Perturbation Adversary Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    # State Perturbation
    parser.add_argument("--noise-type", type=str, default="Linear", help="Linear, Gaussian")
    parser.add_argument("--noise-variance", type=float, default=1, help="variance of gaussian noise, 0.1, 0.2, 0.5, 1, 2, 3")
    parser.add_argument("--constraint-epsilon", type=float, default=0.5, help="the constraint parameter: 0.1, 0.25, 0.5, 0.75, 1, 2")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def mlp_model_adv(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, constraint_epsilon = 0.5):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return tf.clip_by_value(out, clip_value_min=-constraint_epsilon, clip_value_max=constraint_epsilon)



def get_noise(adv_action_n, var):
    noise = []
    for i in range(len(adv_action_n)):
        noise.append(np.random.normal(adv_action_n[i], var)) 
    return np.array(noise)

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    p_model = mlp_model
    q_model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "r_agent_%d" % i, p_model, q_model, obs_shape_n, env.observation_space, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "r_agent_%d" % i, p_model, q_model, obs_shape_n, env.observation_space, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def get_adversaries(env, obs_shape_n, arglist):
    adversaries = []

    def mlp_model_adv_arg(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, constraint_epsilon = arglist.constraint_epsilon):
        # This model takes as input an observation and returns values of all actions
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            return tf.clip_by_value(out, clip_value_min=-constraint_epsilon, clip_value_max=constraint_epsilon)
    
    p_model = mlp_model_adv_arg
    q_model = mlp_model
    adversary = MADDPGAgentTrainer
    observation_space = [gym.spaces.Discrete(env.observation_space[i].shape[0]) for i in range(env.n)]
    # print(observation_space)
    for i in range(env.n):
        adversaries.append(adversary(
            "r_adversary_%d" % i, p_model, q_model, obs_shape_n, env.observation_space, env.observation_space, i, arglist,
            local_q_func=(arglist.noise_policy=='ddpg'), ADV=True))
    return adversaries


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # print("env.action_space is {} ".format(env.action_space))
        # print("env.observation_space is {} ".format(env.observation_space))
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        # print("obs_shape_n is {} ".format(obs_shape_n))
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        # print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        
        adversaries = get_adversaries(env, obs_shape_n, arglist)
        # print('Using noise policy {}'.format(arglist.noise_policy))
        # print('There is {} adversaries'.format(str(len(adversaries))))


        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info

        
        adversary_rewards = [[0.0] for _ in range(env.n)]  # individual adversary reward
        adversary_inf = [[[]]] # placeholder for benchmarking info

        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            
            adv_action_n = [adv.action(obs) for adv, obs in zip(adversaries,obs_n)]
            adv_action_n = np.clip(adv_action_n, a_min = -arglist.constraint_epsilon, a_max = arglist.constraint_epsilon)
            if arglist.noise_type == "Linear":
                # print("Use Linear Noise")
                disturbed_obs_n = [act+obs for act, obs in zip(adv_action_n,obs_n)]
            elif arglist.noise_type == "Gaussian":
                # print("Use Gaussian Noise")
                noise_n = get_noise(adv_action_n = adv_action_n, var = arglist.noise_variance)
                disturbed_obs_n = [act+obs for act, obs in zip(noise_n,obs_n)]
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,disturbed_obs_n)]
            # action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            
            for i, adv in enumerate(adversaries):
                adv.experience(obs_n[i], adv_action_n[i], -rew_n[i], new_obs_n[i], done_n[i], terminal)
            for i, rew in enumerate(rew_n):
                adversary_rewards[i][-1] += -rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                
                adversary_inf.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished agent benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                # loss = agent.update(trainers, train_step)
                loss = agent.update(trainers+adversaries, train_step)
            
            
            adv_loss = None
            for adv in adversaries:
                adv.preupdate()
            for adv in adversaries:
                # adv_loss = adv.update(adversaries, train_step)
                adv_loss = adv.update(adversaries+trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

# if __name__ == '__main__':
#     arglist = parse_args()
#     train(arglist)

from send_email import *
if __name__ == '__main__':
    arglist = parse_args()
    print(arglist)
    if arglist.do_not_use_send_email:
        if not os.path.exists(arglist.save_dir):
            os.mkdir(arglist.save_dir)
        train(arglist)
    else:
        Emails = ServerEmail(mail_host="smtp.gmail.com", 
                                    mail_sender="your.email@gmail.com", 
                                    mail_license="abcdefghijklmnop", 
                                    mail_receivers="sihong.he@uconn.edu", 
                                    server_name=arglist.server_name)
        Emails.send_begin_email(exp_name = arglist.exp_name, arglist = arglist)
        print("send begin email")
        try:
            if not os.path.exists(arglist.save_dir):
                os.mkdir(arglist.save_dir)
            train(arglist)
            Emails.send_end_email(exp_name = arglist.exp_name, arglist = arglist)
            print("send finish email")
        except Exception as e:
            Emails.send_einfo_email(exp_name = arglist.exp_name, einfo = e)
            print("send error email")

