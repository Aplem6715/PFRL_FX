import argparse
import json
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from sklearn import preprocessing


import fx_env_gasf_range


import pfrl
from pfrl.experiments.evaluator import Evaluator
from pfrl.experiments.evaluator import save_agent
from pfrl import action_value
from pfrl import agents
from pfrl import experiments
from pfrl import explorers
from pfrl import nn as pnn
from pfrl import utils
from pfrl.q_function import StateQFunction
from pfrl.q_functions.dueling_dqn import init_chainer_default, constant_bias_initializer
from pfrl import replay_buffers
from pfrl.wrappers import atari_wrappers

import processing
from tqdm import tqdm

linear_features = ['position', 'pips']
gasf_techs = ['SMA5', 'SMA25', 'MACD',
              'MACD_SI', 'BBAND_U2', 'BBAND_L2', 'RSI14', 'MOM25']

nb_kernel1 = 16
nb_kernel2 = 16
# nb_kernel3 = 64
k_size1 = 4
k_size2 = 2
# k_size3 = 3
k_stride1 = 2
k_stride2 = 1
# k_stride3 = 1
dense_units = [64, 64]

LEARN_DURATION = 2500


def train_agent(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    logger=None,
):

    logger = logger or logging.getLogger(__name__)

    episode_r = 0
    episode_idx = 0

    # o_0, r_0
    obs = env.reset()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    episode_len = 0
    try:
        while t < steps:
            if t % 200 == 0:
                print('step: ', t, '/', steps, end='\r')
            # a_t
            action = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(action)
            t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get(
                "needs_reset", False)
            agent.observe(obs, r, done, reset)

            for hook in step_hooks:
                hook(env, agent, t)

            if done or reset or t == steps:
                logger.info(
                    "step:%s episode:%s R:{:.1f} balance:{:.1f}".format(
                        episode_r,
                        env.broker.balance),
                    t,
                    episode_idx,
                )
                logger.info("statistics:%s", agent.get_statistics())
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(
                        t=t, episodes=episode_idx + 1)
                    if (
                        successful_score is not None
                        and evaluator.max_score >= successful_score
                    ):
                        break
                if t == steps:
                    break
                # Start a new episode
                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix="_finish")


def train_agent_with_evaluation(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    train_max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    eval_env=None,
    successful_score=None,
    step_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    logger=None,
):
    """Train an agent while periodically evaluating it.

    Args:
        agent: A pfrl.agent.Agent
        env: Environment train the agent against.
        steps (int): Total number of timesteps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_episodes (int): Number of episodes at each evaluation phase.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output data.
        checkpoint_freq (int): frequency at which agents are stored.
        train_max_episode_len (int): Maximum episode length during training.
        step_offset (int): Time step from which training starts.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If None, train_max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            than or equal to this value if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation
            phase, if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len

    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        save_best_so_far_agent=save_best_so_far_agent,
        use_tensorboard=use_tensorboard,
        logger=logger,
    )

    train_agent(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger,
    )


class MyDistributionalDuelingDQN(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(
        self,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        input_width,
        n_input_channels=4,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        self.z_values = torch.linspace(
            v_min, v_max, n_atoms, dtype=torch.float32)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, nb_kernel1,
                          k_size1, stride=k_stride1),
                nn.Conv2d(nb_kernel1, nb_kernel2, k_size2, stride=k_stride2),
                # nn.Conv2d(nb_kernel2, nb_kernel3, k_size3, stride=k_stride3),
            ]
        )
        input_size = int((input_width - k_size1) / k_stride1) + 1
        input_size = int((input_size - k_size2) / k_stride2) + 1
        input_size = input_size ** 2 * nb_kernel2

        self.main_stream = nn.ModuleList(
            [
                nn.Linear(input_size + len(linear_features), dense_units[0]),
                nn.Linear(dense_units[0], dense_units[1]),
            ]
        )
        self.a_stream = nn.Linear(dense_units[1]//2, n_actions * n_atoms)
        self.v_stream = nn.Linear(dense_units[1]//2, n_atoms)

        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        # split gasf, linear_features
        h, linear = x[0], x[1]
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = h.shape[0]
        # Flatten
        h = h.view(batch_size, -1)
        fc_in = torch.cat([h, linear], dim=1)

        for l in self.main_stream:
            fc_in = self.activation(l(fc_in))

        h_a, h_v = torch.chunk(fc_in, 2, dim=1)
        ya = self.a_stream(h_a).reshape(
            (batch_size, self.n_actions, self.n_atoms))

        mean = ya.sum(dim=1, keepdim=True) / self.n_actions

        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(h.device)
        return action_value.DistributionalDiscreteActionValue(q, self.z_values)


def main():
    df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

    train_df = df[((df['Datetime'] >= dt.datetime(2018, 1, 1))
                   & (df['Datetime'] < dt.datetime(2019, 1, 1)))]
    valid_df = df[((df['Datetime'] >= dt.datetime(2016, 1, 1))
                   & (df['Datetime'] < dt.datetime(2018, 1, 1)))]

    #gasf_cols = ['Open', 'High', 'Low', 'Close'] + gasf_techs
    train_gasf = pickle.load(open('M30_2018-2019_12candle.gasf2', 'rb'))
    valid_gasf = pickle.load(open('M30_2016-2018_12candle.gasf2', 'rb'))
    train_gasf = processing.nwhc2nchw_array(train_gasf)
    valid_gasf = processing.nwhc2nchw_array(valid_gasf)
    train_gasf = train_gasf.astype(np.float32)
    valid_gasf = valid_gasf.astype(np.float32)
    # 環境の生成
    #train_env = fx_env.FxEnv(train_df, scaler)
    #valid_env = fx_env.FxEnv(valid_df, scaler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results_rainbow",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed [0, 2 ** 31)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--demo", action="store_true", default=False)

    # Greedy Setting
    parser.add_argument("--start-epsilon", type=float, default=0.4)
    parser.add_argument("--end-epsilon", type=float, default=0.005)
    parser.add_argument("--greedy-steps", type=int,
                        default=(5 * 10 ** 6) // 4)

    # Agent Settings
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--noisy-net-sigma", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=5 * 10 ** 6)

    parser.add_argument("--replay-start-size", type=int, default=10 ** 4)
    parser.add_argument("--eval-n-steps", type=int, default=75000)
    parser.add_argument("--eval-interval", type=int, default=25000)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument("--n-best-episodes", type=int, default=200)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test):
        '''
        if test:
            return fx_env_gasf.FxEnv_GASF(valid_df, valid_gasf, mode=fx_env_gasf.FxEnv_GASF.TEST_MODE)
        else:
            return fx_env_gasf.FxEnv_GASF(train_df, train_gasf, mode=fx_env_gasf.FxEnv_GASF.TRAIN_MODE)
        '''
        if test:
            return fx_env_gasf_range.FxEnv_GASF(valid_df, valid_gasf, mode=fx_env_gasf_range.FxEnv_GASF.TEST_MODE, trade_duration=len(valid_df))
        else:
            return fx_env_gasf_range.FxEnv_GASF(train_df, train_gasf, mode=fx_env_gasf_range.FxEnv_GASF.TRAIN_MODE, trade_duration=LEARN_DURATION)

    env = make_env(test=False)
    eval_env = make_env(test=True)

    obs_shape = env.observation_space.low.shape
    n_actions = env.action_space.n
    obs_ch = obs_shape[1]
    obs_width = obs_shape[2]

    n_atoms = 51
    v_max = 1
    v_min = -1
    # 128ユニット3層のDeep Q関数
    q_func = MyDistributionalDuelingDQN(
        n_actions, n_atoms, v_min, v_max, input_width=obs_width, n_input_channels=obs_ch)

    # Noisy nets
    # pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
    # Turn off explorer
    explorer = explorers.LinearDecayEpsilonGreedy(
        args.start_epsilon, args.end_epsilon, args.greedy_steps, env.action_space.sample)

    # Use the same hyper parameters as https://arxiv.org/abs/1710.02298
    opt = torch.optim.Adam(q_func.parameters(), 6.25e-5, eps=1.5 * 10 ** -4)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 10
    betasteps = args.steps / update_interval
    rbuf = replay_buffers.PrioritizedReplayBuffer(
        10 ** 5,
        alpha=0.5,
        beta0=0.4,
        betasteps=betasteps,
        num_steps=3,
        normalize_by_max="memory",
    )

    '''
    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255
    '''

    Agent = agents.CategoricalDoubleDQN
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=args.gamma,
        explorer=explorer,
        minibatch_size=32,
        replay_start_size=args.replay_start_size,
        target_update_interval=3200,
        update_interval=update_interval,
        batch_accumulator="mean",
    )

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
        )
        print(
            "n_episodes: {} mean: {} median: {} stdev {}".format(
                eval_stats["episodes"],
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )

    else:
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 200 evaluation episodes, each capped at 30 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=5*10**4,
            logger=None,
        )
        with open(os.path.join(args.outdir, "bestscores.json"), "w") as f:
            json.dump(stats, f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))


if __name__ == "__main__":
    main()
