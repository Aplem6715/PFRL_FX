import argparse
import json
import os

import torch
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import fx_env2

from sklearn import preprocessing

import pfrl
from pfrl import agents
from pfrl import experiments
from pfrl import explorers
from pfrl import nn as pnn
from pfrl import utils
from pfrl.q_functions import DistributionalFCStateQFunctionWithDiscreteAction
from pfrl import replay_buffers
from pfrl.wrappers import atari_wrappers


def main():

    df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df.iloc[:, 1:])

    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    train_df = df[((df['Datetime'] >= dt.datetime(2016, 1, 1))
                   & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
    valid_df = df[((df['Datetime'] >= dt.datetime(2018, 6, 1))
                   & (df['Datetime'] < dt.datetime(2019, 1, 1)))]
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

    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--noisy-net-sigma", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=5 * 10 ** 6)

    parser.add_argument("--replay-start-size", type=int, default=10 ** 4)
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    parser.add_argument("--eval-interval", type=int, default=250000)
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
        if test:
            return fx_env2.FxEnv(valid_df, scaler)
        else:
            return fx_env2.FxEnv(train_df, scaler)

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    n_obs = env.observation_space.low.size

    n_atoms = 51
    v_max = 10
    v_min = -10
    # 128ユニット3層のDeep Q関数
    q_func = DistributionalFCStateQFunctionWithDiscreteAction(
        n_obs, n_actions, n_atoms, v_min, v_max, 128, 3)

    # Noisy nets
    pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
    # Turn off explorer
    explorer = explorers.Greedy()

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

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = agents.CategoricalDoubleDQN
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        minibatch_size=32,
        replay_start_size=args.replay_start_size,
        target_update_interval=3200,
        update_interval=update_interval,
        batch_accumulator="mean",
        phi=phi,
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
        experiments.train_agent_with_evaluation(
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