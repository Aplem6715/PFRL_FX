# パッケージのインポート
import pfrl
import argparse
import torch
import torch.nn
import gym
import numpy as np
import fx_env
import pandas as pd
import datetime as dt

from torch import nn
from pfrl import experiments
from pfrl.agents import acer
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.replay_buffers import EpisodicReplayBuffer

from sklearn import preprocessing


parser = argparse.ArgumentParser()
parser.add_argument("--processes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed [0, 2 ** 31)")
parser.add_argument(
    "--outdir",
    type=str,
    default="results_acer",
    help=(
        "Directory path to save output files."
        " If it does not exist, it will be created."
    ),
)
parser.add_argument("--t-max", type=int, default=5)
parser.add_argument("--replay-start-size", type=int, default=10000)
parser.add_argument("--n-times-replay", type=int, default=4)
parser.add_argument("--beta", type=float, default=1e-2)
parser.add_argument("--profile", action="store_false")
parser.add_argument("--steps", type=int, default=10 ** 7)

parser.add_argument("--lr", type=float, default=7e-4)
parser.add_argument("--eval-interval", type=int, default=10 ** 5)
parser.add_argument("--eval-n-runs", type=int, default=5)
args = parser.parse_args()


df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

scaler = preprocessing.MinMaxScaler()
scaler.fit(df.iloc[:, 1:])

train_df = df[((df['Datetime'] >= dt.datetime(2016, 1, 1))
               & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
valid_df = df[((df['Datetime'] >= dt.datetime(2018, 6, 1))
               & (df['Datetime'] < dt.datetime(2019, 1, 1)))]
# 環境の生成
train_env = fx_env.FxEnv(train_df, scaler)
valid_env = fx_env.FxEnv(valid_df, scaler)

# Q関数の定義
obs_size = train_env.observation_space.low.size
n_actions = train_env.action_space.n

input_to_hidden = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
)

head = acer.ACERDiscreteActionHead(
    pi=nn.Sequential(nn.Linear(32, n_actions), SoftmaxCategoricalHead(),),
    q=nn.Sequential(nn.Linear(32, n_actions), DiscreteActionValueHead(),),
)

model = nn.Sequential(input_to_hidden, head)
model.apply(pfrl.initializers.init_chainer_default)
opt = pfrl.optimizers.SharedRMSpropEpsInsideSqrt(
    model.parameters(), lr=args.lr, eps=4e-3, alpha=0.99
)
replay_buffer = EpisodicReplayBuffer(10 ** 6 // args.processes)


def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255


# エージェントの生成
agent = acer.ACER(
    model,
    opt,
    t_max=args.t_max,
    gamma=0.99,
    replay_buffer=replay_buffer,
    n_times_replay=args.n_times_replay,
    replay_start_size=args.replay_start_size,
    beta=args.beta,
    phi=phi,
    max_grad_norm=40,
    recurrent=False,
)


def make_env(process_idx, test):
    df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df.iloc[:, 1:])

    train_df = df[((df['Datetime'] >= dt.datetime(2016, 1, 1))
                   & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
    return fx_env.FxEnv(train_df, scaler)


# Linearly decay the learning rate to zero
def lr_setter(env, agent, value):
    for pg in agent.optimizer.param_groups:
        assert "lr" in pg
        pg["lr"] = value


lr_decay_hook = experiments.LinearInterpolationHook(
    args.steps, args.lr, 0, lr_setter
)
experiments.train_agent_async(
    agent=agent,
    outdir=args.outdir,
    processes=args.processes,
    make_env=make_env,
    profile=args.profile,
    steps=args.steps,
    eval_n_steps=None,
    eval_n_episodes=args.eval_n_runs,
    eval_interval=args.eval_interval,
    global_step_hooks=[lr_decay_hook],
    save_best_so_far_agent=True,
)


'''
# エージェントのテスト
with agent.eval_mode():
    # 環境のリセット
    obs = valid_env.reset()
    R = 0  # エピソード報酬

    # ステップの反復
    while True:
        # 環境の描画
        valid_env.render()

        # 環境の1ステップ実行
        action = agent.act(obs)
        obs, r, done, _ = valid_env.step(action)
        R += r
        agent.observe(obs, r, done, False)

        # エピソード完了
        if done:
            break
    print('R:', R)
'''
agent.save('agent_acer')
