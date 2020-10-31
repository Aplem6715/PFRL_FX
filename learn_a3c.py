# パッケージのインポート
import pfrl
import torch
import torch.nn as nn
import gym
import numpy as np
import fx_env_gasf_concat
import pandas as pd
import datetime as dt
import cProfile
import pprint
import pickle
import argparse
from sklearn import preprocessing

from pfrl import experiments
from pfrl import utils
from pfrl.agents import a3c
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt
from pfrl.action_value import DiscreteActionValue

import processing

linear_features = ['position', 'pips']
gasf_techs = ['SMA5', 'SMA25', 'MACD',
              'MACD_SI', 'BBAND_U2', 'BBAND_L2', 'RSI14', 'MOM25']

nb_kernel1 = 16
#nb_kernel2 = 16
k_size1 = 5
#k_size2 = 2
k_stride1 = 1
#k_stride2 = 1
dense_units = [126, 128, 64, 64]


class A3C_Q_Func(torch.nn.Module):
    def __init__(
        self,
        n_actions,
        input_width,
        n_input_channels=4,
        activation=torch.relu,
    ):

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        super().__init__()

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, nb_kernel1,
                          k_size1, stride=k_stride1),
                # nn.Conv2d(nb_kernel1, nb_kernel2, k_size2, stride=k_stride2),
                # nn.Conv2d(nb_kernel2, nb_kernel3, k_size3, stride=k_stride3),
            ]
        )
        #input_size = int((input_width - k_size1) / k_stride1) + 1
        #input_size = int((input_size - k_size2) / k_stride2) + 1
        input_size = int((input_width - k_size1) / k_stride1) + 1
        input_size = input_size ** 2 * nb_kernel1

        fc_layer = [
            nn.Linear(input_size + len(linear_features), dense_units[0])
        ]
        for i in range(1, len(dense_units)):
            fc_layer.append(nn.Linear(dense_units[i - 1], dense_units[i]))
        self.fc_layers = nn.ModuleList(fc_layer)

        self.output_layer = pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(dense_units[-1], n_actions), SoftmaxCategoricalHead(),),
            nn.Linear(dense_units[-1], 1),
        )
        torch.manual_seed(42)

    def forward(self, x):
        # split gasf, linear_features
        h, linear = x[0], x[1]
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Flatten
        h = nn.Flatten()(h)
        fc_in = torch.cat([h, linear], dim=1)

        for l in self.fc_layers:
            fc_in = self.activation(l(fc_in))

        return self.output_layer(fc_in)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=5 * 10 ** 6)

    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load-pretrained",
                        action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
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
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    # Load GASF
    df = pd.read_csv('./../M30_201001-201912_Tech7.csv', parse_dates=[0])

    train_df = df[((df['Datetime'] >= dt.datetime(2016, 1, 1))
                   & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
    valid_df = df[((df['Datetime'] >= dt.datetime(2018, 1, 1))
                   & (df['Datetime'] < dt.datetime(2019, 1, 1)))]

    #gasf_cols = ['Open', 'High', 'Low', 'Close'] + gasf_techs
    train_gasf = pickle.load(open('M30_2016-2018_12candle.gasf2', 'rb'))
    valid_gasf = pickle.load(open('M30_2018-2019_12candle.gasf2', 'rb'))
    train_gasf = processing.nwhc2nchw_array(train_gasf)
    valid_gasf = processing.nwhc2nchw_array(valid_gasf)
    train_gasf = train_gasf.astype(np.float32)
    valid_gasf = valid_gasf.astype(np.float32)

    def make_env(idx, test):
        '''
        if test:
            return fx_env_gasf.FxEnv_GASF(valid_df, valid_gasf, mode=fx_env_gasf.FxEnv_GASF.TEST_MODE)
        else:
            return fx_env_gasf.FxEnv_GASF(train_df, train_gasf, mode=fx_env_gasf.FxEnv_GASF.TRAIN_MODE)
        '''
        if test:
            return fx_env_gasf_concat.FxEnv_GASF(valid_df, valid_gasf, mode=fx_env_gasf_concat.FxEnv_GASF.TEST_MODE)
        else:
            return fx_env_gasf_concat.FxEnv_GASF(train_df, train_gasf, mode=fx_env_gasf_concat.FxEnv_GASF.TRAIN_MODE)

    env = make_env(0, False)
    obs_shape = env.observation_space.low.shape
    n_actions = env.action_space.n
    obs_ch = obs_shape[1]
    obs_width = obs_shape[2]

    model = A3C_Q_Func(n_actions, obs_width, obs_ch)
    # SharedRMSprop is same as torch.optim.RMSprop except that it initializes
    # its state in __init__, allowing it to be moved to shared memory.
    opt = SharedRMSpropEpsInsideSqrt(
        model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99)
    assert opt.state_dict()["state"], (
        "To share optimizer state across processes, the state must be"
        " initialized before training."
    )

    '''
    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255
    '''

    agent = a3c.A3C(
        model,
        opt,
        t_max=args.t_max,
        gamma=0.99,
        beta=args.beta,
        # phi=phi,
        max_grad_norm=40.0,
    )

    if args.load_pretrained:
        raise Exception("Pretrained models are currently unsupported.")

    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
        )
        print(
            "n_steps: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_steps,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:

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
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            use_tensorboard=True,
            eval_interval=args.eval_interval,
            global_step_hooks=[lr_decay_hook],
            save_best_so_far_agent=True,
        )


if __name__ == "__main__":
    main()
