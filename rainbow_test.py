import torch
import pfrl
import numpy as np
import pandas as pd
import datetime as dt
import fx_env2

from sklearn import preprocessing
from pfrl import agents
from pfrl import explorers
from pfrl import replay_buffers
from pfrl import nn as pnn
from pfrl.q_functions import DistributionalFCStateQFunctionWithDiscreteAction

import matplotlib.pyplot as plt


def make_agent(env):
    n_actions = env.action_space.n
    n_obs = env.observation_space.low.size

    n_atoms = 51
    v_max = 10
    v_min = -10
    # 128ユニット3層のDeep Q関数
    q_func = DistributionalFCStateQFunctionWithDiscreteAction(
        n_obs, n_actions, n_atoms, v_min, v_max, 128, 3)

    # Noisy nets
    pnn.to_factorized_noisy(q_func, sigma_scale=0.5)
    # Turn off explorer
    explorer = explorers.Greedy()

    # Use the same hyper parameters as https://arxiv.org/abs/1710.02298
    opt = torch.optim.Adam(q_func.parameters(), 6.25e-5, eps=1.5 * 10 ** -4)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 10
    betasteps = 5 * 10 ** 6 / update_interval
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
        gpu=0,
        gamma=0.99,
        explorer=explorer,
        minibatch_size=32,
        replay_start_size=10 ** 4,
        target_update_interval=3200,
        update_interval=update_interval,
        batch_accumulator="mean",
        phi=phi,
    )
    return agent


def main():
    df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df.iloc[:, 1:])

    test_df = df[((df['Datetime'] >= dt.datetime(2018, 1, 1))
                  & (df['Datetime'] < dt.datetime(2021, 1, 1)))]
    test_env = fx_env2.FxEnv(test_df, scaler)

    duration = test_df.iloc[-1]['Datetime'] - \
        test_df.iloc[0]['Datetime']  # type: dt.timedelta
    dur_year = duration.days/365

    agent = make_agent(test_env)
    agent.load('best_model_rainbow')

    date_list = []
    pl_list = []

    # エージェントのテスト
    with agent.eval_mode():
        obs = test_env.reset()
        R = 0  # エピソード報酬
        # ステップの反復
        while True:
            # 環境の描画
            test_env.render()

            # 環境の1ステップ実行
            action = agent.act(obs)
            obs, r, done, info = test_env.step(action)

            if r != 0:
                pl_list.append(r)
                date_list.append(info['datetime'])

            R += r
            agent.observe(obs, r, done, False)

            # エピソード完了
            if done:
                break
        year_pl = R / dur_year
        print('[', test_df.iloc[0]['Datetime'], ']  →  [',
              test_df.iloc[-1]['Datetime'], ']  : ', duration.days, 'days              ')
        print('balance:{:.2f}$,   Reward:{:.2f}$,   年利益: {:.1f}$,   年利率: {:.1f}%        '.format(
            test_env.account.balance, R, year_pl,
            (test_env.account.balance-test_env.init_balance)/test_env.init_balance/dur_year*100))

    print('nb_trades\t', len(pl_list))
    print('pl average\t{:.2f} $'.format(np.average(pl_list)))
    print('pl std\t\t{:.2f} $'.format(np.std(pl_list)))
    pl_ser = pd.Series(pl_list)
    pl_ser.hist(bins=100, range=(-6, 6))
    plt.xlabel('profit[$]')
    plt.ylabel('trade count')
    plt.show()


if __name__ == "__main__":
    main()
