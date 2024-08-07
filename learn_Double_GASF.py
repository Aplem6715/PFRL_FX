# パッケージのインポート
import pfrl
import torch
import torch.nn
import gym
import numpy
import fx_env_gasf
import pandas as pd
import datetime as dt
import cProfile
import pprint
import pickle
from sklearn import preprocessing

import processing

nb_kernel1 = 8
nb_kernel2 = 8
k_size1 = 4
k_size2 = 3
k_stride1 = 1
k_stride2 = 1
dense_units = 32

df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

train_df = df[((df['Datetime'] >= dt.datetime(2010, 1, 1))
               & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
valid_df = df[((df['Datetime'] >= dt.datetime(2018, 1, 1))
               & (df['Datetime'] < dt.datetime(2019, 1, 1)))]

gasf = processing.get_ohlc_culr_gasf(train_df.loc[:, 'Open': 'Close'], 16)
pickle.dump(gasf, open('M30_2010-2018_16candle.gasf', 'wb'))
gasf = processing.get_ohlc_culr_gasf(valid_df.loc[:, 'Open': 'Close'], 16)
pickle.dump(gasf, open('M30_2018-2019_16candle.gasf', 'wb'))

train_gasf = pickle.load(open('M30_2014-2018.gasf', 'rb'))
valid_gasf = pickle.load(open('M30_2018-2019.gasf', 'rb'))
train_gasf = processing.nwhc2nchw_array(train_gasf)
valid_gasf = processing.nwhc2nchw_array(valid_gasf)

# 環境の生成
train_env = fx_env_gasf.FxEnv_GASF(
    train_df, train_gasf, fx_env_gasf.FxEnv_GASF.TRAIN_MODE)
valid_env = fx_env_gasf.FxEnv_GASF(
    valid_df, valid_gasf, fx_env_gasf.FxEnv_GASF.TEST_MODE)

# Q関数の定義
obs_shape = train_env.observation_space.low.shape
n_actions = train_env.action_space.n
obs_ch = obs_shape[0]
obs_width = obs_shape[1]
input_size = int((obs_width - k_size1) / k_stride1) + 1
input_size = int((input_size - k_size2) / k_stride2) + 1


q_func = torch.nn.Sequential(
    torch.nn.Conv2d(obs_ch, nb_kernel1, k_size1, stride=k_stride1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(nb_kernel1, nb_kernel2, k_size2, stride=k_stride2),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    #torch.nn.Conv2d(obs_size, 32, 8, stride=4),
    torch.nn.Linear(input_size*input_size*nb_kernel2, dense_units),
    torch.nn.ReLU(),
    torch.nn.Linear(dense_units, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)

n_episodes = 50  # エピソード数

# エージェントの生成
agent = pfrl.agents.DoubleDQN(
    q_func,  # Q関数
    optimizer=torch.optim.Adam(
        q_func.parameters(), lr=0.0001),  # オプティマイザ
    replay_buffer=pfrl.replay_buffers.ReplayBuffer(
        capacity=8 * 10 ** 4),  # リプレイバッファ 8GB
    gamma=0.9,  # 将来の報酬割引率
    explorer=pfrl.explorers.LinearDecayEpsilonGreedy(  # 探索(ε-greedy)
        start_epsilon=0.5, end_epsilon=0.0, decay_steps=(n_episodes-5)*len(train_df), random_action_func=train_env.action_space.sample),
    replay_start_size=10000,  # リプレイ開始サイズ
    update_interval=5,  # 更新インターバル
    target_update_interval=100,  # ターゲット更新インターバル
    phi=lambda x: x.astype(numpy.float32, copy=False),  # 特徴抽出関数
    gpu=0,  # GPUのデバイスID（-1:CPU）
)

# エージェントの学習


def train():
    max_score = -1000000
    # エピソードの反復
    for i in range(1, n_episodes + 1):
        # 環境のリセット
        obs = train_env.reset()
        rewards = []
        steps = 0
        R = 0  # エピソード報酬

        # ステップの反復
        while True:
            steps += 1
            # 環境の描画
            train_env.render()

            # 行動の推論
            action = agent.act(obs)

            # 環境の1ステップ実行
            obs, reward, done, _ = train_env.step(action)
            rewards.append(reward)
            R += reward
            agent.observe(obs, reward, done, False)

            # エピソード完了
            if done:
                break

        # ログ出力
        if i % 1 == 0 and i != 0:
            print('episode:', i, '\tR:{:.1f}\tnb_trade:{}\tmeanR:{:.3f}\tminR:{:.3f}\tmaxR:{:.3f}\tbalance:{:.1f}                                           '
                  .format(
                      R,
                      len(train_env.broker.long_hists) +
                      len(train_env.broker.short_hists),
                      R / steps,
                      min(rewards), max(rewards),
                      train_env.broker.balance
                  )
                  )
        if i % 5 == 0:
            # エージェントのテスト
            with agent.eval_mode():
                # 環境のリセット
                obs = valid_env.reset()
                rewards = []
                R = 0  # エピソード報酬

                # ステップの反復
                while True:
                    # 環境の描画
                    valid_env.render()

                    # 環境の1ステップ実行
                    action = agent.act(obs)
                    obs, r, done, _ = valid_env.step(action)
                    rewards.append(r)
                    steps += 1
                    R += r
                    agent.observe(obs, r, done, False)

                    # エピソード完了
                    if done:
                        break
                if R > max_score:
                    max_score = R
                    agent.save('agent_double_best{}'.format(int(max_score)))
                print('R:{:.1f}\tnb_trade:{}\tmeanR:{:.3f}\tminR:{:.3f}\tmaxR:{:.3f}\tbalance:{:.1f}                                           '
                      .format(
                          R,
                          len(valid_env.broker.long_hists) +
                          len(valid_env.broker.short_hists),
                          R / steps,
                          min(rewards), max(rewards),
                          valid_env.broker.balance
                      )
                      )
    print('Finished.')

    agent.save('agent_double_last{}'.format(int(max_score)))


if __name__ == '__main__':
    train()
