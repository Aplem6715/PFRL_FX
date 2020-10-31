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
from tqdm import tqdm
from sklearn import preprocessing

import random
from pfrl.action_value import DiscreteActionValue

import processing

linear_features = ['position', 'pips']
gasf_techs = ['SMA5', 'SMA25', 'MACD', 'MACD_SI', 'RSI14']

nb_kernel1 = 4
#nb_kernel2 = 16
k_size1 = 5
#k_size2 = 2
k_stride1 = 1
#k_stride2 = 1
dense_units = [62, 32]


class Q_Func(torch.nn.Module):
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
        fc_layer.append(nn.Linear(dense_units[-1], n_actions))

        self.fc_layers = nn.ModuleList(fc_layer)
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

        return DiscreteActionValue(fc_in)


df = pd.read_csv('./M30_201001-201912_Tech7.csv', parse_dates=[0])

train_df = df[((df['Datetime'] >= dt.datetime(2018, 1, 1))
               & (df['Datetime'] < dt.datetime(2019, 1, 1)))]
valid_df = df[((df['Datetime'] >= dt.datetime(2016, 1, 1))
               & (df['Datetime'] < dt.datetime(2018, 1, 1)))]


gasf_cols = ['Open', 'High', 'Low', 'Close'] + gasf_techs
#gasf = processing.get_culr_tech_gasf(train_df.loc[:, gasf_cols], 12)
#pickle.dump(gasf, open('M30_2016-2018_12candle.gasf2', 'wb'))
#gasf = processing.get_culr_tech_gasf(valid_df.loc[:, gasf_cols], 12)
#pickle.dump(gasf, open('M30_2018-2019_12candle.gasf2', 'wb'))


train_gasf = pickle.load(open('M30_2018-2019_12candle.gasf2', 'rb'))
valid_gasf = pickle.load(open('M30_2016-2018_12candle.gasf2', 'rb'))
train_gasf = processing.nwhc2nchw_array(train_gasf)
valid_gasf = processing.nwhc2nchw_array(valid_gasf)
train_gasf = train_gasf.astype(np.float32)
valid_gasf = valid_gasf.astype(np.float32)

# 環境の生成
train_env = fx_env_gasf_concat.FxEnv_GASF(
    train_df, train_gasf, fx_env_gasf_concat.FxEnv_GASF.TRAIN_MODE)
valid_env = fx_env_gasf_concat.FxEnv_GASF(
    valid_df, valid_gasf, fx_env_gasf_concat.FxEnv_GASF.TEST_MODE)

# Q関数の定義
obs_shape = train_env.observation_space.low.shape
n_actions = train_env.action_space.n
obs_ch = obs_shape[1]
obs_width = obs_shape[2]

'''
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
'''
q_func = Q_Func(n_actions, input_width=obs_width, n_input_channels=obs_ch)

n_episodes = 500  # エピソード数

# エージェントの生成
agent = pfrl.agents.DoubleDQN(
    q_func,  # Q関数
    optimizer=torch.optim.Adam(
        q_func.parameters(), lr=0.0001),  # オプティマイザ
    replay_buffer=pfrl.replay_buffers.ReplayBuffer(
        capacity=8 * 10 ** 4),  # リプレイバッファ 8GB
    gamma=0.9,  # 将来の報酬割引率
    explorer=pfrl.explorers.LinearDecayEpsilonGreedy(  # 探索(ε-greedy)
        start_epsilon=0.33, end_epsilon=0.001, decay_steps=(n_episodes*1/4)*len(train_df), random_action_func=lambda: random.choice([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3])),
    replay_start_size=10000,  # リプレイ開始サイズ
    update_interval=10,  # 更新インターバル
    target_update_interval=1000,  # ターゲット更新インターバル
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
        acts = []
        steps = 0
        R = 0  # エピソード報酬
        #bar = tqdm(total=len(train_df))

        # ステップの反復
        while True:
            # bar.update(1)
            steps += 1
            # 環境の描画
            train_env.render()

            # 行動の推論
            action = agent.act(obs)
            acts.append(action)

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
            print('episode:', i, '\tR:{:.1f}\tnb_trade:{}\tnb_stay:{}\tnb_buy:{}\tnb_sell:{}\tnb_close:{}\tmeanR:{:.3f}\tminR:{:.3f}\tmaxR:{:.3f}\tbalance:{:.1f}              '
                  .format(
                      R,
                      len(train_env.broker.long_hists) +
                      len(train_env.broker.short_hists),
                      acts.count(0),
                      acts.count(1),
                      acts.count(2),
                      acts.count(3),
                      R / steps,
                      min(rewards), max(rewards),
                      train_env.broker.balance
                  )
                  )
        if i % 1 == 0:
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
                # 最大スコアなら保存
                if R > max_score:
                    max_score = R
                    agent.save(
                        'backup_double/agent_double_best_{}'.format(int(max_score)))
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
