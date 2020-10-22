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
from sklearn import preprocessing


nb_kernel1 = 32
nb_kernel2 = 32
k_size1 = 4
k_size2 = 3
k_stride1 = 2
k_stride2 = 1

df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

scaler = preprocessing.MinMaxScaler()
scaler.fit(df.iloc[:, 1:])

train_df = df[((df['Datetime'] >= dt.datetime(2017, 12, 1))
               & (df['Datetime'] < dt.datetime(2018, 1, 1)))]
valid_df = df[((df['Datetime'] >= dt.datetime(2018, 1, 1))
               & (df['Datetime'] < dt.datetime(2019, 1, 1)))]
# 環境の生成
train_env = fx_env_gasf.FxEnv_GASF(
    train_df, fx_env_gasf.FxEnv_GASF.TRAIN_MODE)
valid_env = fx_env_gasf.FxEnv_GASF(
    valid_df, fx_env_gasf.FxEnv_GASF.TEST_MODE)

# Q関数の定義
obs_shape = train_env.observation_space.shape
n_actions = train_env.action_space.n
input_size = int((obs_shape[0] - k_size1) / k_stride1)
input_size = int((input_size - k_size2) / k_stride2)


q_func = torch.nn.Sequential(
    torch.nn.Conv2d(obs_shape[2], nb_kernel1, k_size1, stride=k_stride1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(nb_kernel1, nb_kernel2, k_size2, stride=k_stride2),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    #torch.nn.Conv2d(obs_size, 32, 8, stride=4),
    torch.nn.Linear(input_size*input_size*nb_kernel2, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    pfrl.q_functions.DiscreteActionValueHead(),
)


# エージェントの生成
agent = pfrl.agents.DoubleDQN(
    q_func,  # Q関数
    optimizer=torch.optim.Adam(
        q_func.parameters(), lr=0.0001),  # オプティマイザ
    replay_buffer=pfrl.replay_buffers.ReplayBuffer(
        capacity=10 ** 6),  # リプレイバッファ
    gamma=0.50,  # 将来の報酬割引率
    explorer=pfrl.explorers.ConstantEpsilonGreedy(  # 探索(ε-greedy)
        epsilon=0.3, random_action_func=train_env.action_space.sample),
    replay_start_size=1000,  # リプレイ開始サイズ
    update_interval=5,  # 更新インターバル
    target_update_interval=100,  # ターゲット更新インターバル
    phi=lambda x: x.astype(numpy.float32, copy=False),  # 特徴抽出関数
    gpu=0,  # GPUのデバイスID（-1:CPU）
)

# エージェントの学習
n_episodes = 100  # エピソード数


def train():
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

    agent.save('agent_double')


if __name__ == '__main__':
    cProfile.run('train()', filename='train.prof')
