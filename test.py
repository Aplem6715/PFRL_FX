import pfrl
import torch
import torch.nn
import gym
import numpy
import fx_env
import pandas as pd
import datetime as dt
from sklearn import preprocessing

df = pd.read_csv('M30_201001-201912_Tech7.csv', parse_dates=[0])

scaler = preprocessing.MinMaxScaler()
scaler.fit(df.iloc[:, 1:])

valid_df = df[((df['Datetime'] >= dt.datetime(2015, 1, 1))
               & (df['Datetime'] < dt.datetime(2017, 1, 1)))]

valid_env = fx_env.FxEnv(valid_df, scaler)

# Q関数の定義
obs_size = valid_env.observation_space.low.size
n_actions = valid_env.action_space.n
q_func = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)
# エージェントの生成
agent = pfrl.agents.DoubleDQN(
    q_func,  # Q関数
    optimizer=torch.optim.Adam(q_func.parameters(), eps=1e-2),  # オプティマイザ
    replay_buffer=pfrl.replay_buffers.ReplayBuffer(
        capacity=10 ** 6),  # リプレイバッファ
    gamma=0.9,  # 将来の報酬割引率
    explorer=pfrl.explorers.ConstantEpsilonGreedy(  # 探索(ε-greedy)
        epsilon=0.3, random_action_func=valid_env.action_space.sample),
    replay_start_size=500,  # リプレイ開始サイズ
    update_interval=1,  # 更新インターバル
    target_update_interval=100,  # ターゲット更新インターバル
    phi=lambda x: x.astype(numpy.float32, copy=False),  # 特徴抽出関数
    gpu=0,  # GPUのデバイスID（-1:CPU）
)
agent.load('agent_double')

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
    print('R:{:.3f}'.format(R),
          '                                                              ')
