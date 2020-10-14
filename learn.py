# パッケージのインポート
import pfrl
import torch
import torch.nn
import gym
import numpy

# 環境の生成
env = gym.make('CartPole-v0')


# Q関数の定義
obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, n_actions),
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
        epsilon=0.3, random_action_func=env.action_space.sample),
    replay_start_size=500,  # リプレイ開始サイズ
    update_interval=1,  # 更新インターバル
    target_update_interval=100,  # ターゲット更新インターバル
    phi=lambda x: x.astype(numpy.float32, copy=False),  # 特徴抽出関数
    gpu=-1,  # GPUのデバイスID（-1:CPU）
)

# エージェントの学習
n_episodes = 500  # エピソード数
max_episode_len = 200  # 最大エピソード長

# エピソードの反復
for i in range(1, n_episodes + 1):
    # 環境のリセット
    obs = env.reset()
    R = 0  # エピソード報酬
    t = 0  # ステップ

    # ステップの反復
    while True:
        # 環境の描画
        # env.render()

        # 行動の推論
        action = agent.act(obs)

        # 環境の1ステップ実行
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)

        # エピソード完了
        if done or reset:
            break

    # ログ出力
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')


# エージェントのテスト
with agent.eval_mode():

    # エピソードの反復
    for i in range(10):
        # 環境のリセット
        obs = env.reset()
        R = 0  # エピソード報酬
        t = 0  # ステップ

        # ステップの反復
        while True:
            # 環境の描画
            env.render()

            # 環境の1ステップ実行
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t == 200
            agent.observe(obs, r, done, reset)

            # エピソード完了
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)
