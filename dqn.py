import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2
import copy
import imageio
import os

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用デバイス: {DEVICE}")

# 動画保存フォルダ
os.makedirs('videos/run2', exist_ok=True)

# 環境構築
STAGE_NAME = 'SuperMarioBros-1-1-v0'
env = gym_super_mario_bros.make(STAGE_NAME)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print("環境構築成功")

def preprocess(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MarioAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99995
        self.batch_size = 32
        self.update_freq = 1000

        self.model = DQN(action_size).to(DEVICE)
        self.target_model = DQN(action_size).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
        self.step_count = 0

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

# 学習設定
EPISODES = 1000
action_size = env.action_space.n
agent = MarioAgent(action_size)

# 画面表示
plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#一時停止機能
paused = False
def on_key(event):
    global paused
    if event.key == ' ':
        paused = not paused
        print('⏸️ 一時停止' if paused else '▶️ 再開')
fig.canvas.mpl_connect('key_press_event', on_key)

reward_history = []
best_reward = 0
frames_best = []
all_frames_chunk = []

def save_video(frames, filename):
    print(f"動画保存中: {filename}")
    imageio.mimsave(filename,
        [np.array(img) for i, img in enumerate(frames) if i % 4 == 0],
        fps=15)
    print(f"✅ 保存完了: {filename}")

for episode in range(EPISODES):
    state = env.reset()
    state = preprocess(state)
    state_stack = np.stack([state] * 4, axis=0)
    done = False
    total_reward = 0
    step = 0
    frames = []
    agent.prev_x_pos = 0
    prev_score = 0
    
    while not done:
        # 一時停止
        while paused:
            plt.pause(0.1)
            
        action = agent.get_action(state_stack)
        
        result = env.step(action)
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result

        if hasattr(agent, 'prev_x_pos'):
            x_diff = info['x_pos'] - agent.prev_x_pos
            if x_diff > 0:
                reward += x_diff * 0.5
            elif x_diff < 0:
                reward -= 1
        agent.prev_x_pos = info['x_pos']
        
        if done and not info['flag_get'] and info.get('y_pos', 0) > 200:
            reward -= 50
        
        if done and not info['flag_get']:
            reward -= 30
            
        if info['flag_get']:
            reward += 1500
        
        next_state = preprocess(next_state)
        next_state_stack = np.roll(state_stack, -1, axis=0)
        next_state_stack[-1] = next_state

        agent.remember(state_stack, action, reward, next_state_stack, done)
        agent.learn()

        state_stack = next_state_stack
        total_reward += reward
        step += 1

        frames.append(copy.deepcopy(env.render(mode='rgb_array')))

        if step % 10 == 0:
            axes[0].clear()
            axes[0].imshow(env.render(mode='rgb_array'))
            axes[0].axis('off')
            axes[0].set_title(f'Episode: {episode+1} | Step: {step} | Reward: {total_reward:.1f}')

            axes[1].clear()
            axes[1].plot(reward_history)
            axes[1].set_title('Reward History')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Reward')

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
        
    reward_history.append(total_reward)
    all_frames_chunk.extend(frames)

    # ベスト更新
    if total_reward > best_reward:
        best_reward = total_reward
        frames_best = copy.deepcopy(frames)
        save_video(frames_best, f'videos/run2/best_ep{episode+1}_reward{int(total_reward)}.mp4')

    # 50エピソードごとにまとめて保存
    if (episode + 1) % 50 == 0:
        start_ep = episode - 48
        end_ep = episode + 1
        save_video(all_frames_chunk, f'videos/run2/episode_{start_ep}~{end_ep}.mp4')
        all_frames_chunk = []
        print(f"✅ Episode {start_ep}~{end_ep} 保存完了")

    print(f'Episode: {episode+1} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f} | Steps: {step}')

    if (episode + 1) % 100 == 0:
        torch.save(agent.model.state_dict(), f'mario_dqn_ep{episode+1}.pth')
        print(f"✅ モデル保存: mario_dqn_ep{episode+1}.pth")

    if info['flag_get']:
        print("<<< Mario get the flag. GOOOOOOOOOOOOOOOOOOOOOAL! >>>")
        save_video(frames, f'videos/run2/CLEAR_ep{episode+1}.mp4')
        axes[0].clear()
        axes[0].text(0.5, 0.5, '🎉 GOAL!! 🎉', fontsize=50, color='gold',
                ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_facecolor('black')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(5)
        break

env.close()
plt.ioff()
plt.show()