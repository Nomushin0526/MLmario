import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
from PIL import Image
import base64
import json
import time
import copy
import os
import threading
import requests as req
from dotenv import load_dotenv

# APIキーの読み込み（Gemini用、今は使わない）
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv('GEMINI_API_KEY')
print(f"APIキー確認: {api_key[:5] if api_key else 'なし'}")

# Ollama設定
OLLAMA_URL = "http://localhost:11434/api/generate"

# 環境構築
STAGE_NAME = 'SuperMarioBros-1-1-v0'
env = gym_super_mario_bros.make(STAGE_NAME)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print("環境構築成功")

# プロンプト
prompt = """
この画像はゲーム、スーパーマリオのプレイ画面です。
画面に応じて、以下の7つのボタン操作ができます。
NOOPが操作しない。Aがジャンプ。Bがダッシュです。
0 = 'NOOP'
1 = 'right'
2 = 'right', 'A'
3 = 'right', 'B'
4 = 'right', 'A', 'B'
5 = 'A'
6 = 'left'
以下のJSON形式のみで出力してください。日本語でお願いします。
マークダウンやコードブロックは使わず、JSONのみ出力してください。
{"explanation": "画面の説明", "reason": "ボタン操作の理由", "action": ボタン操作の番号}
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def predict(state):
    image = Image.fromarray(state)
    image = image.resize((128, 120))
    image.save('state.png')
    base64_image = encode_image('state.png')

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "format": "json"
    }

    try:
        response = req.post(OLLAMA_URL, json=payload)
        content_dict = json.loads(response.json().get('response', '{}'))
        action = content_dict.get('action', 1)
        explanation = content_dict.get('explanation', '')
        reason = content_dict.get('reason', '')
    except Exception as e:
        print(f"エラー: {e}")
        action = 1
        explanation = ''
        reason = ''

    if action is None:
        action = 0

    try:
        action = int(action)
    except:
        action = 1

    return action, explanation, reason

# メイン設定
EPISODE_NUMBERS = 10
MAX_TIMESTEP_TEST = 2000
SKIP_RATE = 30

total_reward = [0] * EPISODE_NUMBERS
total_time = [0] * EPISODE_NUMBERS
best_reward = 0
frames_best = []

# スレッド用変数
current_action = 1
ai_running = False

def ai_predict(state):
    global current_action, ai_running
    ai_running = True
    action, explanation, reason = predict(state)
    current_action = action
    print(f'action: {action}')
    print(f'explanation: {explanation}')
    print(f'reason: {reason}')
    ai_running = False

# 画面表示設定
plt.ion()
fig, ax = plt.subplots(figsize=(8, 7))

for i in range(EPISODE_NUMBERS):
    ax.clear()
    ax.text(0.5, 0.5, f'STAGE {i+1}', fontsize=50, color='white',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_facecolor('black')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1.5)
    
    state = env.reset()
    done = False
    total_reward[i] = 0.0
    total_time[i] = 0
    skip_numb = SKIP_RATE
    frames = []
    current_action = 1
    ai_running = False

    while not done and total_time[i] < MAX_TIMESTEP_TEST:
    # AIが空いていたら別スレッドで推論開始
        if skip_numb >= SKIP_RATE and not ai_running:
            skip_numb = 0
            t = threading.Thread(target=ai_predict, args=(state.copy(),))
            t.daemon = True  # ← これを追加！メイン終了時に自動で止まる
            t.start()
        else:
            skip_numb += 1

        # ゲームは常に進み続ける
        result = env.step(current_action)
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            state, reward, done, info = result

        total_reward[i] += reward
        total_time[i] += 1
        frames.append(copy.deepcopy(env.render(mode='rgb_array')))

        # 画面を常に更新
        ax.clear()
        ax.imshow(state)
        ax.axis('off')
        ax.set_title(f'Step: {total_time[i]} | Reward: {total_reward[i]:.1f} | Action: {current_action}')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    if total_reward[i] > best_reward:
        best_reward = total_reward[i]
        frames_best = copy.deepcopy(frames)

    print('test episode:', i, 'reward:', total_reward[i], 'time:', total_time[i])

    if info['flag_get']:
        print("<<< Mario get the flag. GOOOOOOOOOOOOOOOOOOOOOAL! >>>")
        break

print('average reward:', (sum(total_reward) / EPISODE_NUMBERS),
      'average time:', (sum(total_time) / EPISODE_NUMBERS),
      'best_reward:', best_reward)

env.close()
plt.ioff()
plt.show()