import taichi as ti
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
import numpy as np

# Taichiの初期化
ti.init(arch=ti.cpu)

# 物理シミュレーションパラメータ
g = 9.8      # 重力加速度 (m/s^2)
l = 0.5      # ひもの長さの半分 (m)
dt = 0.02    # ステップ時間 (秒)
M = 1.0      # 台車の質量 (kg)
m = 0.1      # ボールの質量 (kg)

# Taichiのデータフィールド定義
theta = ti.field(dtype=ti.f32, shape=())  # ボールの角度 (ラジアン)
omega = ti.field(dtype=ti.f32, shape=())  # ボールの角速度
x = ti.field(dtype=ti.f32, shape=())      # 台車の位置
v = ti.field(dtype=ti.f32, shape=())      # 台車の速度
force = ti.field(dtype=ti.f32, shape=())  # 台車に加える力 (N)

@ti.kernel
def init():
    x[None] = 0.0
    v[None] = 0.0
    # 初期角度は小さなランダムな傾き (-0.05 から 0.05 ラジアン)
    theta[None] = (ti.random() - 0.5) * 0.1
    omega[None] = 0.0

@ti.kernel
def update():
    total_mass = M + m
    sin_theta = ti.sin(theta[None])
    cos_theta = ti.cos(theta[None])
    
    # 倒立振子の運動方程式の中間計算
    temp = (force[None] + m * l * omega[None]**2 * sin_theta) / total_mass
    
    # 正しい物理式の分母: l * (4/3 - m * cos^2(theta) / total_mass)
    denominator = l * (4.0 / 3.0 - m * cos_theta**2 / total_mass)
    
    # ポールの角加速度
    alpha = (g * sin_theta - temp * cos_theta) / denominator
    
    # 台車の加速度
    acc = temp - m * l * alpha * cos_theta / total_mass
    
    # オイラー法による状態の更新
    x[None] += v[None] * dt
    v[None] += acc * dt
    theta[None] += omega[None] * dt
    omega[None] += alpha * dt

# --- Deep Q-Network と Replay Buffer の実装 ---

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(state)),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(np.array(next_state)),
                torch.FloatTensor(done))
                
    def __len__(self):
        return len(self.buffer)

class CartPoleBrain(nn.Module):
    def __init__(self):
        super(CartPoleBrain, self).__init__()
        # 状態数 4 -> 隠れ層 64 -> 隠れ層 64 -> 行動数 2 (左 / 右)
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def soft_update(local_model, target_model, tau=0.005):
    """ターゲットネットワークのパラメータをソフト更新 (Target <- tau * Local + (1 - tau) * Target)"""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def update_brain(replay_buffer, brain, target_brain, optimizer, batch_size, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return 0.0
        
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # 現在のQ値の取得
    q_values = brain(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # ターゲットQ値の計算 (TD誤差ターゲット)
    with torch.no_grad():
        next_q_values = target_brain(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        expected_state_action_values = rewards + gamma * max_next_q_values * (1.0 - dones)
        
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# AIモデルおよびトレーニング用変数の初期化
brain = CartPoleBrain()
target_brain = CartPoleBrain()
target_brain.load_state_dict(brain.state_dict())

optimizer = optim.Adam(brain.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(capacity=20000)

epochs = 1000
batch_size = 64
gamma = 0.99

# Epsilon-Greedyパラメータ
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

best_survival = 0
last_loss = 0.0
fast_forward = False  # True にすると描画をスキップして高速学習

# GUIの初期化
gui = ti.GUI("CartPole DQN Training", res=(800, 400))

# メイントレーニングループ
for episode in range(epochs):
    if not gui.running:
        break
        
    init()
    state = [x[None], v[None], theta[None], omega[None]]
    total_reward = 0
    step_count = 0
    
    while gui.running:
        step_count += 1
        
        # GUIイベント処理 (高速化キー F の切り替え、および終了処理)
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'f' or gui.event.key == 'F':
                fast_forward = not fast_forward
                print(f"Fast-forward status: {fast_forward}")
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
                break
                
        # 1. 状態の認識と行動決定
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = brain(state_tensor)
            
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            action = torch.argmax(q_values).item()
            
        # 2. 物理シミュレーションステップの実行
        # 力の大きさは標準的な 10.0 N に設定
        if action == 0:
            force[None] = -10.0
        else:
            force[None] = 10.0
            
        update()
        
        # 3. 終了条件および報酬の判定
        # 範囲制限: x は [-2.4, 2.4]、角度 theta は [-0.3, 0.3] ラジアン (約17度)
        cart_pos = x[None]
        pole_ang = theta[None]
        done = cart_pos < -2.4 or cart_pos > 2.4 or abs(pole_ang) > 0.3
        
        if not done:
            reward = 1.0
        else:
            reward = -10.0  # 倒れた場合のペナルティを大きくして学習を加速
            
        next_state = [x[None], v[None], theta[None], omega[None]]
        
        # 遷移をリプレイバッファに保存し、学習を実行
        replay_buffer.push(state, action, reward, next_state, float(done))
        
        if len(replay_buffer) >= batch_size:
            last_loss = update_brain(replay_buffer, brain, target_brain, optimizer, batch_size, gamma)
            soft_update(brain, target_brain, tau=0.005)
            
        state = next_state
        total_reward += reward
        
        # 描画処理 (高速化モードでない場合のみ)
        if not fast_forward:
            gui.clear(0x222222)
            
            # 物理座標 x [-2.4, 2.4] を GUI の描画座標範囲 [0.1, 0.9] にマッピング
            render_x = 0.5 + (cart_pos / 2.4) * 0.4
            render_y = 0.2
            
            # 地面と左右の限界線の描画
            gui.line(begin=(0.1, render_y), end=(0.9, render_y), radius=2, color=0x888888)
            gui.line(begin=(0.1, render_y - 0.02), end=(0.1, render_y + 0.02), radius=2, color=0x888888)
            gui.line(begin=(0.9, render_y - 0.02), end=(0.9, render_y + 0.02), radius=2, color=0x888888)
            
            # 台車 (ライトグレーの四角形)
            gui.line(begin=(render_x - 0.06, render_y), end=(render_x + 0.06, render_y), radius=15, color=0xAAAAAA)
            
            # 振り子 (白線)
            draw_L = 0.35
            tip_x = render_x + draw_L * math.sin(pole_ang)
            tip_y = render_y + draw_L * math.cos(pole_ang)
            gui.line(begin=(render_x, render_y), end=(tip_x, tip_y), radius=4, color=0xFFFFFF)
            
            # 支点 (緑の円)
            gui.circle(pos=(render_x, render_y), radius=6, color=0x00FF00)
            
            # テキストによる統計情報の描画
            gui.text(f"Episode: {episode + 1}/{epochs}", pos=(0.05, 0.92), color=0xFFFFFF, font_size=16)
            gui.text(f"Survival Steps: {step_count}", pos=(0.05, 0.86), color=0xFFFFFF, font_size=16)
            gui.text(f"Best Survival: {best_survival}", pos=(0.05, 0.80), color=0xFFFFFF, font_size=16)
            gui.text(f"Epsilon (Explore): {epsilon:.3f}", pos=(0.05, 0.74), color=0xFFFFFF, font_size=16)
            gui.text(f"Loss: {last_loss:.5f}", pos=(0.05, 0.68), color=0xFFFFFF, font_size=16)
            gui.text("Press [F] to Toggle Fast-Forward (Train Faster)", pos=(0.05, 0.62), color=0x00FF00, font_size=14)
            
            gui.show()
            # 描画を滑らかにするためのスリープ
            time.sleep(0.01)
        else:
            # 高速化モード時は、ウインドウフリーズ防止のため10ステップごとにGUIイベントを取得
            if step_count % 10 == 0:
                gui.get_event()
                
        if done or step_count >= 500:  # 500ステップ（安定状態）に達したらそのエピソードを終了
            break
            
    # エピソード終了時に探索率 epsilon を減衰
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    if step_count > best_survival:
        best_survival = step_count
        
    print(f"Episode {episode + 1:3d} finished | Survival steps: {step_count:3d} | Epsilon: {epsilon:.3f} | Loss: {last_loss:.5f}")
    
    # ユーザーがウインドウを閉じた場合は終了
    if not gui.running:
        break

print("Training finished.")