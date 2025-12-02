# 変更前
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import time
import math
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Webotsコントローラライブラリ
from controller import Supervisor, Motor, GPS, InertialUnit, PositionSensor

# --- グローバルパラメータ ---
TIME_STEP = 128
# 各関節の初期姿勢からの可動域（度）
maxAngle1 = 20
minAngle1 = -20
# 1ステップで動かす角度（度）
stepAngle = 10


# --- prototype_4.py に倣った構造 ---
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """経験を保存・サンプリングするためのリプレイバッファ"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Q値を予測するためのニューラルネットワークモデル"""
    def __init__(self, state_size, output_size, hidden_size=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 1. 環境 (Environment) ---
class HexapodEnv:
    """Webotsシミュレータと連携するヘキサポッド環境クラス"""
    def __init__(self, robot, num_joints=18):
        self.robot = robot
        self.num_joints = num_joints
        
        # 安定した初期姿勢（prototype_4.py参考）
        self.initial_motor_positions = [-45, 70, -110, 45, 70, -110, -45, -70, 110, 45, -70, 110, 0, -70, 110, 0, 70, -110]
        self.start_position = [0, 0.0, -0.07]
        self.start_rotation = [1, 0, 0, 1.57079632678966]
        self.goal_y_position = 0.05 # Y方向に5m進むのが目標
        self.goal_count = 0

        self._setup_robot()
        
        self.previous_pos = np.array(self.gps.getValues())
        self.previous_time = self.robot.getTime()

        self.state_size = len(self.get_state())
        self.action_options = [-1, 0, 1]
        
        self.steps = 0

    def _setup_robot(self):
        """Webotsのノードやセンサーを初期化"""
        self.robot_node = self.robot.getFromDef("IRSL-XR06-01")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        self.motors = [self.robot.getMotor(f'motor{i+1}') for i in range(self.num_joints)]
        self.position_sensors = []
        for i in range(self.num_joints):
            ps = self.robot.getPositionSensor(f'PS{i+1}')
            ps.enable(TIME_STEP)
            self.position_sensors.append(ps)
        
        self.gps = self.robot.getGPS('gps1')
        self.gps.enable(TIME_STEP)
        self.imu = self.robot.getInertialUnit('IU')
        self.imu.enable(TIME_STEP)

    def get_state(self):
        """センサーから現在の状態を取得し、ベクトルとして返す"""
        current_pos = np.array(self.gps.getValues())
        imu_values = self.imu.getRollPitchYaw()
        motor_positions = [math.degrees(ps.getValue()) for ps in self.position_sensors]
        
        current_time = self.robot.getTime()
        dt = current_time - self.previous_time
        velocity = (current_pos - self.previous_pos) / dt if dt > 0 else np.zeros(3)

        relative_goal_pos_y = self.goal_y_position - current_pos[1]

        state = [
            imu_values[0] / math.pi, imu_values[1] / math.pi, # Roll, Pitch正規化
            velocity[0], velocity[1], velocity[2],            # 速度
            *[mp / 180.0 for mp in motor_positions],         # 関節角度正規化
            relative_goal_pos_y / 5.0                        # ゴールまでの距離正規化
        ]
        return np.array(state, dtype=np.float32)

    def execute_action(self, action_vector):
        """18次元の行動ベクトルをモーター指令に変換"""
        current_angles = [math.degrees(ps.getValue()) for ps in self.position_sensors]
        for i, action_val in enumerate(action_vector):
            target_angle = current_angles[i] + action_val * stepAngle
            
            # 可動域制限
            init_pos = self.initial_motor_positions[i]
            clamped_target = max(init_pos + minAngle1, min(target_angle, init_pos + maxAngle1))
            
            self.motors[i].setPosition(math.radians(clamped_target))

    def calculate_reward(self):
        """報酬を計算"""
        current_pos = np.array(self.gps.getValues())
        current_time = self.robot.getTime()
        imu_values = self.imu.getRollPitchYaw()
        dt = current_time - self.previous_time

        velocity_y = (current_pos[1] - self.previous_pos[1]) / dt if dt > 0 else 0
        velocity_x = (current_pos[0] - self.previous_pos[0]) / dt if dt > 0 else 0
        print(f"Velocity Y: {abs(velocity_y):.4f} m/s")
        velocity_y *= 20 
        velocity_x *= 20 #速度の正規化

        reward = 0.0

        reward += min(velocity_y, 4.0)
        reward -= 0.005 * (velocity_x ** 2 + velocity_z ** 2)
        reward -= 0.05 * current_pos[2] ** 2 # 高さペナルティ
        reward -= 0.02

        if current_pos[1] >= self.goal_y_position:
            reward += 50.0
            
        if abs(imu_values[0]) > math.pi / 2.5 or abs(imu_values[1]) > math.pi / 2.5:
            reward -= 1000
        
        return reward

    def is_done(self):
        """エピソードの終了条件を判定"""
        current_pos = self.gps.getValues()
        imu_values = self.imu.getRollPitchYaw()
        
        if current_pos[1] >= self.goal_y_position:
            self.goal_count += 1
            if self.goal_count % 50 == 0:
                self.goal_y_position += 0.3 # ゴールを更新
            return True, "Goal reached!"
        if abs(imu_values[0]) > math.pi / 2.5 or abs(imu_values[1]) > math.pi / 2.5:
            return True, "Robot fell over!"
        if self.steps >= 30000: # 最大ステップ数
            return True, "Max steps reached!"
        return False, ""

    def step(self, action_vector):
        """環境を1ステップ進める"""
        self.steps += 1
        self.previous_pos = np.array(self.gps.getValues())
        self.previous_time = self.robot.getTime()

        self.execute_action(action_vector)
        
        # シミュレータを1ステップ進める
        if self.robot.step(TIME_STEP) == -1:
            # シミュレーションが停止した場合
            return self.get_state(), -100, True, "Simulation stopped"
        
        next_state = self.get_state()
        reward = self.calculate_reward()
        done, comment = self.is_done()
        
        return next_state, reward, done, comment

    def reset(self):
        """ロボットを初期状態に戻す"""
        self.translation_field.setSFVec3f(self.start_position)
        self.rotation_field.setSFRotation(self.start_rotation)
        self.robot_node.resetPhysics()
        
        for i, angle in enumerate(self.initial_motor_positions):
            self.motors[i].setPosition(math.radians(angle))

        self.robot.step(TIME_STEP * 10) # 姿勢が安定するまで待つ
        self.steps = 0
        self.previous_pos = np.array(self.gps.getValues())
        self.previous_time = self.robot.getTime()
        
        return self.get_state()

# --- 2. エージェント (Agent) ---
class DQNAgent:
    """多次元離散行動空間(18x3)を扱うDQNエージェント (PyTorch版) - ボルツマン探索"""
    def __init__(self, state_size, num_joints, action_dims, config):
        self.state_size = state_size
        self.num_joints = num_joints
        self.action_dims = action_dims
        self.output_size = self.num_joints * self.action_dims

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = config.get('learning_rate', 1e-4)
        self.gamma = config.get('gamma', 0.99)
        
        # --- ボルツマン探索用のパラメータ（報酬連動型） ---
        self.temperature = config.get('temp_start', 1.0)
        self.temp_min = config.get('temp_min', 0.01)
        self.temp_max = config.get('temp_start', 1.0) # 温度の最大値
        self.temp_update_rate = config.get('temp_update_rate', 0.0001) # 温度の更新レート
        # ---------------------------------

        self.batch_size = config.get('batch_size', 128)
        self.target_update = config.get('target_update', 10)
        
        self.q_network = DQN(state_size, self.output_size).to(self.device)
        self.target_network = DQN(state_size, self.output_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(config.get('memory_size', 50000))
        self.steps_done = 0
        
        self.action_map = {-1: 0, 0: 1, 1: 2}
        self.index_to_action = {v: k for k, v in self.action_map.items()}
        self.update_target()

    def act(self, state):
        """ボルツマン選択に基づいて各関節の行動を決定"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values_flat = self.q_network(state_tensor)[0]
        self.q_network.train()

        # Q値を (18, 3) の形状に変形
        q_values_matrix = q_values_flat.view(self.num_joints, self.action_dims)
        
        # 温度でスケールし、ソフトマックス関数で確率を計算
        scaled_q_values = q_values_matrix / self.temperature
        probabilities = torch.softmax(scaled_q_values, dim=1)
        
        # 各関節に対して、確率分布から行動をサンプリング
        action_indices = torch.multinomial(probabilities, 1).squeeze(1).cpu().numpy()
        
        return [self.index_to_action[i] for i in action_indices]

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size: return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = np.array(batch.action)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)

        current_q_values = self.q_network(state_batch)
        next_q_values_flat = self.target_network(next_state_batch).detach()
        next_q_matrix = next_q_values_flat.view(self.batch_size, self.num_joints, self.action_dims)
        max_next_q_values = torch.max(next_q_matrix, dim=2)[0]

        target_q_values = current_q_values.clone()

        for i in range(self.batch_size):
            is_final_state = done_batch[i]
            for j in range(self.num_joints):
                action_idx = self.action_map[action_batch[i, j]]
                flat_idx = j * self.action_dims + action_idx
                if is_final_state:
                    target_q_values[i, flat_idx] = reward_batch[i]
                else:
                    target_q_values[i, flat_idx] = reward_batch[i] + (self.gamma * max_next_q_values[i, j])
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()

        self.steps_done += 1

    def update_temperature(self, reward):
        """報酬に基づいて温度を更新する"""
        # 報酬を[-3, 2.5]の範囲から[0, 1]の範囲に正規化
        reward_min = -3.0
        reward_max = 2.5
        normalized_reward = (reward - reward_min) / (reward_max - reward_min)
        normalized_reward = max(0, min(1, normalized_reward)) # 範囲内にクリップ

        # 報酬が低いほど温度を上げ(探索)、高いほど下げる(活用)
        # normalized_rewardが0.5(報酬が-0.25)あたりで変化がゼロになる
        temp_change_factor = 0.5 - normalized_reward
        
        change = self.temp_update_rate * temp_change_factor
        self.temperature += change
        
        # 温度を[temp_min, temp_max]の範囲にクリップ
        self.temperature = max(self.temp_min, min(self.temperature, self.temp_max))

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

# --- 3. 学習の実行 ---
def main():
    robot = Supervisor()
    env = HexapodEnv(robot)
    
    # --- コンフィグを報酬連動ボルツマン用に変更 ---
    config = {
        'learning_rate': 1e-4, 'gamma': 0.99, 'temp_start': 1.0,
        'temp_min': 0.01, 'temp_update_rate': 0.0001, 'batch_size': 128,
        'target_update': 10, 'memory_size': 50000
    }
    # ------------------------------------
    
    agent = DQNAgent(env.state_size, env.num_joints, len(env.action_options), config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./dqn_training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard_logs"))
    summary_path = os.path.join(save_dir, "episode_summary.csv")
    with open(summary_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["Episode", "Steps", "Total Reward", "Final Y", "Comment"])

    all_episode_rewards = []
    print("Starting training for Hexapod on Webots...")
    
    for episode in range(500): # エピソード数を増やす
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, comment = env.step(action)
            agent.update_temperature(reward) # ★ 報酬に基づいて温度を更新
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            # --- 表示をTemperatureに変更 ---
            print(f"Step: {env.steps}, PosX: {env.gps.getValues()[0]:>6.2f}, PosY: {env.gps.getValues()[1]:>6.2f}, Reward: {reward:.2f}, Temp: {agent.temperature:.3f}, Action: {action}")
            # -----------------------------
            total_reward += reward
            if env.robot.step(0) == -1: # シミュレーションが閉じられたか確認
                done = True
                comment = "Simulation ended by user."

        if (episode + 1) % agent.target_update == 0:
            agent.update_target()
        
        final_y_pos = env.gps.getValues()[1]
        all_episode_rewards.append(total_reward)
        # --- 表示をTemperatureに変更 ---
        print(f"EP {episode+1} | Steps: {env.steps} | Reward: {total_reward:.2f} | Y Pos: {final_y_pos:.2f} | Temp: {agent.temperature:.3f} | {comment}")
        # -----------------------------

        with open(summary_path, mode='a', newline='') as f:
            csv.writer(f).writerow([episode + 1, env.steps, total_reward, final_y_pos, comment])
        tb_writer.add_scalar('Metrics/Total Reward', total_reward, episode)
        tb_writer.add_scalar('Metrics/Episode Length', env.steps, episode)
        tb_writer.add_scalar('Metrics/Final Y Position', final_y_pos, episode)
        # --- TensorBoardのログをTemperatureに変更 ---
        tb_writer.add_scalar('Hyperparameters/Temperature', agent.temperature, episode)
        # -----------------------------------------
        
        if (episode + 1) % 50 == 0:
            agent.save(os.path.join(save_dir, f"model_ep_{episode+1}.pth"))

    tb_writer.close()
    print("Training completed!")
    agent.save(os.path.join(save_dir, "final_model.pth"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.savefig(os.path.join(save_dir, "rewards_plot.png"))
    plt.show()

if __name__ == "__main__":
    main()
