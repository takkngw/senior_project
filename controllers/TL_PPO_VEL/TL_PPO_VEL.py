"""
DQN (sep_3_bolzman.py) を PPO (Proximal Policy Optimization) に書き換えたコントローラ
RL_PPO.py に詳細なステップログ出力を追加したバージョン (RL_PPO_VEL.py)
さらに学習済みモデルを読み込む機能を追加したバージョン (TL_PPO_VEL.py)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import time
import math
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from controller import Supervisor, Motor, GPS, InertialUnit, PositionSensor
from collections import deque

TIME_STEP = 32
# 各関節の初期姿勢からの可動域（度）
maxAngle1 = 20
minAngle1 = -20
# 1ステップで動かす角度（度）
stepAngle = 10


# --- 1. 環境 (Environment) ---
class HexapodEnv:
    """Webotsシミュレータと連携するヘキサポッド環境クラス"""
    def __init__(self, robot, num_joints=18):
        self.robot = robot
        self.num_joints = num_joints
        
        self.initial_motor_positions = [-45, 70, -110, 45, 70, -110, -45, -70, 110, 45, -70, 110, 0, -70, 110, 0, 70, -110]
        self.start_position = [0, 0.0, -0.07]
        self.start_rotation = [1, 0, 0, 1.57079632678966]
        self.goal_y_position = 0.5 # 転移学習を行う場合は、ここを 1.0 や 2.0 に変更してください
    
        self.goal_count = 0
        self.current_velocity = np.zeros(3) # 速度を保持する変数を追加

        self._setup_robot()
        
        self.previous_pos = np.array(self.gps.getValues())
        self.previous_time = self.robot.getTime()

        # 歩容周期性のための設定
        self.gait_cycle_period = 10  # 周期性の評価に使うステップ数（調整可能）
        self.joint_history = deque(maxlen=self.gait_cycle_period)

        self.state_size = len(self.get_state())
        
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

    def update_physics_state(self):
        """物理状態（速度など）の更新を一箇所にまとめる"""
        current_pos = np.array(self.gps.getValues())
        current_time = self.robot.getTime()
        dt = current_time - self.previous_time
        
        if dt > 0:
            self.current_velocity = (current_pos - self.previous_pos) / dt
        else:
            self.current_velocity = np.zeros(3)
            
        # 更新後に現在値を保存
        self.previous_pos = current_pos
        self.previous_time = current_time

    def get_state(self):
        """センサーから現在の状態を取得"""
        # 注: ここでの速度計算は削除し、self.current_velocityを使う
        
        imu_values = self.imu.getRollPitchYaw()
        motor_positions = [math.degrees(ps.getValue()) for ps in self.position_sensors]
        current_pos = np.array(self.gps.getValues())
        
        relative_goal_pos_y = self.goal_y_position - current_pos[1]

        # 速度の正規化 (最大速度を想定して -1~1 に収める)
        max_lin_vel = 0.5 
        norm_vel = np.clip(self.current_velocity / max_lin_vel, -1.0, 1.0)

        state = [
            imu_values[0] / math.pi, imu_values[1] / math.pi, imu_values[2] / math.pi,
            norm_vel[0], norm_vel[1], norm_vel[2], # 計算済みの速度を使用
            *[mp / 180.0 for mp in motor_positions],
            relative_goal_pos_y / 5.0
        ]
        # NaNを0に置換し、値があまりに大きくならないようにクリップする安全策
        return np.clip(np.nan_to_num(np.array(state, dtype=np.float32), nan=0.0), -10.0, 10.0)
    
    def execute_action(self, action_vector):
        """18次元の行動ベクトル（-1, 0, 1）をモーター指令に変換"""
        current_angles = [math.degrees(ps.getValue()) for ps in self.position_sensors]
        for i, action_val in enumerate(action_vector):
            target_angle = current_angles[i] + action_val * stepAngle
            
            init_pos = self.initial_motor_positions[i]
            clamped_target = max(init_pos + minAngle1, min(target_angle, init_pos + maxAngle1))
            
            self.motors[i].setPosition(math.radians(clamped_target))

    def calculate_reward(self):
        """報酬を計算（スケーリングを修正）"""
        # ここでの速度計算も削除し、self.current_velocityを使う
        
        vel_x, vel_y, vel_z = self.current_velocity
        current_pos = np.array(self.gps.getValues())
        imu_values = self.imu.getRollPitchYaw()

        reward = 0.0
        
        # 1. 前進報酬 (係数を 20 -> 2.0 程度に下げる -> 10.0に上げる)
        # 目標: ステップごとに +0.01 ~ +0.1 程度稼ぐイメージ -> もっと強く動機づける
        reward += np.clip(vel_y * 10.0, -1.0, 2.0)
        
        # 2. 安定性ペナルティ (横ズレ、高さブレ)
        reward -= 0.05 * (abs(vel_x) + abs(vel_z))
        reward -= 0.05 * abs(current_pos[0])
        
        # 3. 姿勢ペナルティ (傾きすぎたら減点)
        if abs(imu_values[0]) > 0.5 or abs(imu_values[1]) > 0.5:
             reward -= 0.1

        # 4. 生存ボーナス (転ばなければ少しプラス)
        reward -= 0.1

        # 5. 高さ維持 (低すぎるとペナルティ)
        if current_pos[2] < 0.03: # ボディが地面につきそう
            reward -= 0.05

        # 6. ゴール到達 /転倒 (大きな報酬はそのまま)
        if current_pos[1] >= self.goal_y_position:
            reward += 10.0 # 50だと大きすぎて勾配が跳ねることがあるので10程度推奨
            
        if abs(imu_values[0]) > math.pi / 2.5 or abs(imu_values[1]) > math.pi / 2.5:
            reward -= 10.0 # -1000は大きすぎる。エピソード即終了なら -10 程度で十分

        return reward
    
    def is_done(self):
        """エピソードの終了条件を判定"""
        current_pos = self.gps.getValues()
        imu_values = self.imu.getRollPitchYaw()
        max_steps = 10000
        max_count = 0

        if current_pos[1] >= self.goal_y_position:
            # self.goal_count += 1
            # if self.goal_count > 0 and self.goal_count % 50 == 0:
            #     self.goal_y_position = min(5.0, self.goal_y_position + 0.5)
            return True, "Goal reached!"
        if abs(imu_values[0]) > math.pi / 2.5 or abs(imu_values[1]) > math.pi / 2.5:
            return True, "Robot fell over!"
        if self.steps >= max_steps: # 最大ステップ数
            if current_pos[1] > self.goal_y_position - 0.5:
                max_count += 1
            if max_count == 10:
                max_count = 0
                max_steps += 1000
            return True, "Max steps reached!"
        return False, ""

    def step(self, action_vector):
        self.steps += 1
        
        self.execute_action(action_vector)
        
        if self.robot.step(TIME_STEP) == -1:
            return self.get_state(), 0, True, "Simulation stopped"
        
        # ★先に物理状態を更新してから、StateとRewardを計算する
        self.update_physics_state()
        
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

        self.robot.step(TIME_STEP * 10) # 安定待機
        self.current_velocity = np.zeros(3)
        self.steps = 0
        self.previous_pos = np.array(self.gps.getValues())
        self.previous_time = self.robot.getTime()
        
        # 周期性の履歴をクリア
        self.joint_history.clear()
        
        return self.get_state()

from torch.distributions.normal import Normal

# --- 2. PPO用ネットワーク (Actor-Critic) ---
class ActorCritic(nn.Module):
    """PPOのためのActor-Criticネットワーク（連続行動空間版）"""
    def __init__(self, state_size, num_joints, hidden_size=512):
        super(ActorCritic, self).__init__()
        self.num_joints = num_joints
        
        # 共通のベースネットワーク
        self.shared_base = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actorヘッド (行動の平均を出力)
        self.actor_mean = nn.Linear(hidden_size, num_joints)
        
        # 行動の標準偏差の対数 (学習可能なパラメータ)
        # 独立したパラメータにすることで、学習初期の不安定さを軽減
        self.actor_log_std = nn.Parameter(torch.zeros(1, num_joints) * -0.5) # 初期値を小さく設定 (-1.0 -> -0.5 で探索を増やす)

        # Criticヘッド (状態価値を出力)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if torch.isnan(x).any():
            print(f"[NaN Detected] Input x contains {torch.isnan(x).sum().item()} NaNs")

        base_out = self.shared_base(x)
        
        # Actor: 平均を計算し、tanhで[-1, 1]に収める
        mean = torch.tanh(self.actor_mean(base_out))
        if torch.isnan(mean).any():
            print(f"[NaN Detected] Actor mean contains {torch.isnan(mean).sum().item()} NaNs")
        
        # log_stdをバッチサイズに合わせる
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        if torch.isnan(std).any():
            print(f"[NaN Detected] Actor std contains {torch.isnan(std).sum().item()} NaNs")
        
        # Critic: 状態価値を計算
        value = self.critic_head(base_out)
        
        # 行動の確率分布 (正規分布)
        dist = Normal(mean, std)
        
        return dist, value

        # モデルの初期化 (安定性向上のため)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # 共通層にはsqrt(2)でReLUに適した初期化
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0.0)
                
        self.shared_base.apply(init_weights)
        
        # Actorヘッドは行動を小さく保つために標準偏差0.01で初期
        torch.nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        torch.nn.init.constant_(self.actor_mean.bias, 0.0)
        
        # Criticヘッドは標準偏差1.0で初期化
        torch.nn.init.orthogonal_(self.critic_head.weight, 1.0)
        torch.nn.init.constant_(self.critic_head.bias, 0.0)

# --- 3. PPOエージェント (Agent) ---
class PPOAgent:
    """PPOアルゴリズムを実行するエージェント（連続行動空間版）"""
    def __init__(self, state_size, num_joints, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.num_joints = num_joints
        
        # PPOハイパーパラメータ
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_coef = config.get('clip_coef', 0.2)
        self.ent_coef = config.get('ent_coef', 0.001)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.n_steps = config.get('n_steps', 4096)
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 64)

        self.network = ActorCritic(state_size, num_joints).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)
        
        # PPO用オンポリシー・ストレージの初期化
        self.storage_size = self.n_steps
        self.states = torch.zeros((self.storage_size, self.state_size)).to(self.device)
        self.actions = torch.zeros((self.storage_size, self.num_joints)).to(self.device)
        self.log_probs = torch.zeros((self.storage_size, self.num_joints)).to(self.device)
        self.rewards = torch.zeros(self.storage_size).to(self.device)
        self.dones = torch.zeros(self.storage_size).to(self.device)
        self.values = torch.zeros(self.storage_size).to(self.device)
        self.storage_idx = 0

    def act(self, state):
        """現在の状態から連続行動をサンプリング"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            dist, value = self.network(state_tensor)
        self.network.train()
        
        # 分布からアクションをサンプリング
        action = dist.sample() # (1, 18)
        
        # 対数確率を計算
        log_prob = dist.log_prob(action) # (1, 18)
        
        # 環境に渡すアクションは[-1, 1]にクリップする
        action_for_env = torch.clamp(action, -1.0, 1.0)
        
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0), action_for_env.squeeze(0).cpu().numpy()

    def store(self, state, action, log_prob, reward, done, value):
        """ストレージに経験を保存"""
        if self.storage_idx < self.storage_size:
            self.states[self.storage_idx] = torch.FloatTensor(state).to(self.device)
            self.actions[self.storage_idx] = action
            self.log_probs[self.storage_idx] = log_prob
            self.rewards[self.storage_idx] = torch.tensor(reward).to(self.device)
            self.dones[self.storage_idx] = torch.tensor(done).to(self.device)
            self.values[self.storage_idx] = value
            self.storage_idx += 1

    def compute_gae(self, next_value, next_done):
        """GAE (Generalized Advantage Estimation) を計算"""
        advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(self.storage_size)):
            if t == self.storage_size - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + self.values
        return advantages, returns

    def learn(self, next_value, next_done):
        """収集したデータでPPO学習を実行"""
        advantages, returns = self.compute_gae(next_value, next_done)
        
        b_states = self.states
        b_actions = self.actions # 連続値なので.long()は不要
        b_old_log_probs_matrix = self.log_probs
        b_old_log_probs = b_old_log_probs_matrix.sum(dim=1)
        b_advantages = advantages
        b_returns = returns
        
        clip_fracs = []
        approx_kls = []
        
        for epoch in range(self.n_epochs):
            indices = np.arange(self.storage_size)
            np.random.shuffle(indices)
            
            for start in range(0, self.storage_size, self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                mb_states = b_states[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_log_probs = b_old_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]

                # 現在のネットワークで分布と価値を再計算
                dist, new_values = self.network(mb_states)
                
                # 新しい対数確率
                new_log_probs_matrix = dist.log_prob(mb_actions)
                new_log_probs = new_log_probs_matrix.sum(dim=1)
                
                # エントロピー計算
                entropy = dist.entropy().sum(dim=1).mean()
                
                new_values = new_values.view(-1)
                
                # --- PPO 損失計算 ---
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    clip_frac = ((torch.abs(ratio - 1.0) > self.clip_coef).float().mean().item())
                    clip_fracs.append(clip_frac)

                # Advantage の正規化
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic 損失 (Value Loss)
                v_loss = F.mse_loss(new_values, mb_returns)

                # 合計損失
                loss = pg_loss - self.ent_coef * entropy + self.vf_coef * v_loss
                
                if torch.isnan(loss).any():
                    print(f"[NaN Detected] Loss contains {torch.isnan(loss).sum().item()} NaNs")
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        self.storage_idx = 0
        return v_loss.item(), pg_loss.item(), entropy.item(), np.mean(approx_kls), np.mean(clip_fracs)
        
    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)

# --- 4. 学習の実行 (PPOループ) ---
def main():
    robot = Supervisor()
    env = HexapodEnv(robot)
    
    # --- PPO用コンフィグ ---
    config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_coef': 0.2,
        'ent_coef': 0.02, # 連続行動空間ではエントロピーボーナスを小さめに設定することが多い -> 探索不足解消のため 0.001 -> 0.01 -> 0.02
        'vf_coef': 0.5,
        'n_steps': 4096,  # このステップ数収集したら学習
        'n_epochs': 10,   # 収集したデータで何回学習するか
        'batch_size': 64,
    }
    # ----------------------

    total_episodes = 1000

    num_joints = env.num_joints
    
    agent = PPOAgent(env.state_size, num_joints, config)

    # --- モデルのロード (Transfer Learning) ---
    # 読み込みたいモデルのパスを指定してください
    # 例: "ppo_training_20251201_120000/model_ep_500.pth"
    model_path = "pretrained_model.pth" 
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        try:
            agent.network.load_state_dict(torch.load(model_path, map_location=agent.device))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
    else:
        print(f"No model found at {model_path}, starting from scratch.")
    # ---------------------------------------
    
    # --- ログ設定 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./ppo_training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard_logs"))
    summary_path = os.path.join(save_dir, "episode_summary.csv")
    with open(summary_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["Episode", "Steps", "Total Reward", "Final Y", "Comment"])
    
    # --- Step-by-step data logging setup ---
    step_data_path = os.path.join(save_dir, "step_data.csv")
    with open(step_data_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["Episode", "Step", "X", "Y", "Z", "Vel_X", "Vel_Y", "Vel_Z", "Reward"])

    print("Starting PPO training for Hexapod on Webots...")

    # --- PPO 学習ループ ---
    global_step = 0
    episode_count = 0

    # --- プロット用のリスト ---
    episodes = []
    total_rewards = []
    episode_lengths = []
    final_y_positions = []
    
    learning_episodes = []
    value_losses = []
    policy_losses = []
    entropies = []
    approx_kls = []
    clip_fractions = []
    
    # 初期状態
    state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0

    while episode_count < total_episodes:
        
        # 1. データ収集フェーズ (n_steps)
        for step in range(agent.n_steps):
            global_step += 1
            episode_steps += 1
            
            # 行動を決定
            action, log_probs, value, action_values = agent.act(state)
            
            # 環境をステップ
            next_state, reward, done, comment = env.step(action_values)
            
            # --- Log step data ---
            current_pos = env.gps.getValues()
            velocity_xyz = env.current_velocity
            with open(step_data_path, mode='a', newline='') as f:
                csv.writer(f).writerow([episode_count + 1, global_step, current_pos[0], current_pos[1], current_pos[2], velocity_xyz[0], velocity_xyz[1], velocity_xyz[2], reward])
            # --- End log step data ---

            # ストレージに保存
            agent.store(state, action, log_probs, reward, done, value)
            
            state = next_state
            episode_reward += reward

            if episode_steps > 0 and episode_steps % 500 == 0:
                current_y_pos = env.gps.getValues()[1]
                print(f"  ... (EP {episode_count + 1} / Step: {episode_steps}) Y Pos: {current_y_pos:.2f}, Last Reward: {reward:.2f}")

            if done or env.robot.step(0) == -1:
                if env.robot.step(0) == -1: comment = "Sim stopped"
                
                final_y_pos = env.gps.getValues()[1]
                print(f"EP {episode_count+1} | Steps: {episode_steps} | Reward: {episode_reward:.2f} | Y Pos: {final_y_pos:.2f} | {comment}")

                tb_writer.add_scalar('Metrics/Total Reward', episode_reward, episode_count + 1)
                tb_writer.add_scalar('Metrics/Episode Length', episode_steps, episode_count + 1)
                tb_writer.add_scalar('Metrics/Final Y Position', final_y_pos, episode_count + 1)
                with open(summary_path, mode='a', newline='') as f:
                    csv.writer(f).writerow([episode_count + 1, episode_steps, episode_reward, final_y_pos, comment])

                # プロット用リストにデータを追加
                episodes.append(episode_count + 1)
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                final_y_positions.append(final_y_pos)
                
                # リセット
                state = env.reset()
                done = False
                episode_reward = 0
                episode_steps = 0
                episode_count += 1
                
                if env.robot.step(0) == -1:
                    print("Simulation manually stopped.")
                    break
        
        if env.robot.step(0) == -1:
            break

        # 2. 学習フェーズ
        # 次のステップの価値(Value)を計算（GAEの計算に必要）
        next_value = 0.0
        next_done = 0.0
        if not done: # n_stepsの最後にエピソードが終了していなかった場合
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                _, next_value_tensor = agent.network(state_tensor)
                next_value = next_value_tensor.squeeze(0).item()
            next_done = done
            
        # 学習を実行
        v_loss, pg_loss, entropy, approx_kl, clip_frac = agent.learn(next_value, next_done)

        tb_writer.add_scalar('Loss/Value Loss', v_loss, episode_count)
        tb_writer.add_scalar('Loss/Policy Loss', pg_loss, episode_count)
        tb_writer.add_scalar('Metrics/Entropy', entropy, episode_count)
        tb_writer.add_scalar('Metrics/Approx KL', approx_kl, episode_count)
        tb_writer.add_scalar('Metrics/Clip Fraction', clip_frac, episode_count)

        # プロット用リストにデータを追加
        learning_episodes.append(episode_count)
        value_losses.append(v_loss)
        policy_losses.append(pg_loss)
        entropies.append(entropy)
        approx_kls.append(approx_kl)
        clip_fractions.append(clip_frac)

        # 3. モデルの保存
        if (global_step // agent.n_steps) % 25 == 0: # 25回学習するごとに保存
            agent.save(os.path.join(save_dir, f"model_ep_{episode_count}.pth"))

    tb_writer.close()
    print("Training completed!")
    agent.save(os.path.join(save_dir, "final_model.pth"))

    # --- 学習結果のプロットと保存 ---
    print("Saving plots...")

    # エピソードベースのメトリクスをプロット
    if episodes:
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        fig.suptitle('Episode Metrics', fontsize=16)

        axs[0].plot(episodes, total_rewards, label='Total Reward')
        axs[0].set_ylabel("Total Reward")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(episodes, episode_lengths, label='Episode Length')
        axs[1].set_ylabel("Steps")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(episodes, final_y_positions, label='Final Y Position')
        axs[2].set_ylabel("Y Position (m)")
        axs[2].set_xlabel("Episode")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, "episode_metrics.png"))
        plt.close(fig)

    # 学習ベースのメトリクスをプロット
    if learning_episodes: # 学習が1回は行われたか確認
        fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)
        fig.suptitle('Learning Metrics', fontsize=16)

        axs[0].plot(learning_episodes, value_losses, label='Value Loss')
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(learning_episodes, policy_losses, label='Policy Loss')
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(learning_episodes, entropies, label='Entropy')
        axs[2].set_ylabel("Entropy")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(learning_episodes, approx_kls, label='Approx KL')
        axs[3].set_ylabel("KL")
        axs[3].legend()
        axs[3].grid(True)

        axs[4].plot(learning_episodes, clip_fractions, label='Clip Fraction')
        axs[4].set_ylabel("Fraction")
        axs[4].set_xlabel("Episode (at time of learning)")
        axs[4].legend()
        axs[4].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, "learning_metrics.png"))
        plt.close(fig)
    
    print("Plots saved.")
    
if __name__ == "__main__":
    main()
