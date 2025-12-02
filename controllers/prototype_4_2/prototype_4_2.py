# -------------------------------------------------------------------------------------------
# [変更の概要]
# ロボットが毎ステップ転倒する問題を解決するため、`my_controllerA4.py`を参考に以下の点を修正しました。
#
# 1. 安定した初期姿勢の導入:
#    - 参考ファイルからロボットが安定して立てる各関節の初期角度を導入し、エピソード開始時に適用するようにしました。
#
# 2. アクションの再定義:
#    - 不安定さの原因であった単純なトライポッド歩容を廃止し、参考ファイルに基づいた
#      6種類の協調動作 x 2方向 = 12種類のアクションを新たに定義しました。
#
# 3. パラメータ調整:
#    - 関節の可動域や一度に動かす角度を、より安定する値に調整しました。
#
# 4. ゴール判定の変更:
#    - 参考ファイルに合わせて、ゴール判定をシンプルなZ座標による判定に変更しました。
#
# 5. [NEW] ステップ毎のログ出力機能を追加:
#    - mainループ内に、現在のステップ数、座標、報酬をコンソールに表示するprint文を追加しました。
# -------------------------------------------------------------------------------------------

from controller import Motor, GPS, PositionSensor, Supervisor, InertialUnit, Compass
import math
import random
import time
import numpy as np
import csv
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# グローバルパラメータ (my_controllerA4.pyを参考)
TIME_STEP = 64
maxAngle1 = 31
minAngle1 = -31
stepAngle = 15

num_episodes = 1000
max_steps_per_episode = 1000000

# --- my_controllerA4.pyを参考にした新しいアクションマッピング ---
# (動作パターンn, 方向dir)
actions_map = {
    0: (1, 1),   1: (1, -1),
    2: (2, 1),   3: (2, -1),
    4: (3, 1),   5: (3, -1),
    6: (4, 1),   7: (4, -1),
    8: (5, 1),   9: (5, -1),
    10: (6, 1),  11: (6, -1),
}

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Experience(*args))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = config.get('learning_rate', 1e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.99999)
        self.batch_size = config.get('batch_size', 128)
        self.target_update = config.get('target_update', 100)
        self.memory_size = config.get('memory_size', 100000)
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.memory_size)
        self.steps_done = 0

    def act(self, state):
        if random.random() > self.epsilon:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size: return
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)
    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target()

class RobotEnvironment:
    def __init__(self, robot):
        self.robot = robot
        self.TIME_STEP = TIME_STEP
        
        self.start_position = [0, 0.0, -0.07]
        self.start_rotation = [1, 0, 0, 1.57079632678966]
        # my_controllerA4.py の initmot を導入
        self.initial_motor_positions = [-45, 90, -130, 45, 90, -130, -45, -90, 130, 45, -90, 130, 0, -90, 130, 0, 90, -130]
        
        # 新しいゴール設定
        self.goal_y_coord = 0.1
        self.goal_x_coord = 0.0
        
        self.setup_robot()
        
        self.previous_pos = np.array(self.start_position)
        self.previous_dist_to_goal = self._get_dist_to_goal(self.previous_pos)
        self.previous_time = self.robot.getTime()
        self.step_count = 0
        self.episode_trajectory = []

    def setup_robot(self):
        self.robot_node = self.robot.getFromDef("IRSL-XR06-01")
        if self.robot_node is None:
            raise ValueError("Robot node 'IRSL-XR06-01' not found.")

        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        self.motors = [self.robot.getMotor(f'motor{i+1}') for i in range(18)]
        self.position_sensors = []
        for i in range(18):
            ps = self.robot.getPositionSensor(f'PS{i+1}')
            ps.enable(self.TIME_STEP)
            self.position_sensors.append(ps)
        
        self.gps = self.robot.getGPS('gps1')
        self.gps.enable(self.TIME_STEP)
        self.imu = self.robot.getInertialUnit('IU')
        self.imu.enable(self.TIME_STEP)
        
        self.touch_sensors = [self.robot.getTouchSensor(f'TS{i+1}') for i in range(6)]
        for ts in self.touch_sensors:
            ts.enable(self.TIME_STEP)
    
    def _get_dist_to_goal(self, position):
        """現在の位置からゴールまでの距離を計算"""
        return np.sqrt((position[0] - self.goal_x_coord)**2 + (position[1] - self.goal_y_coord)**2)

    def get_state(self):
        current_pos = np.array(self.gps.getValues())
        imu_values = self.imu.getRollPitchYaw()
        motor_positions = [math.degrees(ps.getValue()) for ps in self.position_sensors]
        touch_values = [ts.getValue() for ts in self.touch_sensors]
        
        current_time = self.robot.getTime()
        dt = current_time - self.previous_time
        velocity = (current_pos - self.previous_pos) / dt if dt > 0 else np.zeros(3)
        # print(f"Velocity: {velocity}")  # デバッグ用に速度を表示

        # ゴール座標を動的に生成
        goal_position_vec = np.array([self.goal_x_coord, self.goal_y_coord, current_pos[2]])
        relative_goal_pos = goal_position_vec - current_pos

        state = [
            imu_values[0] / math.pi,
            imu_values[1] / math.pi,
            imu_values[2] / math.pi,
            velocity[0] / 2.0,
            velocity[1] / 2.0,
            velocity[2] / 2.0,
            *[mp / 180.0 for mp in motor_positions], # 正規化範囲を-180~180に
            *touch_values,
            relative_goal_pos[0] / 5.0,
            relative_goal_pos[1] / 5.0, # Y座標の相対位置に変更
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        self.step_count += 1
        self.previous_pos = np.array(self.gps.getValues())
        self.previous_dist_to_goal = self._get_dist_to_goal(self.previous_pos)
        self.previous_time = self.robot.getTime()

        self.execute_action(action)
        
        if self.robot.step(self.TIME_STEP * 2) == -1: # 動作が反映されるまで少し待つ
            return self.get_state(), -100, True, "Simulation stopped"
        
        next_state = self.get_state()
        reward = self.calculate_reward()
        done, comment = self.is_done()
        if done and comment == "Goal reached!":
            reward += 200.0 # ゴール達成ボーナス
        return next_state, reward, done, comment
    
    def _set_motor_angle(self, motor_idx, current_angle, target_angle):
        """モーターの角度を可動域制限付きで設定"""
        init_pos = self.initial_motor_positions[motor_idx]
        if not (init_pos + minAngle1 <= target_angle <= init_pos + maxAngle1):
            target_angle = current_angle # 範囲外なら動かさない
        self.motors[motor_idx].setPosition(math.radians(target_angle))

    def execute_action(self, action_id):
        if action_id not in actions_map: return
        n, dr = actions_map[action_id]
        
        current_vals = [math.degrees(ps.getValue()) for ps in self.position_sensors]

        if n == 1:
            indices = [0, 3, 12]
            dirs = [dr, dr, -dr]
        elif n == 2:
            indices = [15, 9, 6]
            dirs = [-dr, dr, dr]
        elif n == 3:
            indices = [16, 10, 7]
            dirs = [-dr, dr, dr]
        elif n == 4:
            indices = [13, 1, 4]
            dirs = [-dr, dr, dr]
        elif n == 5:
            indices = [11, 17, 8]
            dirs = [dr, -dr, dr]
        elif n == 6:
            indices = [14, 2, 5]
            dirs = [dr, -dr, -dr]
        else:
            return

        for i, motor_idx in enumerate(indices):
            current_angle = current_vals[motor_idx]
            target_angle = current_angle + dirs[i] * stepAngle
            self._set_motor_angle(motor_idx, current_angle, target_angle)

    def calculate_reward(self):
        current_pos = np.array(self.gps.getValues())
        imu_values = self.imu.getRollPitchYaw()

        # XとY座標の両方を考慮したゴールへの距離に基づく報酬
        current_dist_to_goal = self._get_dist_to_goal(current_pos)
        distance_reward = (self.previous_dist_to_goal - current_dist_to_goal) * 300.0

        fall_penalty = -10.0 if abs(imu_values[0]) > math.pi / 3 or abs(imu_values[1]) > math.pi / 3 else 0

        control_cost = -0.01
        
        reward = distance_reward + fall_penalty + control_cost
        
        return reward

    def is_done(self):
        current_pos = self.gps.getValues()
        imu_values = self.imu.getRollPitchYaw()
        
        # 新しいゴール判定
        is_goal_x = self.goal_x_coord - 0.5 <= current_pos[0] <= self.goal_x_coord + 0.5
        is_goal_y = self.goal_y_coord <= current_pos[1]

        # if is_goal_x and is_goal_y:
        if is_goal_y:
            self.goal_y_coord += 0.02 # 次のゴールを少し先に設定
            if self.goal_y_coord > 20.0:
                self.goal_y_coord = 20.0
            return True, "Goal reached!"

        if abs(imu_values[0]) > math.pi / 2.5 or abs(imu_values[1]) > math.pi / 2.5:
            print("imu:", imu_values[0], imu_values[1])
            return True, "Robot fell over!"
        if self.step_count >= max_steps_per_episode:
            return True, "Max steps reached!"
        return False, ""

    def reset(self, episode_number):
        self.translation_field.setSFVec3f(self.start_position)
        self.rotation_field.setSFRotation(self.start_rotation)
        self.robot_node.resetPhysics()
        
        for i, angle in enumerate(self.initial_motor_positions):
            self.motors[i].setPosition(math.radians(angle))

        self.robot.step(self.TIME_STEP * 10)
        self.step_count = 0
        self.previous_pos = np.array(self.gps.getValues())
        self.previous_dist_to_goal = self._get_dist_to_goal(self.previous_pos)
        self.previous_time = self.robot.getTime()
        self.episode_trajectory = []
        return self.get_state()

def main():
    robot = Supervisor()
    env = RobotEnvironment(robot)
    config = {
        'learning_rate': 5e-5, 'gamma': 0.99, 'epsilon_start': 1.0,
        'epsilon_min': 0.01, 'epsilon_decay': 0.99995, 'batch_size': 128,
        'ta55rget_update': 200, 'memory_size': 150000
    }
    state_size, action_size = len(env.get_state()), len(actions_map)
    agent = DQNAgent(state_size, action_size, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./dqn_training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard_logs"))
    summary_path = os.path.join(save_dir, "episode_summary.csv")
    with open(summary_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["Episode", "Steps", "Total Reward", "Final Y", "Comment"])

    all_episode_rewards = []
    agent_path = [] # To store trajectory of the final episode
    print("Starting training for Hexapod...")
    for episode in range(num_episodes):
        state = env.reset(episode)
        total_reward = 0
        print(f"--- Episode {episode + 1}/{num_episodes} ---")
        for step in range(max_steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, comment = env.step(action)

            # Record trajectory for the final episode
            if episode == num_episodes - 1:
                current_pos = env.gps.getValues()
                agent_path.append((current_pos[0], current_pos[1]))

            # [変更点] ステップ毎の情報を表示
            current_pos = env.gps.getValues()
            print(f"  Step: {step + 1:<4} | Pos: (X:{current_pos[0]:>6.2f}, Y:{current_pos[1]:>6.2f}, Z:{current_pos[2]:>6.2f}) | Reward: {reward:>7.2f}")

            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if done: break
        
        final_y_pos = env.gps.getValues()[1]
        print(f"EP {episode + 1} done. Steps: {env.step_count}, Reward: {total_reward:.2f}, Final Y: {final_y_pos:.2f}, {comment}")
        with open(summary_path, mode='a', newline='') as f:
            csv.writer(f).writerow([episode + 1, env.step_count, total_reward, final_y_pos, comment])
        
        tb_writer.add_scalar('Metrics/Total Reward', total_reward, episode)
        tb_writer.add_scalar('Metrics/Episode Length', env.step_count, episode)
        tb_writer.add_scalar('Metrics/Final Y Position', final_y_pos, episode)
        tb_writer.add_scalar('Hyperparameters/Epsilon', agent.epsilon, episode)
        all_episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            agent.save(os.path.join(save_dir, f"model_ep_{episode+1}.pth"))

    tb_writer.close()
    print("Training completed!")
    agent.save(os.path.join(save_dir, "final_model.pth"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.savefig(os.path.join(save_dir, "rewards_plot.png"))
    plt.close()

    # Save and plot trajectory for the final episode
    if agent_path:
        # Save trajectory to CSV
        trajectory_filepath = os.path.join(save_dir, "final_episode_trajectory.csv")
        with open(trajectory_filepath, mode='w', newline='') as f:
            csv_writer_traj = csv.writer(f)
            csv_writer_traj.writerow(["Step", "X", "Y"])
            for i, (x, y) in enumerate(agent_path):
                csv_writer_traj.writerow([i, x, y])
        print(f"Final episode trajectory saved to {trajectory_filepath}")

        # Plot trajectory
        path_x, path_y = zip(*agent_path)
        plt.figure(figsize=(8, 8))
        plt.plot(path_x, path_y, marker='o', linestyle='-', markersize=2, label='Robot Trajectory')
        plt.scatter(env.start_position[0], env.start_position[1], color='green', s=100, zorder=5, label='Start Point')
        
        # # ゴールエリアを描画
        # goal_x_range = [env.goal_x_coord - 0.5, env.goal_x_coord + 0.5]
        # plot_y_max = max(path_y)
        # goal_area_top = max(plot_y_max, env.goal_y_coord) + 0.5
        # plt.fill_between(goal_x_range, env.goal_y_coord, goal_area_top, color='r', alpha=0.2, label='Goal Area')

        plt.axhline(y=env.goal_y_coord, color='r', linestyle='--', label='Goal Y-coordinate')

        plt.title(f'Robot Trajectory for Final Episode ({num_episodes})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        trajectory_plot_filepath = os.path.join(save_dir, "final_trajectory_plot.png")
        plt.savefig(trajectory_plot_filepath)
        print(f"Final trajectory plot saved to {trajectory_plot_filepath}")
        plt.close()

if __name__ == "__main__":
    main()