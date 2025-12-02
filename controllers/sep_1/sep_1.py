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

TIME_STEP = 32
max_angle1 = 31
min_angle1 = -31
step_angle = 5

num_episodes = 1000
max_steps_per_episode = 1000000

# 脚数指定


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
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
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
    def __init__(self, robot, num_joints=18):
        self.robot = robot
        self.num_joints = num_joints
        self.TIME_STEP = TIME_STEP
        self.start_position = [0, 0.0, -0.07]
        self.start_rotation = [1, 0, 0, 1.5708]  # 四元数(x, y, z, w)
        self.initial_motor_positions = [-45, 90, -130, 45, 90, -130, -45, -90, 130, 45, -90, 130, 0, -90, 130, 0, 90, -130]
        self.action_options = [-1, 0, 1]
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
        if self.robot_nodo is None:
            raise ValueError("Robot node 'IRSL-XR06-01' not found.")
        
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.motors = [self.robot.getMotor(f"motor{i+1}") for i in range(18)]
        self.position_sensors = []
        for i in range(18):
            ps = self.robot.getPositionSensor(f"ps{i+1}")
            ps.enable(self.TIME_STEP)
            self.position_sensors.append(ps)
        
        self.gps = self.robot.getGPS("gps1")
        self.gps.enable(self.TIME_STEP)
        self.imu = self.robot.getInertialUnit("IU")
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

        if self.robot.step(self.TIME_STEP * 2) == -1:
            return self.get_state(), -100, True, "Simulation stopped"
        
        next_state = self.get_state()
        reward = self.calculate_reward()
        done, comment = self.is_done()
        if done and comment == "Goal reached!":
            reward += 200
        return next_state, reward, done, comment
    
    