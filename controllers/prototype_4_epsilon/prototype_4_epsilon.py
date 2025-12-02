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

#-----------------------------------------------------------------------------------------------------------------------
# Global parameters from prototype_1.py
TIME_STEP = 32
maxAngle1 = 35
minAngle1 = -35
stepAngle = 5

# ステップ数を可変にした
num_episodes = 1000  # Number of episodes for training
max_steps = 10000 # Maximum steps per episode

# Define action mappings from prototype_1.py
actions_map = {
    0: (1, 1),   # n=1, dir=1
    1: (1, -1),  # n=1, dir=-1
    2: (2, -1),  # n=2, dir=-1
    3: (2, 1),   # n=2, dir=1
    4: (3, -1),  # n=3, dir=-1
    5: (3, 1),   # n=3, dir=1
    6: (4, -1),  # n=4, dir=-1
    7: (4, 1),   # n=4, dir=1
}

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        """Save an experience"""
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network with improved architecture"""
    
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
    """DQN Agent with proper implementation"""
    
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = config.get('learning_rate', 1e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.target_update = config.get('target_update', 1000)
        self.memory_size = config.get('memory_size', 100000)
        
        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        
        # Tracking
        self.steps_done = 0
        self.loss_history = []
        self.reward_history = []
        
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() > self.epsilon:
            # Exploitation
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval() # Set to evaluation mode
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()
            self.q_network.train() # Set back to training mode
        else:
            # Exploration
            action = random.randrange(self.action_size)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target()
    
    def update_target(self):
        """Copy weights to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']

class RobotEnvironment:
    """Robot environment wrapper"""
    
    def __init__(self, robot):
        self.robot = robot
        self.TIME_STEP = TIME_STEP
        
        # Robot components setup
        self.setup_robot()
        
        # Environment parameters
        self.start_position = [0, 0, -4.4]
        self.start_rotation = [1, 0, 0, 1.57079632678966]
        self.goal_position = [0, 0.01, -4.4] # Goal is y >= 1
        self.initial_motor_positions = [0, 0, 0, 0, 0, 0, -25, 25, -25, 25, -25, 25] # From prototype_1.py
        
        # State tracking
        self.previous_distance = None
        self.step_count = 0
        self.episode_trajectory = [] # To store x,y coordinates for the final episode
        
    def setup_robot(self):
        """Initialize robot components"""
        # Get robot node
        self.robot_node = self.robot.getFromDef("IRSL-XR06-01") # Assuming this is the robot DEF name
        if self.robot_node is None:
            self.robot_node = self.robot.getFromDef("IRSL-XR06-02") # Fallback
            if self.robot_node is None:
                print("Error: Robot node 'IRSL-XR06-02' or 'IRSL-XR06-01' not found.")
                exit()

        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        # Motors
        RMnames = [
            'motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'motor6', 'motor7', 'motor8', 'motor9', 'motor10'
            , 'motor11', 'motor12', 'motor13', 'motor14', 'motor15', 'motor16', 'motor17', 'motor18'
        ]
        self.motors = [self.robot.getMotor(name) for name in RMnames]
        
        # Position sensors
        RMPS = [
            'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'PS7', 'PS8', 'PS9', 'PS10', 'PS11', 'PS12', 'PS13', 'PS14', 'PS15'
            , 'PS16', 'PS17', 'PS18'
        ]
        self.position_sensors = []
        for name in RMPS:
            ps = self.robot.getPositionSensor(name)
            ps.enable(self.TIME_STEP)
            self.position_sensors.append(ps)
        
        # GPS
        self.gps = self.robot.getGPS('gps2')
        self.gps.enable(self.TIME_STEP)
        
        # Inertial Unit
        self.imu = self.robot.getInertialUnit('IU2')
        self.imu.enable(self.TIME_STEP)
        
        # Touch sensors
        TSS = ['TS4', 'TS1', 'TS2', 'TS5', 'TS0', 'TS3']
        self.touch_sensors = []
        for name in TSS:
            ts = self.robot.getTouchSensor(name)
            ts.enable(self.TIME_STEP)
            self.touch_sensors.append(ts)
    
    def get_motor_values(self):
        """Get current motor positions in degrees"""
        return [math.degrees(ps.getValue()) for ps in self.position_sensors]

    def get_gps_values(self):
        """Get GPS values"""
        return self.gps.getValues()

    def get_imu_values(self):
        """Get Inertial Unit values (Roll, Pitch, Yaw)"""
        return self.imu.getRollPitchYaw()

    def get_touch_sensor_values(self):
        """Get Touch Sensor values"""
        return [ts.getValue() for ts in self.touch_sensors]

    def get_state(self):
        """Get normalized state vector"""
        gps_values = self.get_gps_values()
        imu_values = self.get_imu_values()
        motor_positions = self.get_motor_values()
        touch_values = self.get_touch_sensor_values()

        # State components based on prototype_1.py's C_STATE and additional useful info
        # GPS Z, Yaw, Motor 0, 1, 7, 6, GGM1, GGM0, TouchSensor[0-5]
        # For GGM, we'll use a placeholder or derive from movement if available.
        # For simplicity, let's use a state vector that combines relevant info.
        
        # Normalization factors (adjust as needed based on robot's actual ranges)
        # GPS coordinates typically range a few meters, IMU angles in radians, motor angles in degrees.
        # Let's normalize to roughly [-1, 1] or [0, 1]
        
        # GPS Z (vertical position, usually small changes)
        gps_z_norm = gps_values[2] / 5.0 # Assuming Z range of +/- 5m

        # Yaw angle (from IMU, in radians, normalize to [-1, 1] by dividing by pi)
        yaw_norm = imu_values[2] / math.pi

        # Selected motor positions (normalize degrees to [-1, 1] by dividing by max angle, e.g., 45)
        # Motor 0, 1, 7, 6 from prototype_1.py's C_STATE
        motor_0_norm = motor_positions[0] / 45.0
        motor_1_norm = motor_positions[1] / 45.0
        motor_7_norm = motor_positions[7] / 45.0
        motor_6_norm = motor_positions[6] / 45.0

        # GGM (placeholder, if not directly available or derivable, use 0)
        ggm0_norm = 0.0
        ggm1_norm = 0.0

        # Touch sensor values (already 0 or 1, no normalization needed)
        
        # Additional state components for better learning (from prototype_2.py)
        # Distance to goal
        current_pos_x = gps_values[0]
        current_pos_y = gps_values[1]
        distance_to_goal = abs(current_pos_y - self.goal_position[1])
        distance_to_goal_norm = distance_to_goal / 5.0 # Assuming max distance of 5m

        # Roll and Pitch (from IMU, for stability)
        roll_norm = imu_values[0] / math.pi
        pitch_norm = imu_values[1] / math.pi

        state = [
            gps_z_norm,
            yaw_norm,
            motor_0_norm,
            motor_1_norm,
            motor_7_norm,
            motor_6_norm,
            ggm0_norm,
            ggm1_norm,
            *touch_values, # Unpack touch sensor values
            distance_to_goal_norm,
            roll_norm,
            pitch_norm
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.step_count += 1
        
        # Execute action
        self.execute_action(action)
        
        # Let simulation run for a few steps to observe effect
        for _ in range(5): # Run for 5 TIME_STEPs
            if self.robot.step(self.TIME_STEP) == -1:
                break
        
        # Get new state
        next_state = self.get_state()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if done
        done, comment = self.is_done()

        # Store trajectory for the final episode
        current_pos = self.get_gps_values()
        self.episode_trajectory.append((current_pos[0], current_pos[1]))
        
        return next_state, reward, done, comment
    
    def doAct0(self, n, dir):
        """
        Controls specific motors based on 'n' and 'dir' parameters,
        rotating the leg by a specified value.
        Adapted from prototype_1.py.
        """
        current_motor_values = self.get_motor_values()
        
        # INITMOT from prototype_1.py, used for angle limits
        # Assuming INITMOT is a list of initial motor positions or limits
        # For simplicity, using 0 for initial position and maxAngle1/minAngle1 for limits
        INITMOT = [0] * 12 # Placeholder, adjust if actual initial positions are needed

        if n == 1: # Motors 1, 2, 4 (indices 0-11)
            va = current_motor_values[1]
            vb = current_motor_values[2]
            vc = current_motor_values[4]
            rada = va + dir * stepAngle
            radb = vb + dir * stepAngle
            radc = vc + dir * stepAngle
            
            # Check limits
            if not (INITMOT[1] + minAngle1 <= rada <= INITMOT[1] + maxAngle1 and \
                    INITMOT[2] + minAngle1 <= radb <= INITMOT[2] + maxAngle1 and \
                    INITMOT[4] + minAngle1 <= radc <= INITMOT[4] + maxAngle1):
                rada, radb, radc = va, vb, vc # Revert if out of bounds

            self.motors[1].setPosition(math.radians(rada))
            self.motors[2].setPosition(math.radians(radb))
            self.motors[4].setPosition(math.radians(radc))
            
        elif n == 2: # Motors 0, 3, 5
            va = current_motor_values[0]
            vb = current_motor_values[3]
            vc = current_motor_values[5]
            rada = va + dir * stepAngle
            radb = vb + dir * stepAngle
            radc = vc + dir * stepAngle

            if not (INITMOT[0] + minAngle1 <= rada <= INITMOT[0] + maxAngle1 and \
                    INITMOT[3] + minAngle1 <= radb <= INITMOT[3] + maxAngle1 and \
                    INITMOT[5] + minAngle1 <= radc <= INITMOT[5] + maxAngle1):
                rada, radb, radc = va, vb, vc

            self.motors[0].setPosition(math.radians(rada))
            self.motors[3].setPosition(math.radians(radb))
            self.motors[5].setPosition(math.radians(radc))

        elif n == 3: # Motors 6, 9, 11
            va = current_motor_values[6]
            vb = current_motor_values[9]
            vc = current_motor_values[11]
            rada = va + dir * stepAngle # Note the -1 for motor 6
            radb = vb + dir * stepAngle
            radc = vc + dir * stepAngle

            if not (INITMOT[6] + minAngle1 <= rada <= INITMOT[6] + maxAngle1 and \
                    INITMOT[9] + minAngle1 <= radb <= INITMOT[9] + maxAngle1 and \
                    INITMOT[11] + minAngle1 <= radc <= INITMOT[11] + maxAngle1):
                rada, radb, radc = va, vb, vc

            self.motors[6].setPosition(math.radians(rada))
            self.motors[9].setPosition(math.radians(radb))
            self.motors[11].setPosition(math.radians(radc))
                    
        elif n == 4: # Motors 7, 8, 10
            va = current_motor_values[7]
            vb = current_motor_values[8]
            vc = current_motor_values[10]
            rada = va + dir * stepAngle # Note the -1 for motor 7
            radb = vb + dir * stepAngle
            radc = vc + dir * stepAngle

            if not (INITMOT[7] + minAngle1 <= rada <= INITMOT[7] + maxAngle1 and \
                    INITMOT[8] + minAngle1 <= radb <= INITMOT[8] + maxAngle1 and \
                    INITMOT[10] + minAngle1 <= radc <= INITMOT[10] + maxAngle1):
                rada, radb, radc = va, vb, vc

            self.motors[7].setPosition(math.radians(rada))
            self.motors[8].setPosition(math.radians(radb))
            self.motors[10].setPosition(math.radians(radc))

    def execute_action(self, action_id):
        """Map action_id to doAct0 parameters and execute."""
        if action_id in actions_map:
            n_param, dir_param = actions_map[action_id]
            self.doAct0(n_param, dir_param)
        else:
            # Default action: set all motors to initial position if action_id is invalid
            for m_idx in range(len(self.motors)):
                self.motors[m_idx].setPosition(math.radians(self.initial_motor_positions[m_idx]))
    
    def calculate_reward(self):
        """Calculate reward based on current state, adapted from prototype_1.py"""
        current_pos = self.get_gps_values()
        current_distance = abs(current_pos[1] - self.goal_position[1])
        imu_values = self.get_imu_values()
        
        rew = 0
        
        # Distance-based reward (how much closer to goal)
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - current_distance

            # As the number of steps increases, the reward/penalty for distance changes is scaled down.
            # This is to balance the total reward across episodes of varying length.
            decay_factor = 1.0 / (1.0 + self.current_episode / num_episodes)

            if distance_improvement > 0:
                # Reward for moving closer to the goal.
                reward_value = distance_improvement * 500 * decay_factor
                rew += reward_value
            else:
                # Penalize for moving away from the goal.
                # The penalty is made explicit by using abs() and subtracting.
                penalty_value = abs(distance_improvement) * 500 * decay_factor
                rew -= penalty_value
        
        self.previous_distance = current_distance
        
        # Goal reached reward
        if current_pos[1] >= self.goal_position[1]: # Goal is y >= 1
            rew += 500


        # Step penalty (encourage efficiency)
        rew -= 0.1
        
        # # Boundary penalty (if robot moves too far from origin)
        # if abs(current_pos[0]) > 2 or abs(current_pos[1]) > 3:
        #     rew -= 100

        if abs(imu_values[0]) > math.pi/4 or abs(imu_values[1]) > math.pi/4:
            rew -= 10000

        if abs(current_pos[0]) > 2.5 or abs(current_pos[1]) > 3.5:
            rew -= 10000
        
        return rew
    
    def is_done(self):
        """Check if episode is done"""
        current_pos = self.get_gps_values()
        imu_values = self.get_imu_values()
        comment = ""
        
        # Goal reached (y-coordinate check)
        if current_pos[1] >= self.goal_position[1]:
            comment = "Goal reached!"
            self.goal_position[1] += 0.01
            print(comment)
            return True, comment
        
        # Robot fell over (roll or pitch too extreme)
        # prototype_1.py used math.degrees(gax) <= -45 or math.degrees(gax) >= 45 or abs(gax) >= 0.8 or abs(gay) >= 1.8
        # Let's use a simpler radian check for now, approx 45 degrees is pi/4
        if abs(imu_values[0]) > math.pi/4 or abs(imu_values[1]) > math.pi/4:
            comment = "Robot fell over!"
            print(comment)
            return True, comment
        
        # Out of bounds (if robot moves too far from origin)
        if abs(current_pos[0]) > 2.5 or abs(current_pos[1]) > 3.5: # Slightly larger bounds than penalty
            comment = "Out of bounds!"
            print(comment)
            return True, comment
        
        # # Max steps reached
        # if self.step_count >= self.max_steps:
        #     comment = "Max steps reached!"
        #     print(comment)
        #     return True, comment
        
        return False, comment
    
    def reset(self, episode_number):
        """Reset environment, update episode count, and set dynamic max_steps."""
        # Update episode number and calculate dynamic max_steps
        self.current_episode = episode_number
        self.max_steps = max_steps

        self.translation_field.setSFVec3f(self.start_position)
        self.rotation_field.setSFRotation(self.start_rotation)
        self.robot_node.resetPhysics()
        
        # Reset motors to initial positions
        for i, angle in enumerate(self.initial_motor_positions):
            self.motors[i].setPosition(math.radians(angle))
        
        # Wait for stabilization
        for _ in range(50): # Run for 50 TIME_STEPs for stabilization
            if self.robot.step(self.TIME_STEP) == -1:
                break
        
        self.step_count = 0
        self.previous_distance = None
        self.episode_trajectory = [] # Clear trajectory for new episode
        
        # Announce the new episode's max steps for clarity
        print(f"--- Episode {self.current_episode + 1} starting with max_steps = {self.max_steps} ---")

        return self.get_state()

def main():
    """Main training loop"""
    # Initialize Webots
    robot = Supervisor()
    env = RobotEnvironment(robot)
    
    # DQN configuration
    config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.9995, # Slightly slower decay than prototype_2.py
        'batch_size': 64,
        'target_update': 100, # Update target network more frequently (every 100 steps)
        'memory_size': 100000
    }
    
    # Initialize agent
    state_size = len(env.get_state())
    action_size = len(actions_map) # Number of possible actions
    agent = DQNAgent(state_size, action_size, config)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./dqn_training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- TensorBoard Setup ---
    tb_log_dir = os.path.join(save_dir, "tensorboard_logs")
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    # -------------------------

    # CSV file for episode summaries
    episode_summary_filepath = os.path.join(save_dir, "episode_summary.csv")
    with open(episode_summary_filepath, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Episode", "Steps", "Total Reward"])

    # Lists to store data for plotting
    all_episode_rewards = []
    all_episode_lengths = []
    
    print("Starting training...")

    for episode in range(num_episodes):
        state = env.reset(episode)
        total_reward = 0
        episode_length = 0
        
        print(f"EPISODE : {episode + 1}/{num_episodes}")
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, done, comment = env.step(action)

            # Get additional info for printing
            current_pos = env.get_gps_values()
            imu_values = env.get_imu_values()
            
            # Print current step info
            print(f"  EP: {env.current_episode + 1:<4} STEP: {env.step_count:<5} Action: {action:<2} Epsilon: {agent.epsilon:.4f} Reward: {reward:+9.2f} Y: {current_pos[1]:+8.4f} Roll: {math.degrees(imu_values[0]):+7.2f} Pitch: {math.degrees(imu_values[1]):+7.2f}")
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            state = next_state
            total_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Episode end logging and saving
        print(f"EPISODE {episode + 1} finished. Steps: {episode_length}, Total Reward: {total_reward:.2f}")
        with open(episode_summary_filepath, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([episode + 1, episode_length, total_reward, comment])
        
        # --- TensorBoard Logging ---
        tb_writer.add_scalar('Metrics/Total Reward', total_reward, episode)
        tb_writer.add_scalar('Metrics/Episode Length', episode_length, episode)
        tb_writer.add_scalar('Hyperparameters/Epsilon', agent.epsilon, episode)
        # ---------------------------

        all_episode_rewards.append(total_reward)
        all_episode_lengths.append(episode_length)

        # Save trajectory for the final episode
        if episode == num_episodes - 1:
            trajectory_filepath = os.path.join(save_dir, "final_episode_trajectory.csv")
            with open(trajectory_filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "X", "Y"])
                for i, (x, y) in enumerate(env.episode_trajectory):
                    writer.writerow([i, x, y])
            print(f"Final episode trajectory saved to {trajectory_filepath}")

    # --- Close TensorBoard Writer ---
    tb_writer.close()
    # --------------------------------

    print("Training completed!")
    
    # Final save of the model
    final_model_filepath = os.path.join(save_dir, "final_model.pth")
    agent.save(final_model_filepath)
    print(f"Final model saved to {final_model_filepath}")
    
    # Plotting results
    plt.figure(figsize=(14, 6))
    
    # Plot Episode Rewards
    plt.subplot(1, 2, 1)
    plt.plot(all_episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot Episode Lengths
    plt.subplot(1, 2, 2)
    plt.plot(all_episode_lengths)
    plt.title('Episode Lengths Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    plt.tight_layout()
    rewards_lengths_plot_filepath = os.path.join(save_dir, "rewards_lengths_plot.png")
    plt.savefig(rewards_lengths_plot_filepath)
    print(f"Rewards and lengths plot saved to {rewards_lengths_plot_filepath}")
    # plt.show() # Commented out for CLI environment

    # Plot Final Episode Trajectory
    if num_episodes > 0: # Ensure there was at least one episode
        final_trajectory_x = [p[0] for p in env.episode_trajectory]
        final_trajectory_y = [p[1] for p in env.episode_trajectory]

        plt.figure(figsize=(8, 8))
        plt.plot(final_trajectory_x, final_trajectory_y, marker='o', linestyle='-', markersize=2, label='Robot Trajectory')
        plt.scatter(env.start_position[0], env.start_position[1], color='green', s=100, zorder=5, label='Start Point')
        plt.scatter(env.goal_position[0], env.goal_position[1], color='red', s=100, zorder=5, label='Goal Point')
        plt.axhline(y=env.goal_position[1], color='r', linestyle='--', label='Goal Y-coordinate')
        plt.title(f'Robot Trajectory for Final Episode ({num_episodes})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.axis('equal') # Ensure equal scaling for x and y axes
        plt.legend()
        trajectory_plot_filepath = os.path.join(save_dir, "final_trajectory_plot.png")
        plt.savefig(trajectory_plot_filepath)
        print(f"Final trajectory plot saved to {trajectory_plot_filepath}")
        # plt.show() # Commented out for CLI environment

if __name__ == "__main__":
    main()