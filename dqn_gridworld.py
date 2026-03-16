import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
import os

# ==========================================
# 1. 全局配置
# ==========================================
GRID_SIZE = 10
NUM_ACTIONS = 5
DX =[-1, 1, 0, 0, 0]
DY =[0, 0, -1, 1, 0]
WHITE, YELLOW, BLUE = 0, 1, 2

# ==========================================
# 2. DQN 神经网络模型 (带 Embedding，完美适应迷宫)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, grid_size, num_actions):
        super(QNetwork, self).__init__()
        self.embedding = nn.Embedding(grid_size * grid_size, 128)
        # 自定义网络层
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state_idx):
        emb = self.embedding(state_idx)
        return self.fc(emb)

# ==========================================
# 3. 经验回放池 (Off-Policy 的核心)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.long),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.long),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )
        
    def __len__(self):
        return len(self.buffer)

# ==========================================
# 4. 环境类 GridWorld (严格基于你的奖励逻辑)
# ==========================================
class GridWorld:
    def __init__(self):
        self.grid = np.full((GRID_SIZE, GRID_SIZE), WHITE)
        self.goal_pos = (GRID_SIZE - 1, 0)
        self.start_pos = (0, 0)
        
        # 奖励设定 (保持你的设定，合理缩放)
        self.r_white = -0.1     # 稍微增大每步惩罚，逼迫它快跑
        self.r_yellow = -10.0
        self.r_blue = 10.0
        self.r_boundary = -2.0 # 撞墙惩罚，不要太大以免引发过度避险

    # 孤岛填埋算法，确保所有白色区域都能被访问到
    def prune_unreachable_areas(self):
        visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        queue = deque([self.goal_pos])
        visited[self.goal_pos[0], self.goal_pos[1]] = True

        while queue:
            curr_x, curr_y = queue.popleft()
            for i in range(4):
                nx, ny = curr_x + DX[i], curr_y + DY[i]
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if self.grid[nx, ny] != YELLOW and not visited[nx, ny]:
                        visited[nx, ny] = True
                        queue.append((nx, ny))

        converted_count = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] == WHITE and not visited[i, j]:
                    self.grid[i, j] = YELLOW
                    converted_count += 1
        return converted_count < (GRID_SIZE * GRID_SIZE * 0.05)

    def generate_map(self):
        valid = False
        while not valid:
            self.grid.fill(WHITE)
            self.grid[self.goal_pos[0], self.goal_pos[1]] = BLUE

            num_yellows = int(GRID_SIZE * GRID_SIZE * 0.35)
            placed, attempts = 0, 0
            while placed < num_yellows and attempts < num_yellows * 5:
                r, c = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                if self.grid[r, c] == WHITE:
                    self.grid[r, c] = YELLOW
                    placed += 1
                attempts += 1
            if self.prune_unreachable_areas():
                valid = True

    def step(self, current_pos, action):
        cx, cy = current_pos
        nx = cx + DX[action]
        ny = cy + DY[action]
        
        reward = 0.0
        done = False
        next_pos = current_pos

        if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
            reward = self.r_boundary
            next_pos = current_pos 
        else:
            next_pos = (nx, ny)
            if self.grid[nx, ny] == YELLOW:
                reward = self.r_yellow
                done = True # 踩黄死
            elif self.grid[nx, ny] == BLUE:
                reward = self.r_blue
                done = True # 到达终点
            else:
                reward = self.r_white

        return next_pos, reward, done

    def render(self, robot_pos):
        os.system("cls" if os.name == "nt" else "clear")
        print("  " + " ".join([f"{j:2d}" for j in range(GRID_SIZE)]))
        for i in range(GRID_SIZE):
            row_str = f"{i:2d} "
            for j in range(GRID_SIZE):
                if (i, j) == robot_pos:
                    row_str += "@  "
                elif self.grid[i, j] == BLUE:
                    row_str += "B  "
                elif self.grid[i, j] == YELLOW:
                    row_str += "Y  "
                else:
                    row_str += ".  "
            print(row_str)
        print("-------------------")

# 辅助函数：将二维坐标压平成一维索引 (用于 Embedding)
def get_state_idx(pos):
    return pos[0] * GRID_SIZE + pos[1]

# ==========================================
# 5. Double DQN 核心训练
# ==========================================
def train_dqn():
    env = GridWorld()
    env.generate_map()
    
    episodes = 2000     
    max_steps = 100
    batch_size = 128
    
    # ！！！核心修改 1 ！！！
    # 绝对不能写 1.0。0.95 对于 10x10 地图已经完全足够引导其走向终点，且保证数学上绝对收敛。
    gamma = 0.95  
    lr = 1e-3
    
    # 按“全局步数”更新，而不是按 episode
    target_update_freq_steps = 500 

    # ε 应该在一个合理的进度内降下来，比如前 60% 的时间用来降，后 40% 用来精调
    epsilon_initial = 1.0
    epsilon_min = 0.0

    q_net = QNetwork(GRID_SIZE, NUM_ACTIONS)
    target_net = QNetwork(GRID_SIZE, NUM_ACTIONS)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # Q值被控制住了，可以直接用更敏感的 MSE
    memory = ReplayBuffer()
    
    success_count = 0
    global_step = 0
    print(f"=== 开始 Double DQN 训练 | Gamma: {gamma} ===")
    
    for ep in range(episodes):
        # Exploring Starts
        while True:
            sx, sy = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if env.grid[sx, sy] == WHITE:
                state = (sx, sy)
                break
        
        # ！！！核心修改 2： ε 必须在训练结束前降到底，否则后期全在发疯 ！！！
        # 我们让它在前 60% 的 episode 衰减完毕
        progress = min(1.0, ep / (episodes * 0.6))
        epsilon = max(epsilon_min, epsilon_initial * ((1.0 - progress) ** 2))

        for step in range(max_steps):
            global_step += 1
            state_idx = get_state_idx(state)
            
            #选择动作：ε-greedy 策略
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor([state_idx], dtype=torch.long)
                    q_values = q_net(s_tensor)
                    action = q_values.argmax().item()
                    
            next_state, reward, done = env.step(state, action)
            
            # 超时截断
            if not done and step == max_steps - 1:
                reward = -5.0 
                
            next_state_idx = get_state_idx(next_state)
            memory.push(state_idx, action, reward, next_state_idx, done)
            state = next_state
            
            # === Double DQN 更新 ===
            if len(memory) > batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
                curr_q = q_net(states_b).gather(1, actions_b)
                
                # # ！！！核心修改 3: Double DQN ！！！
                # with torch.no_grad():
                #     # 1. 用当前的 q_net 选出下一个状态下最好的动作 (解耦)
                #     best_next_actions = q_net(next_states_b).argmax(dim=1, keepdim=True)
                #     # 2. 用 target_net 评估这个动作的价值
                #     max_next_q = target_net(next_states_b).gather(1, best_next_actions)
                #     target_q = rewards_b + gamma * max_next_q * (1.0 - dones_b)

                # 目标 Target = R + gamma * max_a( Q_target(s', a) )
                with torch.no_grad():
                    max_next_q = target_net(next_states_b).max(1)[0].unsqueeze(1)
                    target_q = rewards_b + gamma * max_next_q * (1.0 - dones_b)
                    
                loss = loss_fn(curr_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

            # 按全局步数平滑更新 Target Network
            if global_step % target_update_freq_steps == 0:
                target_net.load_state_dict(q_net.state_dict())

            # 行动结束
            if done:
                if env.grid[state[0], state[1]] == BLUE:
                    success_count += 1
                break
            
        if (ep + 1) % 50 == 0:
            success_rate = (success_count / 50.0) * 100
            print(f"Episode {ep+1:4d}/{episodes} | Epsilon: {epsilon:.4f} | "
                  f"Success Rate: {success_rate:5.1f}% | Buffer: {len(memory)}")
            success_count = 0

    return q_net, env

# ==========================================
# 6. 交互演示模式
# ==========================================
def evaluate_dqn(q_net, env):
    q_net.eval()
    try:
        while True:
            env.render((-1, -1))
            user_input = input(f"\n请输入起点坐标(如 '0 0')，输入 '-1 -1' 退出: ")
            parts = user_input.strip().split()
            if len(parts) != 2: continue
            startX, startY = int(parts[0]), int(parts[1])
            
            if startX == -1: break
            if startX < 0 or startX >= GRID_SIZE or startY < 0 or startY >= GRID_SIZE: continue
                
            state = (startX, startY)
            if env.grid[state[0], state[1]] == BLUE: continue
                
            done = False
            steps = 0
            
            while not done and steps < 30:
                env.render(state)
                
                state_idx = get_state_idx(state)
                with torch.no_grad():
                    s_tensor = torch.tensor([state_idx], dtype=torch.long)
                    q_values = q_net(s_tensor)[0]
                    action = q_values.argmax().item()
                    
                action_strs =["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
                print(f"Step: {steps} | Pos: ({state[0]}, {state[1]}) | Action: {action_strs[action]}")
                # 现在你能看到正常的 Q 值了（一般在 -20 到 +10 之间）
                print(f"Q-values: UP={q_values[0]:.1f}, DOWN={q_values[1]:.1f}, LEFT={q_values[2]:.1f}, RIGHT={q_values[3]:.1f}, STAY={q_values[4]:.1f}")
                
                state, reward, done = env.step(state, action)
                steps += 1
                time.sleep(0.3)
                
            env.render(state)
            if done and env.grid[state[0], state[1]] == BLUE:
                print("\n>>> 到达目标！ <<<")
            else:
                print("\n>>> 失败！ <<<")
            input("\n(按回车继续...)")
            
    except KeyboardInterrupt:
        print("\n\n>>> 退出程序。")

if __name__ == "__main__":
    trained_net, environment = train_dqn()
    evaluate_dqn(trained_net, environment)