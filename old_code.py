import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
import os

# ==========================================
# 自动检测 GPU (CUDA / Apple MPS / CPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"当前使用的计算设备: {device}")

# ==========================================
# 1. 全局配置与环境设定
# ==========================================
GRID_SIZE = 25  # 根据你的测试调成了 10，随时可以改回 38
NUM_ACTIONS = 4

DX = [-1, 1, 0, 0, 0]
DY = [0, 0, -1, 1, 0]

WHITE, YELLOW, BLUE = 0, 1, 2

# ==========================================
# 2. DQN 神经网络模型 (带 Embedding，完美适应迷宫)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, grid_size, num_actions):
        super(QNetwork, self).__init__()
        # 【核心修正】: 使用 Embedding 层。这能让网络明确区分每一个具体的格子，
        # 获得类似你 C++ 多维数组 Q[x][y] 的“独立查表”能力，同时具备深度学习的泛化力。
        self.embedding = nn.Embedding(grid_size * grid_size, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state_idx):
        # state_idx 是一维的整数索引 (batch_size,)
        emb = self.embedding(state_idx)
        return self.fc(emb)

# ==========================================
# 3. 经验回放池 (Off-Policy 的核心)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # return (
        #     torch.tensor(states, dtype=torch.long),
        #     torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
        #     torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
        #     torch.tensor(next_states, dtype=torch.long),
        #     torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        # )
        # 【GPU修改点】: 采样出来的数据直接送到 device 上
        return (
            torch.tensor(states, dtype=torch.long).to(device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(next_states, dtype=torch.long).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
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
        
        # 【设定 2 & 3】: 严格按照你的数值，走白格子-1逼迫行动，黄-1000，蓝+1000，越界-1000
        self.r_white = -0.1  # 大地图上适当的小惩罚，防止原地乱绕
        self.r_yellow = -10.0
        self.r_blue = 100.0
        self.r_boundary = -10.0

    # (修剪逻辑与 C++ 一致，略过复杂连通性判断，直接填埋孤岛)
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

            num_yellows = int(GRID_SIZE * GRID_SIZE * 0.38)
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
            next_pos = current_pos # 越界不移动
        else:
            next_pos = (nx, ny)
            if self.grid[nx, ny] == YELLOW:
                reward = self.r_yellow
                # 【设定 4】: 踩入黄色禁区直接视为死亡，触发回合终止
                done = True
            elif self.grid[nx, ny] == BLUE:
                reward = self.r_blue
                # 到达终点终止
                done = True
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
# 5. DQN 核心训练逻辑 (Off-Policy Q-Learning)
# ==========================================
def train_dqn():
    env = GridWorld()
    env.generate_map()
    
    # 训练超参数
    episodes = 3000       
    max_steps = 200       # 地图大了，横跨对角线需要更多步，步数太少永远走不到终点
    batch_size = 128      # GPU 并行能力强，稍微加大 Batch 可以学得更平稳
    
    # 【设定 1】: Gamma = 1，完全追求全局最优解，不打折扣
    gamma = 0.97 
    lr = 1e-3
    target_update_freq = 10 
    target_update_freq_steps = 1000 # 【设定 5】: ε 的初始值与非线性衰减参数 k
    epsilon_initial = 1.0
    epsilon_min = 0.0
    k = 4 # 你设定的进度次方 k0 

    # q_net = QNetwork(GRID_SIZE, NUM_ACTIONS)
    # target_net = QNetwork(GRID_SIZE, NUM_ACTIONS)
    # target_net.load_state_dict(q_net.state_dict())

    # 【GPU修改点】: 将网络送到 device
    q_net = QNetwork(GRID_SIZE, NUM_ACTIONS).to(device)
    target_net = QNetwork(GRID_SIZE, NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    
    # 因为奖励设定到了 ±1000，为了防止神经网络梯度爆炸，使用 Huber Loss (SmoothL1Loss) 是最稳妥的
    loss_fn = nn.SmoothL1Loss() 
    memory = ReplayBuffer()
    
    success_count = 0
    global_step = 0
    print(f"=== 开始 DQN 训练 | GRID: {GRID_SIZE}x{GRID_SIZE} | Gamma: {gamma} ===")
    
    try:
        for ep in range(episodes):
            # 强制 Exploring Starts：随机出生在安全的白格子
            while True:
                sx, sy = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                if env.grid[sx, sy] == WHITE:
                    state = (sx, sy)
                    break
            
            # 【设定 5】: 非线性衰减公式 epsilon = epsilon_initial * (1 - progress)^k
            progress = min(1.0, ep / (episodes * 0.8))
            epsilon = max(epsilon_min, epsilon_initial * (1.0 - progress) ** k)
            # epsilon = 1.0

            for step in range(max_steps):
                global_step += 1
                state_idx = get_state_idx(state)
                
                # Epsilon-Greedy 动作选择
                if random.random() < epsilon:
                    action = random.randint(0, NUM_ACTIONS - 1)
                else:
                    with torch.no_grad():
                        # 输入为单个标量组成的张量 [1]
                        # s_tensor = torch.tensor([state_idx], dtype=torch.long)
                        # 【GPU修改点】: 前向传播时的输入张量也要放到 device
                        s_tensor = torch.tensor([state_idx], dtype=torch.long).to(device)
                        q_values = q_net(s_tensor)
                        action = q_values.argmax().item()
                        
                next_state, reward, done = env.step(state, action)
                
                # 【截断机制补充】: 耗尽步数未到达目标给予极大惩罚，逼迫它去冒险而不是原地苟活
                if not done and step == max_steps - 1:
                    reward = -100.0 
                    
                next_state_idx = get_state_idx(next_state)
                
                # 存入回放池
                memory.push(state_idx, action, reward, next_state_idx, done)
                # if len(memory) < memory.buffer.maxlen:
                #     # 才把新东西放进去
                #     memory.push(state_idx, action, reward, next_state_idx, done)

                state = next_state
                
                # === Off-Policy Q-Learning (DQN) 核心更新 ===
                if len(memory) > batch_size:
                    states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
                    
                    # 当前 Q(s, a)
                    curr_q = q_net(states_b).gather(1, actions_b)
                    
                    # 目标 Target = R + gamma * max_a( Q_target(s', a) )
                    with torch.no_grad():
                        max_next_q = target_net(next_states_b).max(1)[0].unsqueeze(1)
                        target_q = rewards_b + gamma * max_next_q * (1.0 - dones_b)
                        
                    loss = loss_fn(curr_q, target_q)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # 裁剪梯度，确保 ±1000 这样庞大的 TD Error 不会撕裂神经网络
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=8.0)
                    optimizer.step()
                # ============================================

                    # 按全局步数平滑更新 Target Network
                if global_step % target_update_freq_steps == 0:
                    target_net.load_state_dict(q_net.state_dict())

                if done:
                    if env.grid[state[0], state[1]] == BLUE:
                        success_count += 1
                    break

            # # 定期同步 Target Network
            # if ep % target_update_freq == 0:
            #     target_net.load_state_dict(q_net.state_dict())
                
            # 打印日志
            if (ep + 1) % 100 == 0:
                success_rate = (success_count / 100.0) * 100
                print(f"Episode {ep+1:5d}/{episodes} | Epsilon: {epsilon:.7f} | "
                    f"Success Rate: {success_rate:5.1f}% | Buffer: {len(memory)}")
                success_count = 0
    except KeyboardInterrupt:
            print("\n[检测到中断] 正在停止训练，准备进入演示模式...")
            time.sleep(1) # 给用户一点反应时间
    return q_net, env

# ==========================================
# 6. 交互演示模式
# ==========================================
def evaluate_dqn(q_net, env):
    print("\n>>> 训练结束，进入测试阶段！")
    
    q_net.eval()
    
    try:
        while True:
            env.render((-1, -1))
            user_input = input(f"\n请输入起点坐标(如 '0 0')，输入 '-1 -1' 退出: ")
            
            parts = user_input.strip().split()
            if len(parts) != 2: continue
            startX, startY = int(parts[0]), int(parts[1])
            
            if startX == -1: break
            if startX < 0 or startX >= GRID_SIZE or startY < 0 or startY >= GRID_SIZE:
                continue
                
            state = (startX, startY)
            if env.grid[state[0], state[1]] == BLUE:
                print("出生就在终点！")
                time.sleep(1)
                continue
                
            done = False
            steps = 0
            
            try:
                while not done and steps < 150:
                    env.render(state)
                    
                    state_idx = get_state_idx(state)
                    with torch.no_grad():
                        s_tensor = torch.tensor([state_idx], dtype=torch.long)
                        q_values = q_net(s_tensor)[0]
                        action = q_values.argmax().item()
                        
                    action_strs =["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
                    print(f"Step: {steps} | Pos: ({state[0]}, {state[1]}) | Action: {action_strs[action]}")
                    if NUM_ACTIONS == 5:
                        print(f"Q-values: UP={q_values[0]:.1f}, DOWN={q_values[1]:.1f}, LEFT={q_values[2]:.1f}, RIGHT={q_values[3]:.1f}, STAY={q_values[4]:.1f}")
                    elif NUM_ACTIONS == 4:
                        print(f"Q-values: UP={q_values[0]:.1f}, DOWN={q_values[1]:.1f}, LEFT={q_values[2]:.1f}, RIGHT={q_values[3]:.1f}")
                    
                    state, reward, done = env.step(state, action)
                    steps += 1
                    
                    time.sleep(0.3)
                    
                env.render(state)
                if done:
                    if env.grid[state[0], state[1]] == BLUE:
                        print("\n>>> 恭喜！DQN 成功控制智能体到达蓝色目标！ <<<")
                    elif env.grid[state[0], state[1]] == YELLOW:
                        print("\n>>> 惨死！你踩到了黄色陷阱，当场去世！ <<<")
                else:
                    print("\n>>> 失败：步数耗尽 (超时) <<<")
            
            except KeyboardInterrupt:
                print("\n[跳过] 已手动停止当前路径演示。")
            input("\n(按回车继续...)")
            
    except KeyboardInterrupt:
        print("\n\n>>> 退出程序。")

if __name__ == "__main__":
    trained_net, environment = train_dqn()
    input("\n(按回车继续...)")
    evaluate_dqn(trained_net, environment)