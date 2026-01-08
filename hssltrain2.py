# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 20:06:27 2025

@author: wuli
"""

# -*- coding: utf-8 -*-
"""
hssl

@author: wuli
"""

# -*- coding: utf-8 -*-
"""
双数据集PPO完整版 - 每500代保存有效组合
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
import os
import time
from datetime import datetime
from collections import deque
from torch.distributions import Bernoulli
from joblib import Parallel, delayed, parallel_backend  # 使用joblib进行并行计算

# ================== 参数配置 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文件路径
DATA_PATHS = ["train_test.xlsx", "Ntrain_Ntest.xlsx"]
OUTPUT_DIR = "results_PPO"
FEATURE_CSV = os.path.join(OUTPUT_DIR, "selected_features.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练参数
EPISODES = 400000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 1e-4
CLIP_EPS = 0.03
PPO_EPOCHS = 3
MINI_BATCH_SIZE = 128
ENTROPY_COEF = 0.1
HISTORY_WINDOW = 1000

# 网络结构参数（动态设置）
FC1_DIM = 2048
FC2_DIM = 2048
FC3_DIM = 2048

# 奖励参数
MIN_FEATURES = 5
MAX_FEATURES = 25
FEATURE_PENALTY = 40
SAMPLE_PENALTY = 120
MIN_SAMPLES_PCT = 0.1
BASE_REWARD = 200
REWARD_SCALE = 100
INIT_BIAS = -6.8

# ================== 数据加载 ==================
def load_data(paths):
    datasets = []
    feature_dim = None
    for path in paths:
        df = pd.read_excel(path)
        time_col = df.iloc[:, 0].values.astype(np.float32)
        event_col = df.iloc[:, 1].values
        features = df.iloc[:, 2:].values.astype(np.float32)
        
        if feature_dim is None:
            feature_dim = features.shape[1]
        else:
            assert feature_dim == features.shape[1], "特征维度不一致！"
        
        datasets.append((time_col, event_col, features))
    return datasets, feature_dim

# ================== 网络结构 ==================
class ActorCritic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.STATE_DIM = feature_dim + 3  # 基础特征 + 统计特征
        self.ACTION_DIM = feature_dim
        
        self.shared_layers = nn.Sequential(
            nn.Linear(self.STATE_DIM, FC1_DIM),
            nn.LayerNorm(FC1_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(FC1_DIM, FC2_DIM),
            nn.LayerNorm(FC2_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(FC2_DIM, FC3_DIM),
            nn.LayerNorm(FC3_DIM),
            nn.ReLU(),
        )
        self.actor = nn.Linear(FC3_DIM, self.ACTION_DIM)
        self.critic = nn.Sequential(
            nn.Linear(FC3_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.shared_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)
        nn.init.normal_(self.actor.weight, std=0.01)
        nn.init.constant_(self.actor.bias, INIT_BIAS)
        nn.init.xavier_normal_(self.critic[-1].weight)

    def forward(self, x):
        x = self.shared_layers(x)
        action_logits = self.actor(x)
        action_probs = torch.sigmoid(action_logits / 1.5)
        state_value = self.critic(x).squeeze()
        return action_probs, state_value

# ================== PPO Agent ==================
class PPOAgent:
    def __init__(self, feature_dim):
        self.actor_critic = ActorCritic(feature_dim).to(device)
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=LR, weight_decay=1e-4)
        self.buffer = []
        self.recent_actions = deque(maxlen=100)

    def _add_state_features(self, base_state, action):
        selected_count = np.sum(action)
        return np.concatenate([
            base_state,
            [selected_count/MAX_FEATURES,
             max(0, MIN_FEATURES-selected_count)/MIN_FEATURES,
             max(0, selected_count-MAX_FEATURES)/self.actor_critic.ACTION_DIM]
        ])

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor)
        dist = Bernoulli(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.cpu().numpy().squeeze().astype(np.int64), log_prob.item(), value.item()

# ================== 生存分析工具 ==================
def safe_logrank(args):
    action, dataset = args
    time, event, features = dataset
    
    selected = (features @ action) > 0.5
    if np.sum(selected) < 3 or np.sum(~selected) < 3:
        return (1.0, False)
    
    try:
        result = logrank_test(time[selected], time[~selected], 
                            event_observed_A=event[selected], 
                            event_observed_B=event[~selected])
        return (result.p_value, True)
    except:
        return (1.0, False)

# ================== 奖励计算 ==================
def calculate_reward(action, datasets, agent, episode, bootstrap_samples):
    p_values = []
    valid_p = True
    num_features = np.sum(action)
    valid_features = MIN_FEATURES <= num_features <= MAX_FEATURES
    sample_penalties = 0
    
    # 使用joblib并行执行原始logrank检验
    args_list = [(action, dataset) for dataset in datasets]
    with parallel_backend('loky', n_jobs=-1):
        results = Parallel(verbose=0)(delayed(safe_logrank)(arg) for arg in args_list)
    
    # 处理原始结果（根据数据集类型给予不同奖励）
    dataset_rewards = []
    for i, (p, sample_valid) in enumerate(results):
        p_values.append(p)
        if not sample_valid:
            sample_penalties += 1
        
        # 训练组肺癌（第一个数据集）需要p<0.05
        if i == 0:
            if p < 0.05:
                reward = BASE_REWARD * np.sqrt(-np.log(max(p, 1e-20)))
            else:
                reward = -BASE_REWARD
        # Ntrain（第二个数据集）需要p>=0.05
        else:
            if p >= 0.05:
                reward = BASE_REWARD * np.sqrt(-np.log(max(1-p, 1e-20)))
            else:
                reward = -BASE_REWARD
        
        dataset_rewards.append(reward)
    
    # Bootstrap奖励计算
    bootstrap_reward = 0
    bootstrap_passes = []
    for i, samples in enumerate(bootstrap_samples):
        args_list = [(action, sample) for sample in samples]
        with parallel_backend('loky', n_jobs=-1):
            bootstrap_results = Parallel(verbose=0)(delayed(safe_logrank)(arg) for arg in args_list)
        
        pass_count = 0
        for p, valid in bootstrap_results:
            if valid:
                if i == 0 and p < 0.05:
                    pass_count += 1
                elif i == 1 and p >= 0.05:
                    pass_count += 1
        bootstrap_passes.append(pass_count)
        bootstrap_reward += (pass_count / 100) * 150
    
    # 特征惩罚
    feature_penalty = 0
    if num_features < MIN_FEATURES:
        gap = MIN_FEATURES - num_features
        feature_penalty += (gap ** 1.5) * FEATURE_PENALTY
    elif num_features > MAX_FEATURES:
        gap = num_features - MAX_FEATURES
        feature_penalty += (gap ** 1.5) * FEATURE_PENALTY
    
    # 总奖励计算（移除了重复惩罚部分）
    total_reward = (
        sum(dataset_rewards) 
        + bootstrap_reward
        - feature_penalty
        - sample_penalties * SAMPLE_PENALTY
    ) / REWARD_SCALE
    
    valid = valid_p and valid_features
    
    return (
        np.clip(total_reward, -100.0, 100.0),
        valid,
        p_values,
        num_features,
        bootstrap_passes
    )

# ================== 训练进度图保存函数 ==================
def save_training_progress(episode_logs, output_dir, episode):
    """每500代保存一次训练进度图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        
        log_df = pd.DataFrame(episode_logs)
        
        plt.figure(figsize=(18, 6))
        
        # 总损失图
        plt.subplot(1, 3, 1)
        plt.plot(log_df['episode'], log_df['total_loss'], label='总损失')
        plt.plot(log_df['episode'], log_df['value_loss'], label='价值损失')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        # 奖励图
        plt.subplot(1, 3, 2)
        plt.plot(log_df['episode'], log_df['reward'])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title("奖励变化")
        
        # 策略熵图
        plt.subplot(1, 3, 3)
        plt.plot(log_df['episode'], log_df['entropy'])
        plt.xlabel('Episode')
        plt.ylabel('Entropy')
        plt.title("策略熵")
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"training_progress_ep{episode}.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')  # 降低dpi减少文件大小
        plt.close()
        
        return plot_path
    except Exception as e:
        print(f"保存训练进度图失败: {str(e)}")
        return None

# ================== 训练流程 ==================
def train(datasets, feature_dim, bootstrap_samples):
    agent = PPOAgent(feature_dim)
    start_time = time.time()
    # +++ 添加模型加载功能 +++
    MODEL_PATH = "final_model.pth"
    if os.path.exists(MODEL_PATH):
        agent.actor_critic.load_state_dict(torch.load(MODEL_PATH))
        print(f"jiazaiwancheng✅ 加载已保存的模型权重: {MODEL_PATH}1111")
    # +++++++++++++++++++++++
    base_state = np.zeros(feature_dim, dtype=np.float32)
    state = agent._add_state_features(base_state, np.zeros(feature_dim))
    
    episode_logs = []
    recent_losses = []
    recent_value_losses = []
    recent_entropies = []
    valid_combinations_cache = []  # 有效组合缓存

    for episode in range(EPISODES):
        action, log_prob, value = agent.act(state)
        reward, valid, p_values, num_features, bootstrap_passes = calculate_reward(
            action, datasets, agent, episode, bootstrap_samples
        )
        agent.recent_actions.append(action)
        
        # 状态转移
        next_base = 0.9 * base_state + 0.1 * action.astype(np.float32)
        next_state = agent._add_state_features(next_base, action)
        next_val = agent.actor_critic(torch.FloatTensor(next_state).to(device))[1].item()
        
        # 存储经验
        agent.buffer.append((
            state.copy(),
            action.copy(),
            log_prob,
            value,
            reward,
            next_state.copy(),
            next_val,
            False
        ))
        
        # 更新状态
        base_state = next_base.copy()
        state = next_state.copy()
        
        # PPO更新逻辑
        if len(agent.buffer) >= BATCH_SIZE:
            states, actions, old_log_probs, values, rewards, next_states, next_vals, dones = zip(*agent.buffer)
            
            # 计算GAE
            advantages = []
            returns = []
            for i in range(len(rewards)):
                adv = rewards[i] + GAMMA * next_vals[i] - values[i]
                ret = rewards[i] + GAMMA * next_vals[i]
                advantages.append(adv)
                returns.append(ret)
            
            # 转换为张量
            states_t = torch.FloatTensor(np.array(states)).to(device)
            actions_t = torch.FloatTensor(np.array(actions)).to(device)
            old_log_probs_t = torch.FloatTensor(old_log_probs).to(device)
            returns_t = torch.FloatTensor(returns).to(device)
            advantages_tensor = torch.FloatTensor(advantages).to(device)
            advantages_t = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
            # 多轮更新
            epoch_losses = []
            epoch_value_losses = []
            epoch_entropy = []
            
            for _ in range(PPO_EPOCHS):
                perm = np.random.permutation(len(agent.buffer))
                for start in range(0, len(agent.buffer), MINI_BATCH_SIZE):
                    end = start + MINI_BATCH_SIZE
                    indices = perm[start:end]
                    
                    batch_states = states_t[indices]
                    batch_actions = actions_t[indices]
                    batch_old_log_probs = old_log_probs_t[indices]
                    batch_returns = returns_t[indices]
                    batch_advantages = advantages_t[indices]
                    
                    # 前向传播
                    action_probs, values = agent.actor_critic(batch_states)
                    dist = Bernoulli(probs=action_probs)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=1)
                    entropy = dist.entropy().mean()
                    
                    # 计算比率
                    ratio = (new_log_probs - batch_old_log_probs).exp()
                    
                    # 策略损失
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 价值损失
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # 总损失
                    loss = policy_loss + 0.5 * value_loss + ENTROPY_COEF * entropy
                    
                    # 反向传播
                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor_critic.parameters(), 1.0)
                    agent.optimizer.step()
                    
                    # 记录当前batch的损失
                    epoch_losses.append(loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_entropy.append(entropy.item())
            
            # 记录平均损失
            if epoch_losses:
                recent_losses.append(np.mean(epoch_losses))
                recent_value_losses.append(np.mean(epoch_value_losses))
                recent_entropies.append(np.mean(epoch_entropy))
            
            agent.buffer.clear()
        
        # 构建当前episode的日志记录
        current_log = {
            'episode': episode,
            'reward': reward * REWARD_SCALE,
            'total_loss': recent_losses[-1] if recent_losses else np.nan,
            'value_loss': recent_value_losses[-1] if recent_value_losses else np.nan,
            'entropy': recent_entropies[-1] if recent_entropies else np.nan,
            'num_features': num_features,
            'valid': valid
        }
        episode_logs.append(current_log)
        
        # 保存有效结果到缓存
        if valid:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            
            valid_combinations_cache.append({
                'episode': episode,
                'timestamp': timestamp,
                'reward': reward * REWARD_SCALE,
                'num_features': num_features,
                'p_values': p_values,
                'bootstrap_pass_1': bootstrap_passes[0],
                'bootstrap_pass_2': bootstrap_passes[1],
                'features': ','.join(map(str, np.where(action > 0.5)[0])),
            })

        # 定期输出
        if episode % 100 == 0:
            avg_reward = np.nanmean([log['reward'] for log in episode_logs[-100:]])
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            p_str = [f"{p:.4f}" for p in p_values]
            
            # 计算每秒处理量
            elapsed = max(1, time.time() - start_time)
            eps = episode / elapsed
            
            print(f"⏳ Episode: {episode:5d} | Eps/s: {eps:.2f} | Avg Reward: {avg_reward:8.1f} | "
                  f"特征数: {num_features:3.0f} | "
                  f"有效: {valid!s:5} | p值: {p_str} | 时间: {current_time}")

        # 每500代保存一次
        if episode % 500 == 0 and episode > 0:
            # +++ 添加模型保存功能 +++
            torch.save(agent.actor_critic.state_dict(), MODEL_PATH)
            print(f"💾 保存模型权重到: {MODEL_PATH}")
            # +++++++++++++++++++++++
            # 保存有效组合
            if valid_combinations_cache:
                save_df = pd.DataFrame(valid_combinations_cache)
                header = not os.path.exists(FEATURE_CSV)
                save_df.to_csv(FEATURE_CSV, mode='a', header=header, index=False)
                print(f"💾 保存{len(valid_combinations_cache)}个有效组合（Episode {episode}）")
                valid_combinations_cache.clear()
            
            # 保存训练日志
            log_df = pd.DataFrame(episode_logs)
            log_df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)
            
            # 绘制训练进度图（每500代一次）
            plot_path = save_training_progress(episode_logs, OUTPUT_DIR, episode)
            if plot_path:
                print(f"📊 训练进度图已保存: {plot_path}")

    # 训练结束后保存剩余缓存
    if valid_combinations_cache:
        save_df = pd.DataFrame(valid_combinations_cache)
        save_df.to_csv(FEATURE_CSV, mode='a', header=False, index=False)
        print(f"💾 保存最后{len(valid_combinations_cache)}个有效组合")

    # 保存最终模型
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(agent.actor_critic.state_dict(), final_model_path)
    pd.DataFrame(episode_logs).to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)
    print(f"✅ 训练完成！总耗时: {(time.time()-start_time)/3600:.2f} 小时")

# ================== Bootstrap抽样函数 ==================
def bootstrap_sample(dataset):
    """生成bootstrap样本 - 抽样量为原样本的0.3倍"""
    time, event, features = dataset
    n = len(time)
    sample_size = max(1, int(n * 0.3))  # 确保至少有一个样本
    indices = np.random.choice(n, size=sample_size, replace=True)
    return (time[indices], event[indices], features[indices])

if __name__ == "__main__":
    # 加载数据并获取特征维度
    datasets, feature_dim = load_data(DATA_PATHS)
    
    # 预生成Bootstrap样本
    bootstrap_samples = [
        [bootstrap_sample(datasets[0]) for _ in range(100)],  # 训练组肺癌
        [bootstrap_sample(datasets[1]) for _ in range(100)]   # Ntrain
    ]
    
    # 开始训练
    train(datasets, feature_dim, bootstrap_samples)