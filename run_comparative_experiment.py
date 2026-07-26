import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

from environment import RobotNavigationEnv
from vla_guard import VLAGuard

# =====================================================================
# 1. BASELINE D3QN NETWORK (Standard - Sensor Features Only)
# =====================================================================
class BaselineD3QN(nn.Module):
    def __init__(self, state_dim: int = 28, action_dim: int = 5):
        super(BaselineD3QN, self).__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


# =====================================================================
# 2. VLA-GUARDED D3QN NETWORK (Multi-Modal Semantic Fusion)
# =====================================================================
class VLAGuardedD3QN(nn.Module):
    def __init__(self, state_dim: int = 28, action_dim: int = 5, semantic_dim: int = 64):
        super(VLAGuardedD3QN, self).__init__()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.vla_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.fusion_network = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state: torch.Tensor, semantic_emb: torch.Tensor = None) -> torch.Tensor:
        if semantic_emb is None:
            semantic_emb = torch.zeros((state.shape[0], 64), device=state.device)
        sensor_feat = self.sensor_encoder(state)
        vla_feat = self.vla_encoder(semantic_emb)
        combined = torch.cat([sensor_feat, vla_feat], dim=-1)
        fused = self.fusion_network(combined)
        values = self.value_stream(fused)
        advantages = self.advantage_stream(fused)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


# =====================================================================
# 3. TRAINER & EVALUATOR FOR EXPERIMENTAL COMPARISON
# =====================================================================
def run_experiment(model_type="VLA_GUARDED", num_episodes=300, seed=42):
    # Set seeds for fair comparison
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = RobotNavigationEnv(max_steps=250)
    vla_guard = VLAGuard(semantic_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "BASELINE_D3QN":
        online_net = BaselineD3QN(28, 5).to(device)
        target_net = BaselineD3QN(28, 5).to(device)
    else:
        online_net = VLAGuardedD3QN(28, 5, 64).to(device)
        target_net = VLAGuardedD3QN(28, 5, 64).to(device)
        
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
    memory = deque(maxlen=100000)
    
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995
    batch_size = 64
    gamma = 0.99
    
    rewards_history = []
    success_history = deque(maxlen=50)
    moving_success_rates = []
    outcomes = []
    losses = []
    
    print(f"\n--- Running Training for: {model_type} ({num_episodes} Episodes) ---")
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0
        step_count = 0
        done = False
        ep_losses = []
        
        while not done:
            min_lidar = np.min(state[:24])
            token, sem_emb = vla_guard.get_semantic_context(env.robot_x, env.robot_y, min_lidar)
            
            # Action selection
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    if model_type == "BASELINE_D3QN":
                        q_vals = online_net(state_t)
                    else:
                        sem_t = torch.FloatTensor(sem_emb).unsqueeze(0).to(device)
                        q_vals = online_net(state_t, sem_t)
                    action = torch.argmax(q_vals).item()
                    
            next_state, reward, done, info = env.step(action)
            memory.append((state, action, reward, next_state, done, sem_emb))
            
            # Train step
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_sem = zip(*batch)
                
                b_states = torch.FloatTensor(np.array(b_states)).to(device)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                b_next_states = torch.FloatTensor(np.array(b_next_states)).to(device)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(device)
                
                if model_type == "BASELINE_D3QN":
                    curr_q = online_net(b_states).gather(1, b_actions)
                    with torch.no_grad():
                        best_next_a = online_net(b_next_states).argmax(dim=1, keepdim=True)
                        next_q = target_net(b_next_states).gather(1, best_next_a)
                        target_q = b_rewards + (1.0 - b_dones) * gamma * next_q
                else:
                    b_sem = torch.FloatTensor(np.array(b_sem)).to(device)
                    curr_q = online_net(b_states, b_sem).gather(1, b_actions)
                    with torch.no_grad():
                        best_next_a = online_net(b_next_states, b_sem).argmax(dim=1, keepdim=True)
                        next_q = target_net(b_next_states, b_sem).gather(1, best_next_a)
                        target_q = b_rewards + (1.0 - b_dones) * gamma * next_q
                        
                loss = nn.MSELoss()(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())
                
                # Soft sync target net
                for t_p, o_p in zip(target_net.parameters(), online_net.parameters()):
                    t_p.data.copy_(0.005 * o_p.data + 0.995 * t_p.data)
                    
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                
            state = next_state
            ep_reward += reward
            step_count += 1
            
        success = 1 if info["success"] else 0
        outcome = "success" if info["success"] else ("collision" if info["collision"] else "timeout")
        
        rewards_history.append(ep_reward)
        success_history.append(success)
        moving_acc = np.mean(success_history)
        moving_success_rates.append(moving_acc)
        outcomes.append(outcome)
        if len(ep_losses) > 0:
            losses.append(np.mean(ep_losses))
            
        if episode % 50 == 0 or episode == 1:
            print(f"[{model_type}] Ep {episode:3d}/{num_episodes} | Reward: {ep_reward:6.2f} | 50-Ep Success: {moving_acc:.2%} | Epsilon: {epsilon:.4f}")
            
    return {
        "rewards": rewards_history,
        "success_rates": moving_success_rates,
        "outcomes": outcomes,
        "losses": losses
    }


def main():
    os.makedirs("plots", exist_ok=True)
    num_episodes = 300
    
    # 1. Run Baseline D3QN
    baseline_results = run_experiment("BASELINE_D3QN", num_episodes=num_episodes, seed=42)
    
    # 2. Run VLA-Guarded D3QN
    vla_results = run_experiment("VLA_GUARDED", num_episodes=num_episodes, seed=42)
    
    # 3. Export Side-by-Side Comparison CSV
    csv_path = os.path.join("plots", "comparative_metrics.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Baseline_D3QN_Reward", "VLA_Guarded_Reward", 
                         "Baseline_Success_Rate", "VLA_Guarded_Success_Rate",
                         "Baseline_Outcome", "VLA_Guarded_Outcome"])
        for ep in range(num_episodes):
            writer.writerow([
                ep + 1,
                round(baseline_results["rewards"][ep], 4),
                round(vla_results["rewards"][ep], 4),
                round(baseline_results["success_rates"][ep], 4),
                round(vla_results["success_rates"][ep], 4),
                baseline_results["outcomes"][ep],
                vla_results["outcomes"][ep]
            ])
    print(f"\nSaved comparative experiment CSV telemetry to {csv_path}")
    
    # 4. Generate Comparative Analysis Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    # Plot 1: Rewards Comparison
    axes[0].plot(baseline_results["rewards"], color='gray', alpha=0.35, label='Baseline D3QN (Raw)')
    axes[0].plot(vla_results["rewards"], color='dodgerblue', alpha=0.35, label='VLA-Guarded D3QN (Raw)')
    
    b_avg = np.convolve(baseline_results["rewards"], np.ones(30)/30, mode='valid')
    v_avg = np.convolve(vla_results["rewards"], np.ones(30)/30, mode='valid')
    axes[0].plot(range(29, len(baseline_results["rewards"])), b_avg, color='darkred', linewidth=2, label='Baseline 30-Ep Avg')
    axes[0].plot(range(29, len(vla_results["rewards"])), v_avg, color='navy', linewidth=2.5, label='VLA-Guarded 30-Ep Avg')
    axes[0].set_title("Episode Reward Comparison", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Success Rate Comparison
    axes[1].plot(baseline_results["success_rates"], color='crimson', linewidth=2, linestyle='--', label='Baseline D3QN')
    axes[1].plot(vla_results["success_rates"], color='forestgreen', linewidth=2.5, label='VLA-Guarded D3QN')
    axes[1].set_title("Moving Success Rate (50-Episode Window)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: Loss Comparison
    axes[2].plot(baseline_results["losses"], color='orange', alpha=0.6, label='Baseline Loss')
    axes[2].plot(vla_results["losses"], color='purple', alpha=0.6, label='VLA-Guarded Loss')
    axes[2].set_title("Training Loss (MSE) Comparison", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Optimization Steps")
    axes[2].set_ylabel("MSE Loss")
    axes[2].legend()
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join("plots", "d3qn_vs_vla_guarded_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparative visualization plot to {plot_path}")
    
    # 5. Print Quantitative Summary Table
    b_succ_cnt = baseline_results["outcomes"].count("success")
    v_succ_cnt = vla_results["outcomes"].count("success")
    b_coll_cnt = baseline_results["outcomes"].count("collision")
    v_coll_cnt = vla_results["outcomes"].count("collision")
    b_to_cnt = baseline_results["outcomes"].count("timeout")
    v_to_cnt = vla_results["outcomes"].count("timeout")
    
    print("\n" + "="*70)
    print("        QUANTITATIVE EXPERIMENTAL COMPARISON ANALYTICS")
    print("="*70)
    print(f"{'Metric':<30} | {'Baseline D3QN':<16} | {'VLA-Guarded D3QN':<16}")
    print("-" * 70)
    print(f"{'Overall Success Count':<30} | {b_succ_cnt}/{num_episodes} ({b_succ_cnt/num_episodes:.1%})  | {v_succ_cnt}/{num_episodes} ({v_succ_cnt/num_episodes:.1%})")
    print(f"{'Collision Rate':<30} | {b_coll_cnt/num_episodes:.1%}             | {v_coll_cnt/num_episodes:.1%}")
    print(f"{'Timeout Rate':<30} | {b_to_cnt/num_episodes:.1%}             | {v_to_cnt/num_episodes:.1%}")
    print(f"{'Final 50-Ep Avg Reward':<30} | {np.mean(baseline_results['rewards'][-50:]):.2f}            | {np.mean(vla_results['rewards'][-50:]):.2f}")
    print(f"{'Final 50-Ep Success Rate':<30} | {baseline_results['success_rates'][-1]:.1%}            | {vla_results['success_rates'][-1]:.1%}")
    print("="*70)

if __name__ == "__main__":
    main()
