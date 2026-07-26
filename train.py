import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from environment import RobotNavigationEnv
from agent import D3QNAgent
from vla_guard import VLAGuard

def train_agent(num_episodes=500, max_steps=250, save_dir="checkpoints", plot_dir="plots"):
    # Ensure directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize env, VLA Guard, and VLA-Guarded D3QN Agent
    env = RobotNavigationEnv(max_steps=max_steps)
    vla_guard = VLAGuard(semantic_dim=64)
    
    state_dim = 28
    action_dim = 5
    semantic_dim = 64
    
    agent = D3QNAgent(state_dim=state_dim, action_dim=action_dim, semantic_dim=semantic_dim, lr=1e-4, gamma=0.99)
    
    print(f"Starting VLA-Guarded D3QN Training on device: {agent.device}")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}, Semantic Vector Dim: {semantic_dim}")
    print(f"Parameters: Epsilon Start = {agent.epsilon:.2f}, Epsilon Min = {agent.epsilon_min:.2f}, Epsilon Decay = {agent.epsilon_decay:.5f}")
    
    # Metrics tracking
    episode_rewards = []
    episode_steps = []
    success_history = deque(maxlen=50)
    outcomes = []
    losses = []
    
    best_success_rate = 0.0
    best_avg_reward = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # Query VLA Guard for high-level semantic token and 64-dim modulation vector
            min_lidar = np.min(state[:24])
            token, sem_emb = vla_guard.get_semantic_context(env.robot_x, env.robot_y, min_lidar)
            
            # Select action conditioned on VLA semantic embedding vector
            action = agent.select_action(state, semantic_emb=sem_emb, evaluate=False)
            next_state, reward, done, info = env.step(action)
            
            # Store 6-tuple multi-modal transition (state, action, reward, next_state, done, sem_emb)
            agent.store_transition(state, action, reward, next_state, done, semantic_emb=sem_emb)
            loss = agent.train_step()
            
            if loss > 0:
                losses.append(loss)
                agent.soft_update_target_network(tau=0.005)
                
            state = next_state
            episode_reward += reward
            step_count += 1
            
        # Record outcome
        if info["success"]:
            success_history.append(1)
            outcomes.append("success")
        elif info["collision"]:
            success_history.append(0)
            outcomes.append("collision")
        else:
            success_history.append(0)
            outcomes.append("timeout")
            
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        
        current_success_rate = np.mean(success_history) if len(success_history) > 0 else 0.0
        avg_reward = np.mean(episode_rewards[-50:])
        
        # Log progress
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode:4d}/{num_episodes} | Steps: {step_count:3d} | Reward: {episode_reward:6.2f} | "
                  f"Avg Reward (last 50): {avg_reward:6.2f} | Success Rate (last 50): {current_success_rate:.2%} | "
                  f"Epsilon: {agent.epsilon:.4f}")
                  
        # Save checkpoints
        if episode >= 50:
            is_best_success = current_success_rate > best_success_rate
            is_same_success_better_reward = (abs(current_success_rate - best_success_rate) < 1e-5 and avg_reward > best_avg_reward)
            
            if is_best_success or is_same_success_better_reward:
                best_success_rate = max(best_success_rate, current_success_rate)
                best_avg_reward = max(best_avg_reward, avg_reward)
                checkpoint_path = os.path.join(save_dir, "best_model.pth")
                torch.save(agent.online_net.state_dict(), checkpoint_path)
                print(f"--> Saved NEW BEST VLA-Guarded model checkpoint to {checkpoint_path} (Success Rate: {best_success_rate:.2%}, Avg Reward: {best_avg_reward:.2f})")
                
    # Save final model
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(agent.online_net.state_dict(), final_path)
    print(f"Training completed. Final VLA-Guarded model saved to {final_path}")
    
    # Plotting results
    plot_training_results(episode_rewards, outcomes, losses, plot_dir)
    return agent

def plot_training_results(rewards, outcomes, losses, plot_dir):
    plt.figure(figsize=(15, 5))
    
    # Plot Rewards
    plt.subplot(1, 3, 1)
    plt.plot(rewards, color='blue', alpha=0.3, label='Episode Reward')
    if len(rewards) >= 50:
        running_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards)), running_avg, color='red', linewidth=2, label='50-Ep Running Avg')
    plt.title('VLA-Guarded Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot Success Rate
    plt.subplot(1, 3, 2)
    window = 50
    success_binary = [1 if o == "success" else 0 for o in outcomes]
    running_success_rate = []
    for i in range(len(success_binary)):
        start = max(0, i - window + 1)
        running_success_rate.append(np.mean(success_binary[start:i+1]))
        
    plt.plot(running_success_rate, color='green', linewidth=2)
    plt.title('VLA-Guarded Success Rate (Moving Avg 50)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 3, 3)
    if len(losses) > 100:
        step = max(1, len(losses) // 500)
        plt.plot(range(0, len(losses), step), losses[::step], color='purple', alpha=0.5)
    else:
        plt.plot(losses, color='purple', alpha=0.5)
    plt.title('VLA-Guarded Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training curve plots to {plot_path}")

if __name__ == "__main__":
    train_agent(num_episodes=500)
