import os
import csv
import numpy as np
import torch
from environment import RobotNavigationEnv
from agent import D3QNAgent
from dueling_dqn import DuelingDQN

def generate_experiment_data():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Run quick training log generation (100 episodes for demonstration & full CSV logging)
    env = RobotNavigationEnv(max_steps=250)
    agent = D3QNAgent(state_dim=28, action_dim=5, lr=1e-4, gamma=0.99)
    
    # Set seed for 100% deterministic reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_csv_path = os.path.join("plots", "training_metrics.csv")
    eval_csv_path = os.path.join("plots", "evaluation_metrics.csv")
    
    print(f"Generating empirical experiment CSV logs...")
    
    with open(train_csv_path, mode="w", newline="") as f_train:
        writer = csv.writer(f_train)
        writer.writerow(["Episode", "Steps", "Reward", "Success", "Collision", "Timeout", "Moving_Success_Rate_50Ep", "Epsilon", "Avg_Loss"])
        
        rewards_history = []
        success_history = []
        
        for ep in range(1, 501):
            state = env.reset()
            ep_reward = 0
            steps = 0
            done = False
            ep_losses = []
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss > 0:
                    ep_losses.append(loss)
                    agent.soft_update_target_network(tau=0.005)
                state = next_state
                ep_reward += reward
                steps += 1
                
            success = 1 if info["success"] else 0
            collision = 1 if info["collision"] else 0
            timeout = 1 if info["timeout"] else 0
            
            rewards_history.append(ep_reward)
            success_history.append(success)
            
            moving_acc = np.mean(success_history[-50:]) if len(success_history) >= 1 else 0.0
            avg_loss = np.mean(ep_losses) if len(ep_losses) > 0 else 0.0
            
            writer.writerow([ep, steps, round(ep_reward, 4), success, collision, timeout, round(moving_acc, 4), round(agent.epsilon, 4), round(avg_loss, 6)])
            
            if ep % 50 == 0:
                print(f"Logged Episode {ep}/500 | Reward: {ep_reward:.2f} | Success Rate (last 50): {moving_acc:.2%}")
                
    print(f"Saved training CSV to {train_csv_path}")
    
    # 2. Run Evaluation CSV export with trained model checkpoint
    best_model_path = os.path.join("checkpoints", "best_model.pth")
    model = DuelingDQN(28, 5)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        print(f"Loaded trained checkpoint: {best_model_path}")
    model.eval()
    
    with open(eval_csv_path, mode="w", newline="") as f_eval:
        writer = csv.writer(f_eval)
        writer.writerow(["Eval_Episode", "Steps", "Total_Reward", "Final_Dist_To_Goal_m", "Outcome", "Avg_Speed_m_s"])
        
        for ep in range(1, 21):
            state = env.reset()
            done = False
            ep_reward = 0
            speeds = []
            
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_vals = model(state_t)
                    action = torch.argmax(q_vals).item()
                    
                state, reward, done, info = env.step(action)
                ep_reward += reward
                speeds.append(env.v)
                
            outcome = "SUCCESS" if info["success"] else ("COLLISION" if info["collision"] else "TIMEOUT")
            avg_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
            final_dist = info["distance_to_goal"]
            
            writer.writerow([ep, env.step_count, round(ep_reward, 4), round(final_dist, 4), outcome, round(avg_speed, 4)])
            
    print(f"Saved evaluation CSV to {eval_csv_path}")

if __name__ == "__main__":
    generate_experiment_data()
