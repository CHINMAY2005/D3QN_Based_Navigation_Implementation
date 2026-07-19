import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

from environment import RobotNavigationEnv
from dueling_dqn import DuelingDQN

def render_frame(env, current_dist_to_goal):
    """
    Renders the current environment state into a PIL Image.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_title("D3QN Robot Navigation", fontsize=14, fontweight='bold')
    
    # 1. Draw obstacles
    for ox, oy, r in env.obstacles:
        circle = patches.Circle((ox, oy), r, color='#4A4A4A', alpha=0.85, label='Obstacle' if ox == env.obstacles[0][0] else "")
        ax.add_patch(circle)
        
    # 2. Draw goal
    ax.plot(env.goal_x, env.goal_y, 'g*', markersize=15, label='Goal')
    
    # 3. Draw robot trajectory
    if len(env.trajectory) > 1:
        traj_x, traj_y = zip(*env.trajectory)
        ax.plot(traj_x, traj_y, 'k--', alpha=0.5, linewidth=1.5, label='Trajectory')
        
    # 4. Draw LiDAR scan beams
    scan = env._get_lidar_scan()
    angles = [env.robot_theta + j * (2 * math.pi / env.lidar_beams) for j in range(env.lidar_beams)]
    for angle, dist in zip(angles, scan):
        bx = env.robot_x + dist * math.cos(angle)
        by = env.robot_y + dist * math.sin(angle)
        # Red if close collision, orange if cautious, light green if clear
        if dist < env.collision_threshold * 2:
            color = '#FF3B30'  # Red
        elif dist < 1.0:
            color = '#FFCC00'  # Yellow/Orange
        else:
            color = '#34C759'  # Green
        ax.plot([env.robot_x, bx], [env.robot_y, by], color=color, alpha=0.25, linewidth=1)
        
    # 5. Draw robot body and heading direction
    robot_body = patches.Circle((env.robot_x, env.robot_y), 0.20, color='#007AFF', zorder=5, label='Robot')
    ax.add_patch(robot_body)
    hx = env.robot_x + 0.35 * math.cos(env.robot_theta)
    hy = env.robot_y + 0.35 * math.sin(env.robot_theta)
    ax.plot([env.robot_x, hx], [env.robot_y, hy], color='#0051C6', linewidth=2.5, zorder=6)
    
    # 6. Info Overlay
    ax.text(0.3, env.height - 0.6, 
            f"Step: {env.step_count:3d} | Dist to Goal: {current_dist_to_goal:.2f}m\nVel: v={env.v:.1f}, w={env.omega:.1f}", 
            fontsize=9, fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#D1D1D6', boxstyle='round,pad=0.5'))
            
    ax.legend(loc='lower left', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    buf.close()
    return img

def evaluate_agent(model_path="checkpoints/best_model.pth", num_episodes=5, max_steps=250, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    env = RobotNavigationEnv(max_steps=max_steps)
    state_dim = 28
    action_dim = 5
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelingDQN(state_dim, action_dim).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded trained model checkpoint from: {model_path}")
    else:
        print(f"WARNING: Checkpoint {model_path} not found! Running with randomized weights.")
        model.eval()
        
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    first_episode_frames = []
    best_episode_frames = []
    has_saved_gif = False
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        frames = []
        episode_reward = 0
        
        # Render initial frame
        dist_to_goal = env._get_observation()[24]
        if ep == 1:
            first_episode_frames.append(render_frame(env, dist_to_goal))
        elif not has_saved_gif:
            frames.append(render_frame(env, dist_to_goal))
            
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_t)
                action = torch.argmax(q_values).item()
                
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Save frame for GIF generation
            if ep == 1:
                first_episode_frames.append(render_frame(env, info["distance_to_goal"]))
            elif not has_saved_gif:
                frames.append(render_frame(env, info["distance_to_goal"]))
                
        # Update statistics
        if info["success"]:
            success_count += 1
            status = "SUCCESS"
            # If this is the first success, keep these frames
            if not has_saved_gif:
                if ep == 1:
                    best_episode_frames = first_episode_frames
                else:
                    best_episode_frames = frames
                has_saved_gif = True
        elif info["collision"]:
            collision_count += 1
            status = "COLLISION"
        else:
            timeout_count += 1
            status = "TIMEOUT"
            
        print(f"Eval Episode {ep:2d} | Steps: {env.step_count:3d} | Reward: {episode_reward:7.2f} | Status: {status}")
        
    # Save the animated GIF
    gif_frames = best_episode_frames if has_saved_gif else first_episode_frames
    if len(gif_frames) > 0:
        gif_path = os.path.join(save_dir, "navigation_demo.gif")
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=100,  # 100ms per frame (10 FPS)
            loop=0
        )
        print(f"Saved navigation demonstration GIF to {gif_path}")

        
    print("\n--- Evaluation Summary ---")
    print(f"Total Episodes Evaluated: {num_episodes}")
    print(f"Success Rate:             {success_count / num_episodes:.2%}")
    print(f"Collision Rate:           {collision_count / num_episodes:.2%}")
    print(f"Timeout Rate:             {timeout_count / num_episodes:.2%}")

if __name__ == "__main__":
    # Evaluate model
    model_file = "checkpoints/best_model.pth"
    if not os.path.exists(model_file):
        model_file = "checkpoints/final_model.pth"
    evaluate_agent(model_path=model_file, num_episodes=5)
