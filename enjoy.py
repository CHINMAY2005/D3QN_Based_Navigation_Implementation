import os
import math
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Headless backend for GIF rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

from environment import RobotNavigationEnv
from dueling_dqn import VLAGuardedDuelingDQN
from vla_guard import VLAGuard

def get_sample_dataset_images(dataset_root="Datasets/MIT Indoor Scene Recognition.v5-resized416by416_70-20-10split.folder/valid", max_samples=30):
    """Fast sample image loader from valid split."""
    image_paths = []
    if os.path.exists(dataset_root):
        for sub in os.listdir(dataset_root)[:10]:
            sub_dir = os.path.join(dataset_root, sub)
            if os.path.isdir(sub_dir):
                files = [f for f in os.listdir(sub_dir) if f.lower().endswith(('.jpg', '.png'))]
                for f in files[:2]:
                    image_paths.append(os.path.join(sub_dir, f))
    return image_paths[:max_samples]

def render_frame(env, current_dist_to_goal, vla_token="OPEN_WAREHOUSE"):
    """
    Renders the current environment state into a PIL Image with live VLA Guard token overlay.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_title("VLA-Guarded D3QN Navigation & Live Token Generation", fontsize=11, fontweight='bold')
    
    # 1. Draw obstacles
    for ox, oy, r in env.obstacles:
        circle = patches.Circle((ox, oy), r, color='#4A4A4A', alpha=0.85)
        ax.add_patch(circle)
        
    # 2. Draw goal
    ax.plot(env.goal_x, env.goal_y, 'g*', markersize=14, label='Goal')
    
    # 3. Draw robot trajectory
    if len(env.trajectory) > 1:
        traj_x, traj_y = zip(*env.trajectory)
        ax.plot(traj_x, traj_y, 'k--', alpha=0.5, linewidth=1.5)
        
    # 4. Draw LiDAR scan beams
    scan = env._get_lidar_scan()
    angles = [env.robot_theta + j * (2 * math.pi / env.lidar_beams) for j in range(env.lidar_beams)]
    for angle, dist in zip(angles, scan):
        bx = env.robot_x + dist * math.cos(angle)
        by = env.robot_y + dist * math.sin(angle)
        color = '#FF3B30' if dist < env.collision_threshold * 2 else ('#FFCC00' if dist < 1.0 else '#34C759')
        ax.plot([env.robot_x, bx], [env.robot_y, by], color=color, alpha=0.25, linewidth=1)
        
    # 5. Draw robot body
    robot_body = patches.Circle((env.robot_x, env.robot_y), 0.20, color='#007AFF', zorder=5)
    ax.add_patch(robot_body)
    hx = env.robot_x + 0.35 * math.cos(env.robot_theta)
    hy = env.robot_y + 0.35 * math.sin(env.robot_theta)
    ax.plot([env.robot_x, hx], [env.robot_y, hy], color='#0051C6', linewidth=2.5, zorder=6)
    
    # 6. Live Token Badge Overlay
    badge_color = '#FF3B30' if vla_token == "HAZARDOUS_ZONE" else ('#FF9500' if vla_token == "CROWDED_ROOM" else '#34C759')
    
    overlay_text = (f"Step: {env.step_count:3d} | Goal Dist: {current_dist_to_goal:.2f}m\n"
                    f"Vel: v={env.v:.1f}m/s, w={env.omega:.1f}rad/s\n"
                    f"LIVE VLA TOKEN: [{vla_token}]")
                    
    ax.text(0.3, env.height - 0.7, 
            overlay_text, 
            fontsize=8.0, fontfamily='monospace', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=badge_color, linewidth=2, boxstyle='round,pad=0.4'))
            
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
    vla_guard = VLAGuard(semantic_dim=64, model_path="checkpoints/vla_vision_encoder.pth")
    dataset_images = get_sample_dataset_images()
    
    state_dim = 28
    action_dim = 5
    semantic_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLAGuardedDuelingDQN(state_dim, action_dim, semantic_dim).to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model.eval()
            print(f"Loaded trained VLA-Guarded model checkpoint from: {model_path}", flush=True)
        except Exception as e:
            print(f"WARNING: Could not load checkpoint ({e}). Running evaluation.", flush=True)
            model.eval()
    else:
        print(f"WARNING: Checkpoint {model_path} not found! Running evaluation.", flush=True)
        model.eval()
        
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    first_episode_frames = []
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        
        sample_img_path = random.choice(dataset_images) if dataset_images else None
        if sample_img_path:
            token, sem_emb = vla_guard.get_semantic_context_from_image(sample_img_path)
        else:
            min_lidar = np.min(state[:24])
            token, sem_emb = vla_guard.get_semantic_context(env.robot_x, env.robot_y, min_lidar)
            
        dist_to_goal = env._get_observation()[24]
        
        if ep == 1:
            first_episode_frames.append(render_frame(env, dist_to_goal, token))
            
        while not done:
            if sample_img_path and env.step_count % 10 == 0:
                sample_img_path = random.choice(dataset_images)
                token, sem_emb = vla_guard.get_semantic_context_from_image(sample_img_path)
            elif not sample_img_path:
                min_lidar = np.min(state[:24])
                token, sem_emb = vla_guard.get_semantic_context(env.robot_x, env.robot_y, min_lidar)
                
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            sem_t = torch.FloatTensor(sem_emb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = model(state_t, sem_t)
                action = torch.argmax(q_values).item()
                
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if ep == 1 and env.step_count % 3 == 0:
                first_episode_frames.append(render_frame(env, info["distance_to_goal"], token))
                
        # Statistics
        if info["success"]:
            success_count += 1
            status = "SUCCESS"
        elif info["collision"]:
            collision_count += 1
            status = "COLLISION"
        else:
            timeout_count += 1
            status = "TIMEOUT"
            
        print(f"Eval Episode {ep:2d} | Steps: {env.step_count:3d} | Reward: {episode_reward:7.2f} | Status: {status:9s} | Live VLA Token: [{token}]", flush=True)
        
    if len(first_episode_frames) > 0:
        gif_path = os.path.join(save_dir, "vla_guarded_demo.gif")
        first_episode_frames[0].save(
            gif_path,
            save_all=True,
            append_images=first_episode_frames[1:],
            duration=120,
            loop=0
        )
        print(f"Saved VLA-Guarded demonstration GIF with live context token overlays to {gif_path}", flush=True)
        
    print("\n--- Evaluation Summary ---", flush=True)
    print(f"Total Episodes Evaluated: {num_episodes}", flush=True)
    print(f"Success Rate:             {success_count / num_episodes:.2%}", flush=True)
    print(f"Collision Rate:           {collision_count / num_episodes:.2%}", flush=True)
    print(f"Timeout Rate:             {timeout_count / num_episodes:.2%}", flush=True)

if __name__ == "__main__":
    model_file = "checkpoints/best_model.pth"
    if not os.path.exists(model_file):
        model_file = "checkpoints/final_model.pth"
    evaluate_agent(model_path=model_file, num_episodes=5)
