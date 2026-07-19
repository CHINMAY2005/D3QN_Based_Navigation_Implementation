import numpy as np
import math

class RobotNavigationEnv:
    """
    2D Continuous Simulation Environment for Robot Navigation.
    Implements a custom Gymnasium-like API.
    
    State space (dim 28):
      - 24 laser scan beams (max range 3.0m)
      - [distance_to_goal, heading_error_angle, current_linear_velocity, current_angular_velocity]
      
    Action space (dim 5):
      0: [0.2, 0.0]  # Move Straight Fast
      1: [0.1, 0.3]  # Turn Left Soft
      2: [0.1, -0.3] # Turn Right Soft
      3: [0.0, 0.5]  # Pivot Left Hard
      4: [0.0, -0.5] # Pivot Right Hard
    """
    def __init__(self, max_steps: int = 250):
        self.max_steps = max_steps
        self.dt = 0.1  # seconds per simulation step
        self.lidar_beams = 24
        self.lidar_max_range = 3.0
        self.collision_threshold = 0.2  # meters
        self.goal_threshold = 0.3  # meters
        
        # Room boundaries: X in [0, 10], Y in [0, 10]
        self.width = 10.0
        self.height = 10.0
        
        # Circular obstacles: (x, y, radius)
        self.obstacles = [
            (3.0, 3.0, 0.8),
            (7.0, 7.0, 1.0),
            (3.0, 7.0, 0.8),
            (7.0, 3.0, 0.8)
        ]
        
        self.action_map = {
            0: [0.2, 0.0],
            1: [0.1, 0.3],
            2: [0.1, -0.3],
            3: [0.0, 0.5],
            4: [0.0, -0.5]
        }
        
        self.reset()
        
    def reset(self):
        """
        Resets the robot position, goal position, and state tracking.
        """
        self.step_count = 0
        
        # Place robot in a safe initial position (e.g. near bottom-left/middle-left)
        while True:
            rx = np.random.uniform(0.5, 9.5)
            ry = np.random.uniform(0.5, 9.5)
            if self._is_safe(rx, ry, min_dist=1.2):
                self.robot_x = rx
                self.robot_y = ry
                break
                
        self.robot_theta = np.random.uniform(-math.pi, math.pi)
        self.v = 0.0
        self.omega = 0.0
        
        # Place goal at a reasonable distance from the robot
        while True:
            gx = np.random.uniform(0.5, 9.5)
            gy = np.random.uniform(0.5, 9.5)
            dist_to_robot = math.sqrt((gx - self.robot_x)**2 + (gy - self.robot_y)**2)
            if self._is_safe(gx, gy, min_dist=1.0) and dist_to_robot > 3.0:
                self.goal_x = gx
                self.goal_y = gy
                break
                
        self.prev_dist_to_goal = math.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)
        self.trajectory = [(self.robot_x, self.robot_y)]
        
        return self._get_observation()
        
    def _is_safe(self, x: float, y: float, min_dist: float) -> bool:
        """
        Checks if a coordinate is sufficiently far from all obstacles and boundaries.
        """
        if x < min_dist or x > self.width - min_dist or y < min_dist or y > self.height - min_dist:
            return False
            
        for ox, oy, r in self.obstacles:
            dist = math.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < r + min_dist:
                return False
        return True
        
    def _get_lidar_scan(self) -> np.ndarray:
        """
        Computes distances along 24 radial beams from the robot's position.
        """
        scan = np.full(self.lidar_beams, self.lidar_max_range)
        angles = [self.robot_theta + i * (2 * math.pi / self.lidar_beams) for i in range(self.lidar_beams)]
        
        for i, phi in enumerate(angles):
            min_dist = self.lidar_max_range
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)
            
            # --- Check intersection with walls ---
            # Wall x = 10
            if cos_p > 0:
                min_dist = min(min_dist, (self.width - self.robot_x) / cos_p)
            # Wall x = 0
            elif cos_p < 0:
                min_dist = min(min_dist, -self.robot_x / cos_p)
                
            # Wall y = 10
            if sin_p > 0:
                min_dist = min(min_dist, (self.height - self.robot_y) / sin_p)
            # Wall y = 0
            elif sin_p < 0:
                min_dist = min(min_dist, -self.robot_y / sin_p)
                
            # --- Check intersection with obstacles ---
            for ox, oy, r in self.obstacles:
                # Vector from robot to circle center
                vx = ox - self.robot_x
                vy = oy - self.robot_y
                
                # Projection of circle center onto ray
                projection = vx * cos_p + vy * sin_p
                if projection < 0:
                    continue  # circle is behind the ray
                    
                # Distance squared from circle center to projection point
                perp_dist_sq = (vx**2 + vy**2) - projection**2
                if perp_dist_sq > r**2:
                    continue  # ray misses circle
                    
                # Calculate intersection distance
                half_chord = math.sqrt(r**2 - perp_dist_sq)
                dist_intersect = projection - half_chord
                if 0 <= dist_intersect < min_dist:
                    min_dist = dist_intersect
                    
            scan[i] = min_dist
            
        return scan
        
    def _get_observation(self) -> np.ndarray:
        """
        Constructs the state vector [LiDAR (24), Goal Dist, Heading Error, Linear Vel, Angular Vel].
        """
        lidar = self._get_lidar_scan()
        
        # Calculate goal metrics
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = math.sqrt(dx**2 + dy**2)
        
        goal_heading = math.atan2(dy, dx)
        heading_error = goal_heading - self.robot_theta
        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        obs = np.zeros(28, dtype=np.float32)
        obs[:24] = lidar
        obs[24] = dist
        obs[25] = heading_error
        obs[26] = self.v
        obs[27] = self.omega
        
        return obs
        
    def step(self, action: int):
        self.step_count += 1
        
        # Retrieve action values
        self.v, self.omega = self.action_map[action]
        
        # Kinematic Update
        self.robot_theta += self.omega * self.dt
        self.robot_theta = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        
        self.robot_x += self.v * math.cos(self.robot_theta) * self.dt
        self.robot_y += self.v * math.sin(self.robot_theta) * self.dt
        
        self.trajectory.append((self.robot_x, self.robot_y))
        
        # Calculate observation and components for reward
        obs = self._get_observation()
        lidar_scan = obs[:24]
        current_dist_to_goal = obs[24]
        
        # Reward components
        reward = 0.0
        done = False
        info = {
            "success": False,
            "collision": False,
            "timeout": False,
            "distance_to_goal": current_dist_to_goal
        }
        
        # Check Collision
        min_lidar_dist = np.min(lidar_scan)
        out_of_bounds = (self.robot_x < 0.1 or self.robot_x > self.width - 0.1 or 
                         self.robot_y < 0.1 or self.robot_y > self.height - 0.1)
                         
        if min_lidar_dist < self.collision_threshold or out_of_bounds:
            reward += -10.0  # R_collision
            done = True
            info["collision"] = True
            
        # Check Goal Reached
        elif current_dist_to_goal < self.goal_threshold:
            reward += 10.0  # R_goal
            done = True
            info["success"] = True
            
        # Check Timeout
        elif self.step_count >= self.max_steps:
            done = True
            info["timeout"] = True
            
        # Progress Reward
        # R_progress = c * (d_t-1 - d_t)
        # Encourages minimizing the distance to the goal
        c_progress = 5.0
        r_progress = c_progress * (self.prev_dist_to_goal - current_dist_to_goal)
        reward += r_progress
        
        # Smoothness Reward
        # R_smoothness = -0.05 * |omega|
        # Penalizes erratic spinning
        r_smoothness = -0.05 * abs(self.omega)
        reward += r_smoothness
        
        # Save distance for next step calculation
        self.prev_dist_to_goal = current_dist_to_goal
        
        return obs, reward, done, info
