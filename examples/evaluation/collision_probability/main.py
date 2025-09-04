from abc import ABC
from abc import abstractmethod
import argparse
import math
import os
import shutil
import time
from collections import deque

import carla
import cv2
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

# FogSim imports
from fogsim import Env, NetworkConfig
from fogsim.handlers import BaseHandler

# Import PID controllers from separate module
from controller import VehiclePIDController, PIDLongitudinalController, PIDLateralController

def get_speed(vel):
    """
    Compute speed of a vehicle in Km/h.
    :param vel: velocity
    :return: speed as a float in Km/h
    """
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def generate_descriptive_filename(base_name, extension, sync_mode=False, scenario_type=None, unique_id=None, run_number=None):
    """
    Generate descriptive filename with mode, scenario type, and unique identifier.
    
    :param base_name: Base name for the file (e.g., 'trajectory', 'collision_prob')
    :param extension: File extension (e.g., 'png', 'pdf', 'csv')
    :param sync_mode: True for synchronous mode, False for async
    :param scenario_type: Type of scenario ('right_turn', 'left_turn', 'merge')
    :param unique_id: Unique identifier (timestamp or custom ID)
    :param run_number: Optional run number for multiple runs
    :return: Formatted filename string
    """
    mode_str = "sync" if sync_mode else "async"
    
    # Generate unique ID if not provided
    if unique_id is None:
        unique_id = int(time.time() * 1000) % 1000000  # Last 6 digits of timestamp
    
    # Build filename parts
    parts = [base_name]
    
    if scenario_type:
        parts.append(scenario_type)
    
    parts.append(mode_str)
    
    if run_number is not None:
        parts.append(f"run{run_number}")
    
    parts.append(str(unique_id))
    
    filename = "_".join(parts) + "." + extension.lstrip('.')
    return filename

# class CarlaWaypoint:
#     """Simple waypoint wrapper for controller.py compatibility"""
#     def __init__(self, x, y, angle):
#         self.transform = carla.Transform(
#             carla.Location(x=x, y=y, z=0),
#             carla.Rotation(yaw=math.degrees(angle))
#         )




# ========================= Waypoint Classes =========================

class Waypoint:
    """Simple waypoint representation"""
    def __init__(self, x, y, angle, speed=30.0):
        self.x = x
        self.y = y
        self.angle = angle  # in radians
        self.speed = speed  # target speed in km/h

# class PathManager:
#     """Manages waypoints and path following logic"""
#     def __init__(self, waypoints, lookahead_distance=3.0):
#         self.waypoints = waypoints
#         self.lookahead_distance = lookahead_distance
#         self.current_index = 0
#         self.completed = False
    
#     def get_target_waypoint(self, current_pos, sync_mode=False):
#         """
#         Get target waypoint based on current position and lookahead
#         :param current_pos: (x, y) tuple of current position
#         :param sync_mode: whether running in synchronous mode (affects lookahead)
#         :return: target Waypoint or None if path completed
#         """
#         print(f"[get_target_waypoint] Current pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), sync_mode: {sync_mode}")
#         print(f"[get_target_waypoint] Path completed: {self.completed}, current_index: {self.current_index}/{len(self.waypoints)}")
#         if self.completed or self.current_index >= len(self.waypoints):
#             print(f"[get_target_waypoint] Early exit - completed: {self.completed}, index >= waypoints: {self.current_index >= len(self.waypoints)}")
#             return None
        
#         # Find the closest waypoint to current position within a reasonable search range
#         # But ensure forward progress by not going backwards
#         search_range = min(20, len(self.waypoints) - self.current_index)
#         min_dist = float('inf')
#         closest_idx = self.current_index
        
#         print(f"[get_target_waypoint] Searching for closest waypoint in range [{self.current_index}, {self.current_index + search_range})")
        
#         # Check if we should advance past current waypoint first
#         if self.current_index < len(self.waypoints) - 1:
#             current_wp = self.waypoints[self.current_index]
#             current_dist = math.sqrt((current_wp.x - current_pos[0])**2 + (current_wp.y - current_pos[1])**2)
            
#             # If we're close enough to current waypoint (within 2 meters), advance to next
#             if current_dist < 2.0 and self.current_index < len(self.waypoints) - 1:
#                 closest_idx = self.current_index + 1
#                 print(f"[get_target_waypoint] Close to current waypoint ({current_dist:.2f}m), advancing to {closest_idx}")
#             else:
#                 # Search for closest waypoint, but prefer forward waypoints
#                 for i in range(self.current_index, self.current_index + search_range):
#                     wp = self.waypoints[i]
#                     dist = math.sqrt((wp.x - current_pos[0])**2 + (wp.y - current_pos[1])**2)
#                     # Add a small bias toward forward waypoints to prevent getting stuck
#                     if i > self.current_index:
#                         dist -= 0.5  # Slight preference for forward waypoints
#                     if dist < min_dist:
#                         min_dist = dist
#                         closest_idx = i
        
#         print(f"[get_target_waypoint] Closest waypoint: idx={closest_idx}, distance={min_dist:.2f}")
        
#         # Update current index to the closest waypoint we found
#         old_current_index = self.current_index
#         self.current_index = closest_idx
#         print(f"[get_target_waypoint] Updated current_index: {old_current_index} -> {self.current_index}")
        
#         # Calculate lookahead based on sync mode
#         # Key insight: sync mode generates dense waypoints (100Hz), async mode has sparse waypoints
#         if sync_mode:
#             # Dense waypoints: look further ahead (more waypoints for same distance)
#             # For 100Hz sync mode, roughly 10 waypoints per meter at typical speeds
#             lookahead_waypoints = max(5, min(50, int(self.lookahead_distance * 8)))  # 8 waypoints per meter, capped at 50
#             print(f"[get_target_waypoint] Sync mode: lookahead_distance={self.lookahead_distance}m -> {lookahead_waypoints} waypoints")
#         else:
#             # Sparse waypoints: look closer ahead (fewer waypoints for same distance)
#             # For async mode, roughly 1-2 waypoints per meter
#             lookahead_waypoints = max(2, int(self.lookahead_distance * 1.5))         # 1.5 waypoints per meter
#             print(f"[get_target_waypoint] Async mode: lookahead_distance={self.lookahead_distance}m -> {lookahead_waypoints} waypoints")
        
#         target_idx = min(self.current_index + lookahead_waypoints, len(self.waypoints) - 1)
#         print(f"[get_target_waypoint] Initial target_idx: {self.current_index} + {lookahead_waypoints} = {target_idx} (clamped to {len(self.waypoints) - 1})")
        
#         # Ensure we actually move forward, not stay at the same point
#         if target_idx == self.current_index and target_idx < len(self.waypoints) - 1:
#             target_idx = self.current_index + 1
#             print(f"[get_target_waypoint] Forced forward progress: target_idx -> {target_idx}")
        
#         # Check if path is completed
#         if self.current_index >= len(self.waypoints) - 5:  # Near end of path
#             final_wp = self.waypoints[-1]
#             final_dist = math.sqrt((final_wp.x - current_pos[0])**2 + (final_wp.y - current_pos[1])**2)
#             print(f"[get_target_waypoint] Near end check: final_dist={final_dist:.2f}, threshold=3.0")
#             if final_dist < 3.0:  # Close enough to final waypoint
#                 self.completed = True
#                 print(f"[get_target_waypoint] Path completed! Distance to final waypoint: {final_dist:.2f}")
#                 return None
        
#         target_wp = self.waypoints[target_idx]
#         print(f"[get_target_waypoint] Target waypoint: idx={target_idx}, pos=({target_wp.x:.2f}, {target_wp.y:.2f}), speed={target_wp.speed}")
#         return target_wp
    
#     def is_completed(self):
#         return self.completed

#worked for 100hz sync mode, not for 10hz async mode
class PathManager:
    """Manages waypoints and path following logic (metric lookahead)."""
    def __init__(self, waypoints, lookahead_distance=3.0):
        self.waypoints = waypoints            # list of objects with .x, .y, .speed
        self.lookahead_distance = float(lookahead_distance)
        self.current_index = 0
        self.completed = False

    @staticmethod
    def _dist(wp_a, wp_b):
        return math.hypot(wp_a.x - wp_b.x, wp_a.y - wp_b.y)

    def _advance_by_distance(self, start_idx, lookahead_m):
        """Return index >= start_idx whose cumulative arc length >= lookahead_m."""
        i = start_idx
        s = 0.0
        n = len(self.waypoints)
        while i < n - 1 and s < lookahead_m:
            s += self._dist(self.waypoints[i], self.waypoints[i+1])
            i += 1
        return i

    def get_target_waypoint(self, current_pos, sync_mode=False):
        """
        Get target Waypoint based on current position and *metric* lookahead.
        current_pos: (x, y)
        """
        import time 
        print(f"[get_target_waypoint] time: {time.time()} Current pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), sync_mode: {sync_mode}")
        print(f"[get_target_waypoint] Path completed: {self.completed}, current_index: {self.current_index}/{len(self.waypoints)}")

        if not self.waypoints:
            print("[get_target_waypoint] No waypoints.")
            return None

        if self.completed:
            # Hold the final waypoint as target
            final_wp = self.waypoints[-1]
            print(f"[get_target_waypoint] Completed; holding final waypoint at ({final_wp.x:.2f}, {final_wp.y:.2f})")
            return final_wp

        # ---- nearest forward waypoint (monotonic progress, no bias) ----
        search_range = min(200, len(self.waypoints) - self.current_index)
        min_dist = float('inf')
        closest_idx = self.current_index

        # If we're already close to the current index, allow a single-step advance
        if self.current_index < len(self.waypoints) - 1:
            cur_wp = self.waypoints[self.current_index]
            cur_d = math.hypot(cur_wp.x - current_pos[0], cur_wp.y - current_pos[1])
            if cur_d < 2.0:
                closest_idx = self.current_index + 1
                min_dist = cur_d

        if min_dist == float('inf'):  # do an actual search if we didn't advance above
            start = self.current_index
            end = min(len(self.waypoints), start + search_range)
            for i in range(start, end):
                wp = self.waypoints[i]
                d = math.hypot(wp.x - current_pos[0], wp.y - current_pos[1])
                if d < min_dist:
                    min_dist = d
                    closest_idx = i

        # Monotonic: never move backwards
        closest_idx = max(closest_idx, self.current_index)
        print(f"[get_target_waypoint] Closest waypoint: idx={closest_idx}, distance={min_dist:.2f}")

        old_idx = self.current_index
        self.current_index = closest_idx
        print(f"[get_target_waypoint] Updated current_index: {old_idx} -> {self.current_index}")

        # ---- metric lookahead (meters, not indices) ----
        # Optionally scale with speed if available; here we keep it fixed and ignore sync_mode density.
        lookahead_m = max(1.0, float(self.lookahead_distance))  # guard against tiny values
        target_idx = self._advance_by_distance(self.current_index, lookahead_m)
        target_idx = min(target_idx, len(self.waypoints) - 1)
        print(f"[get_target_waypoint] Metric lookahead: {lookahead_m:.2f} m -> target_idx={target_idx}")

        # ---- check completion (close to final) ----
        final_wp = self.waypoints[-1]
        final_dist = math.hypot(final_wp.x - current_pos[0], final_wp.y - current_pos[1])
        near_end = (len(self.waypoints) - 1) - self.current_index <= 5
        if near_end and final_dist < max(0.5, 0.3 * lookahead_m):
            self.completed = True
            print(f"[get_target_waypoint] Path completed! Distance to final waypoint: {final_dist:.2f}")
            return final_wp  # keep returning a stable target instead of None

        target_wp = self.waypoints[target_idx]
        print(f"[get_target_waypoint] Target waypoint: idx={target_idx}, pos=({target_wp.x:.2f}, {target_wp.y:.2f}), speed={target_wp.speed}")
        return target_wp

    def is_completed(self):
        return self.completed

# def _wrap(a):
#     while a <= -math.pi: a += 2*math.pi
#     while a >   math.pi: a -= 2*math.pi
#     return a

# class PathManager:
#     """
#     Robust path follower:
#       • Project ego onto polyline -> s_proj (arc length)
#       • Target at s_proj + Ld (Ld = L0 + kv*|v|)
#       • Optional forward-cone to keep target in front of vehicle
#       • Tight completion: remaining arc AND Euclidean distance must be small
#     Waypoint must have: .x, .y, (optional) .speed
#     """
#     def __init__(self, waypoints, lookahead_distance=None, L0=3.0, kv=0.6,
#                  cone_deg=70.0, base_window=60, expand_rounds=5, recovery_dist=1.0):
#         # if caller uses lookahead_distance, map it to L0
#         if lookahead_distance is not None:
#             L0 = lookahead_distance
#         self.wp = waypoints
#         self.N = len(self.wp)
#         self.L0 = float(L0)
#         self.kv = float(kv)
#         self.cone_cos = math.cos(math.radians(cone_deg))
#         self.base_window = int(base_window)
#         self.expand_rounds = int(expand_rounds)
#         self.recovery_dist = float(recovery_dist)

#         # cumulative arc-length at nodes: s[i]
#         self.s = [0.0]
#         for i in range(1, self.N):
#             dx = self.wp[i].x - self.wp[i-1].x
#             dy = self.wp[i].y - self.wp[i-1].y
#             self.s.append(self.s[-1] + math.hypot(dx, dy))
#         self.S_total = self.s[-1]

#         # progress guard in arc-length space (never go backwards)
#         self.s_min = 0.0
#         self.completed = False

#     # ---------- geometry helpers ----------
#     def _proj_to_seg(self, px, py, i):
#         """Project P onto segment i..i+1 -> (t in [0,1], qx, qy, dist)."""
#         x0, y0 = self.wp[i].x, self.wp[i].y
#         x1, y1 = self.wp[i+1].x, self.wp[i+1].y
#         vx, vy = (x1 - x0), (y1 - y0)
#         wx, wy = (px - x0), (py - y0)
#         vv = vx*vx + vy*vy
#         if vv <= 1e-12:
#             return 0.0, x0, y0, math.hypot(px - x0, py - y0)
#         t = max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
#         qx, qy = x0 + t*vx, y0 + t*vy
#         return t, qx, qy, math.hypot(px - qx, py - qy)

#     def _nearest_projection(self, px, py, start_seg):
#         """
#         Search forward segments with exponentially expanding window.
#         Returns (s_proj, seg_i, t, qx, qy, dist).
#         """
#         best_d = float('inf'); best = None
#         nseg = self.N - 1

#         window = self.base_window
#         seg0 = max(0, min(nseg - 1, start_seg))
#         for _ in range(self.expand_rounds):
#             i0 = max(0, seg0)
#             i1 = min(nseg - 1, seg0 + window)
#             for i in range(i0, i1 + 1):
#                 t, qx, qy, d = self._proj_to_seg(px, py, i)
#                 if d < best_d:
#                     s_proj = self.s[i] + t*(self.s[i+1] - self.s[i])
#                     best_d = d
#                     best = (s_proj, i, t, qx, qy, d)
#             if best_d <= 3.0 or i1 == nseg - 1:
#                 break
#             # expand and step a bit backward to catch misses
#             window *= 2
#             seg0 = max(0, seg0 - window // 4)

#         return best

#     def _global_projection(self, px, py):
#         """Full sweep for recovery. Returns (s_proj, seg_i, t, qx, qy, dist)."""
#         best_d = float('inf'); best = None
#         for i in range(self.N - 1):
#             t, qx, qy, d = self._proj_to_seg(px, py, i)
#             if d < best_d:
#                 s_proj = self.s[i] + t*(self.s[i+1] - self.s[i])
#                 best_d = d
#                 best = (s_proj, i, t, qx, qy, d)
#         return best

#     def _pose_at_s(self, s_target):
#         """Interpolate (x,y,yaw) at arc-length s_target."""
#         s_target = max(0.0, min(self.S_total, s_target))
#         # find i s.t. s[i] <= s_target <= s[i+1]
#         i = 0
#         # small scan is fine; swap to binary search if path is huge
#         while i < self.N - 2 and self.s[i+1] < s_target:
#             i += 1
#         ds = max(1e-9, self.s[i+1] - self.s[i])
#         t = (s_target - self.s[i]) / ds
#         x0, y0 = self.wp[i].x, self.wp[i].y
#         x1, y1 = self.wp[i+1].x, self.wp[i+1].y
#         x = x0 + t*(x1 - x0)
#         y = y0 + t*(y1 - y0)
#         yaw = math.atan2(y1 - y0, x1 - x0)
#         return x, y, yaw, i, t

#     def _in_forward_cone(self, px, py, psi, tx, ty):
#         hx, hy = math.cos(psi), math.sin(psi)
#         vx, vy = (tx - px), (ty - py)
#         vn = math.hypot(vx, vy)
#         if vn < 1e-6:
#             return True
#         return (vx*hx + vy*hy) / vn >= self.cone_cos

#     # ---------- main API ----------
#     def get_target_pose(self, current_pos, current_yaw=None, current_speed=None):
#         """
#         Returns (xt, yt, yaw_path, s_proj, s_target).
#         - current_yaw: if provided, enforce forward-cone on target
#         - current_speed: m/s for speed-aware lookahead; falls back to wp speed
#         """
#         px, py = current_pos

#         # completed -> hold final pose
#         if self.completed:
#             xf, yf = self.wp[-1].x, self.wp[-1].y
#             yawf = math.atan2(self.wp[-1].y - self.wp[-2].y,
#                               self.wp[-1].x - self.wp[-2].x)
#             print(f"[get_target_pose] Completed; holding final ({xf:.2f},{yf:.2f})")
#             return xf, yf, yawf, self.S_total, self.S_total

#         # choose starting segment near previous s_min
#         start_seg = 0
#         while start_seg < self.N - 1 and self.s[start_seg + 1] < self.s_min:
#             start_seg += 1

#         # projection (expand window)
#         proj = self._nearest_projection(px, py, start_seg)
#         s_proj, seg_i, t, qx, qy, d = proj

#         # recovery hatch if very far (handles your "stuck at same index" case)
#         if d > self.recovery_dist:
#             proj = self._global_projection(px, py)
#             s_proj, seg_i, t, qx, qy, d = proj
#             print(f"[get_target_pose] Global recovery: seg={seg_i}, d={d:.2f}")

#         # monotone progress in arc-length
#         if s_proj > self.s_min:
#             self.s_min = s_proj

#         # speed-aware lookahead
#         v = 0.0
#         if current_speed is not None:
#             v = float(current_speed)
#         elif hasattr(self.wp[min(seg_i + 1, self.N - 1)], "speed"):
#             try:
#                 v = float(getattr(self.wp[min(seg_i + 1, self.N - 1)], "speed"))
#             except Exception:
#                 v = 0.0
#         Ld = max(1.0, self.L0 + self.kv * abs(v))

#         s_tgt = min(self.S_total, s_proj + Ld)
#         xt, yt, yaw_t, _, _ = self._pose_at_s(s_tgt)

#         # optional forward-cone (helps in async/sparse)
#         if current_yaw is not None and not self._in_forward_cone(px, py, current_yaw, xt, yt):
#             for _ in range(40):
#                 s_tgt = min(self.S_total, s_tgt + 0.3 * Ld)
#                 xt, yt, yaw_t, _, _ = self._pose_at_s(s_tgt)
#                 if self._in_forward_cone(px, py, current_yaw, xt, yt) or s_tgt >= self.S_total:
#                     break

#         # tightened completion: BOTH arc-length and Euclidean distance must be small
#         remaining_s = self.S_total - s_proj
#         d_final = math.hypot(self.wp[-1].x - px, self.wp[-1].y - py)
#         if remaining_s < max(1.0, 0.5 * Ld) and d_final < max(1.0, 0.5 * Ld):
#             self.completed = True

#         # debug prints (helpful for your logs)
#         print(f"[get_target_pose] s_min={self.s_min:.2f}, s_proj={s_proj:.2f}, d={d:.2f}, "
#               f"Ld={Ld:.2f}, s_tgt={s_tgt:.2f}, tgt=({xt:.2f},{yt:.2f})")

#         return xt, yt, yaw_t, s_proj, s_tgt
#     def get_target_waypoint(self, current_pos, sync_mode=False, current_yaw=None, current_speed=None):
#         """
#         Backward-compatible wrapper for legacy code.
#         Ignores sync_mode (kept for signature). Uses arc-length projection logic via get_target_pose
#         and returns an existing waypoint object close to s_target.
#         """
#         xt, yt, yaw_path, s_proj, s_tgt = self.get_target_pose(
#             current_pos=current_pos,
#             current_yaw=current_yaw,
#             current_speed=current_speed
#         )
#         # Find an actual waypoint index near s_tgt to return (for .speed, etc.)
#         # Reuse the internal interpolator to get segment and param t
#         x, y, yaw, i, t = self._pose_at_s(s_tgt)
#         target_idx = i + (1 if t >= 0.5 else 0)
#         target_idx = max(0, min(self.N - 1, target_idx))
#         return self.wp[target_idx]
#     def is_completed(self):
#         return self.completed
 
# ========================= Configuration =========================

unprotected_right_turn_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000,  # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 15,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/unprotected_right_turn_collision.mp4',
        'fps': 100,
        'width': 640,
        'height': 480,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {
            'x': 4,
            'y': -90,
            'yaw': 0
        },
        'go_straight_ticks': 500,  # * 10ms = 5s
        'turn_ticks': 250,  # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': 0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {
            'x': 19,
            'y': 28,
            'yaw': 90
        },
        'go_straight_ticks': 400,  # * 10ms = 4s
        'turn_ticks': 200,  # * 10ms = 2s
        'after_turn_ticks': 350,  # * 10ms = 3.5s
        'throttle': {
            'straight': 0.52,
            'turn': 0.4,
            'after_turn': 0.5
        },
        'steer': {
            'straight': 0.0,
            'turn': 0.0,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {
            'x': 10,
            'y': 35
        },
        'fov': '90'
    },
    'save_options': {
        'save_video': False,
        'save_images': True,
    }
}

unprotected_left_turn_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000,  # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 20,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/unprotected_left_turn_collision.mp4',
        'fps': 100,
        'width': 640,
        'height': 480,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {
            'x': 7.5,
            'y': -90,
            'yaw': 0
        },
        'go_straight_ticks': 500,  # * 10ms = 5s
        'turn_ticks': 250,  # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.5,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': -0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {
            'x': 3.5,
            'y': 55,
            'yaw': 180
        },
        'go_straight_ticks': 300,
        'turn_ticks': 200,
        'after_turn_ticks': 450,
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.52
        },
        'steer': {
            'straight': 0.0,
            'turn': 0.0,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {
            'x': 10,
            'y': 35
        },
        'fov': '90'
    },
    'save_options': {
        'save_video': False,
        'save_images': True,
    }
}

opposite_direction_merge_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'fps': 100,
        'delta_seconds': 0.01,  # 100fps
        'prediction_steps': 8000,  # 100 frames, need to be divided to delta_k 
        'l_max': 40,
        'delta_k': 40,
        'emergency_brake_threshold': 1.1,
        'cautious_threshold': 0.0,
        'cautious_delta_k': 20,
        'tracker_type': 'ekf'
    },
    'trajectories': {
        'ego': './ego_trajectory.csv',
        'obstacle': './obstacle_trajectory.csv'
    },
    'video': {
        'filename': './bev_images/opposite_direction_merge_collision.mp4',
        'fps': 100,
        'width': 640,
        'height': 480,
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {
            'x': 4,
            'y': -90,
            'yaw': 0
        },
        'go_straight_ticks': 500,  # * 10ms = 5s
        'turn_ticks': 250,  # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': 0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {
            'x': 8,
            'y': 35,
            'yaw': 180
        },
        'go_straight_ticks': 100,
        'turn_ticks': 400,
        'after_turn_ticks': 350,
        'throttle': {
            'straight': 0.52,
            'turn': 0.45,
            'after_turn': 0.5
        },
        'steer': {
            'straight': 0.0,
            'turn': -0.4,
            'after_turn': 0.0
        }
    },
    'camera': {
        'height': 50,
        'offset': {
            'x': 10,
            'y': 35
        },
        'fov': '90'
    },
    'save_options': {
        'save_video': False,
        'save_images': True,
    }
}


class CollisionHandler(BaseHandler):
    """
    FogSim handler for collision avoidance scenario.
    
    Observation: Obstacle vehicle position (x, y, yaw in degrees)
    Action: Binary brake decision (0 = no brake, 1 = brake)
    """
    
    def __init__(self, config, output_dir, no_risk_eval=False):
        self.config = config
        self.output_dir = output_dir
        self.no_risk_eval = no_risk_eval
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.obstacle_vehicle = None
        self.cameras = {}  # Dictionary to store multiple cameras
        self.collision_sensor = None
        
        # Trackers
        self.obstacle_tracker = None
        self.ground_truth_tracker = None
        
        # State tracking
        self.tick = 0
        self.has_collided = False
        self.ego_trajectory = None
        self.current_delta_k = config['simulation']['delta_k']
        
        # Video/image capture
        self.video_writer = None
        self.frame_queues = {}  # Dictionary to store frame queues for each camera
        
        # Collision probability logging
        self.collision_prob_file = os.path.join(output_dir, 
            f'collision_probabilities_{config["simulation"]["l_max"]}_fogsim.csv')
        
        # PID Controller setup - will be initialized in launch() after vehicles are spawned
        self.ego_pid_controller = None
        self.obstacle_pid_controller = None
        
        # Path management
        self.ego_path_manager = None
        self.obstacle_path_manager = None
        
        # Phase control for obstacle vehicle (keep for backwards compatibility)
        self.obstacle_phase = 'straight'
        
        # Actual trajectory tracking
        self.ego_actual_trajectory = []
        self.obstacle_actual_trajectory = []
        
    def launch(self):
        """Initialize CARLA and spawn vehicles."""
        # Connect to CARLA
        self.client = carla.Client(self.config['simulation']['host'],
                                  self.config['simulation']['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town03")
        
        # Set synchronous mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        if self.config['simulation'].get('sync_mode', True):
            settings.fixed_delta_seconds = self.config['simulation']['delta_seconds']
        settings.synchronous_mode = self.config['simulation'].get('sync_mode', False)
        print("sync_mode",  self.config['simulation'].get('sync_mode', True))
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Setup spawn points
        ego_spawn_point = spawn_points[0]
        ego_spawn_point.location.x += self.config['ego_vehicle']['spawn_offset']['x']
        ego_spawn_point.location.y += self.config['ego_vehicle']['spawn_offset']['y']
        ego_spawn_point.rotation.yaw += self.config['ego_vehicle']['spawn_offset']['yaw']
        
        obstacle_spawn_point = spawn_points[1]
        obstacle_spawn_point.location.x = ego_spawn_point.location.x + \
            self.config['obstacle_vehicle']['spawn_offset']['x']
        obstacle_spawn_point.location.y = ego_spawn_point.location.y + \
            self.config['obstacle_vehicle']['spawn_offset']['y']
        obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + \
            self.config['obstacle_vehicle']['spawn_offset']['yaw']
        
        # Spawn vehicles
        ego_bp = blueprint_library.find(self.config['ego_vehicle']['model'])
        ego_bp.set_attribute('role_name', 'ego')
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_spawn_point)
        
        obstacle_bp = blueprint_library.find(self.config['obstacle_vehicle']['model'])
        obstacle_bp.set_attribute('role_name', 'obstacle')
        self.obstacle_vehicle = self.world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)
        
        # Initialize PID controllers after vehicles are spawned
                
        if self.config['simulation'].get('sync_mode', False):
            dt = self.config['simulation']['delta_seconds']
            # Original gains for sync mode
            lateral_args = {'K_P': 1.0, 'K_I': 0.0, 'K_D': 0.1, 'dt': dt}
            longitudinal_args = {'K_P': 1.0, 'K_I': 0.1, 'K_D': 0.0, 'dt': dt}
        else:
            dt = 0.15
            # Reduced gains for async mode stability
            lateral_args = {'K_P': 0.2, 'K_I': 0.0, 'K_D': 0.3, 'dt': dt}
            longitudinal_args = {'K_P': 0.3, 'K_I': 0.05, 'K_D': 0.05, 'dt': dt}
            
        # lateral_args = {'K_P': 1.0, 'K_I': 0.0, 'K_D': 0.1, 'dt': delta_second}
        # longitudinal_args = {'K_P': 1.0, 'K_I': 0.1, 'K_D': 0.0, 'dt': delta_second}
        self.ego_pid_controller = VehiclePIDController(self.ego_vehicle, lateral_args, longitudinal_args)
        self.obstacle_pid_controller = VehiclePIDController(self.obstacle_vehicle, lateral_args, longitudinal_args)
        
        # Setup single BEV camera above the intersection
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.config['video']['width']))
        camera_bp.set_attribute('image_size_y', str(self.config['video']['height']))
        camera_bp.set_attribute('fov', self.config['camera']['fov'])
        
        # BEV camera positioned above the scene, looking down
        bev_transform = carla.Transform(
            carla.Location(
                x=ego_spawn_point.location.x + self.config['camera']['offset']['x'],
                y=ego_spawn_point.location.y + self.config['camera']['offset']['y'],
                z=self.config['camera']['height']
            ),
            carla.Rotation(pitch=-90)  # Looking straight down
        )
        
        camera = self.world.spawn_actor(camera_bp, bev_transform, attach_to=None)
        self.cameras['bev'] = camera
        self.frame_queues['bev'] = []
        
        # Setup collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._collision_callback)
        
        # Setup camera callback for BEV camera
        self.cameras['bev'].listen(lambda image: self._camera_callback(image, 'bev'))
        
        # Generate and load trajectories if they don't exist
        if not os.path.exists(self.config['trajectories']['ego']) or not os.path.exists(self.config['trajectories']['obstacle']):
            print("Trajectories not found, generating them...")
            run_first_simulation(self.config, 
                               ego_trajectory_file=self.config['trajectories']['ego'],
                               obstacle_trajectory_file=self.config['trajectories']['obstacle'])
        
        # Load ego trajectory
        if os.path.exists(self.config['trajectories']['ego']):
            self.ego_trajectory = load_trajectory(self.config['trajectories']['ego'])
            print(f"Loaded ego trajectory with {len(self.ego_trajectory)} points")
        else:
            self.ego_trajectory = []
        

        # Load obstacle trajectory for waypoint generation
        obstacle_trajectory_points = []
        if os.path.exists(self.config['trajectories']['obstacle']):
            obstacle_trajectory_points = load_trajectory(self.config['trajectories']['obstacle'])
            print(f"Loaded obstacle trajectory with {len(obstacle_trajectory_points)} points")
        
        #draw waypoints
        # Visualize the loaded trajectories


        # Convert trajectories to waypoints for PID control
        self.ego_path_manager = self._create_path_manager_from_trajectory(
            self.ego_trajectory, lookahead_distance=5.0)
        self.obstacle_path_manager = self._create_path_manager_from_trajectory(
            obstacle_trajectory_points, lookahead_distance=3.0)
        
        # Initialize trackers
        if not self.no_risk_eval:
            if self.config['simulation']['tracker_type'] == 'ekf':
                self.obstacle_tracker = EKFObstacleTracker(
                    self.ego_vehicle, self.obstacle_vehicle,
                    dt=self.config['simulation']['delta_seconds'])
                self.ground_truth_tracker = EKFObstacleTracker(
                    self.ego_vehicle, self.obstacle_vehicle,
                    dt=self.config['simulation']['delta_seconds'])
            else:
                self.obstacle_tracker = KFObstacleTracker(
                    self.ego_vehicle, self.obstacle_vehicle,
                    dt=self.config['simulation']['delta_seconds'])
                self.ground_truth_tracker = KFObstacleTracker(
                    self.ego_vehicle, self.obstacle_vehicle,
                    dt=self.config['simulation']['delta_seconds'])
        else:
            self.obstacle_tracker = None
            self.ground_truth_tracker = None
        
        # Setup video writer if needed
        if self.config['save_options']['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.config['video']['filename'], fourcc, 
                self.config['video']['fps'],
                (self.config['video']['width'], self.config['video']['height']))
        
        # Initialize CSV file
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.collision_prob_file):
            with open(self.collision_prob_file, 'w') as f:
                f.write('timestamp,tick,delta_k,collision_probability,ground_truth_probability\n')
    
    def _create_path_manager_from_trajectory(self, trajectory_points, lookahead_distance=3.0):
        """Create a PathManager from trajectory points loaded from CSV."""
        if not trajectory_points:
            return None
            
        waypoints = []
        for point in trajectory_points:
            x, y, theta = point  # theta is already in radians from load_trajectory
            # Estimate speed based on distance to next point
            speed = 30.0  # Default speed in km/h
            waypoints.append(Waypoint(x, y, theta, speed))
        
        if waypoints:
            print(f"Created path manager with {len(waypoints)} waypoints")
            return PathManager(waypoints, lookahead_distance=lookahead_distance)
        else:
            return None
    
    def _collision_callback(self, event):
        """Handle collision events."""
        self.has_collided = True
    
    def _camera_callback(self, image, camera_name):
        """Handle camera frames."""
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_bgr = img_array[:, :, :3].copy()
        self.frame_queues[camera_name].append(frame_bgr)
    
    def set_states(self, states=None, action=None):
        """Apply action (brake decision) to ego vehicle using PID control."""
        if action is not None:
            # Action is binary: 0 = no brake, 1 = brake
            brake = int(action) if not isinstance(action, (list, np.ndarray)) else int(action[0])
            
            if brake:
                # Emergency brake - override PID control
                ego_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                self.ego_vehicle.apply_control(ego_control)
            else:
                # Use PID control for normal driving
                try:
                    self._apply_pid_control_ego()
                except Exception as e:
                    print(f"PID control failed, using basic control: {e}")
                    self._apply_basic_ego_control()
        else:
            # No action provided, use PID control
            try:
                self._apply_pid_control_ego()
            except Exception as e:
                print(f"PID control failed, using basic control: {e}")
                self._apply_basic_ego_control()
        
        # Apply PID control for obstacle vehicle (independent of ego action)
        try:
            self._apply_pid_control_obstacle()
        except Exception as e:
            print(f"Obstacle PID control failed, using basic control: {e}")
            self._apply_basic_obstacle_control()
    
    def _apply_pid_control_ego(self):
        """Apply PID control to ego vehicle based on waypoints."""
        if self.ego_pid_controller is None or self.ego_path_manager is None:
            # Path manager not initialized, use basic controls
            self._apply_basic_ego_control()
            return
            
        if self.ego_path_manager.is_completed():
            # Path completed, maintain current speed and direction
            control = carla.VehicleControl(throttle=0.2, brake=0.0, steer=0.0)
            self.ego_vehicle.apply_control(control)
            return
        
        # Get current vehicle state
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        current_speed = get_speed(velocity)
        
        # Create current state object
        class CurrentState:
            def __init__(self, x, y, angle):
                self.x = x
                self.y = y
                self.angle = angle
        
        current_state = CurrentState(
            transform.location.x,
            transform.location.y,
            math.radians(transform.rotation.yaw)
        )
        
        # Get target waypoint with sync mode awareness
        current_pos = (transform.location.x, transform.location.y)
        sync_mode = self.config['simulation'].get('sync_mode', False)
        target_waypoint = self.ego_path_manager.get_target_waypoint(current_pos, sync_mode)
        
        if target_waypoint is None:
            # No target waypoint available, use basic controls
            self._apply_basic_ego_control()
            return
        
        # Apply PID control
        try:
            # carla_waypoint = CarlaWaypoint(target_waypoint.x, target_waypoint.y, target_waypoint.angle)
            
            control = self.ego_pid_controller.run_step(
                current_speed,
                target_waypoint.speed,
                current_state,
                target_waypoint
            )
            self.ego_vehicle.apply_control(control)
        except Exception as e:
            print(f"Error in ego PID control: {e}")
            self._apply_basic_ego_control()
    
    def _apply_basic_ego_control(self):
        """Apply basic tick-based control to ego vehicle as fallback."""
        ego_control = carla.VehicleControl()
        if self.tick < self.config['ego_vehicle']['go_straight_ticks']:
            ego_control.throttle = self.config['ego_vehicle']['throttle']['straight']
            ego_control.steer = 0.0
        elif self.tick < (self.config['ego_vehicle']['go_straight_ticks'] + 
                         self.config['ego_vehicle']['turn_ticks']):
            ego_control.throttle = self.config['ego_vehicle']['throttle']['turn']
            ego_control.steer = self.config['ego_vehicle']['steer']['turn']
        else:
            ego_control.throttle = self.config['ego_vehicle']['throttle']['after_turn']
            ego_control.steer = 0.0
        
        self.ego_vehicle.apply_control(ego_control)
    
    def _apply_pid_control_obstacle(self):
        """Apply PID control to obstacle vehicle based on waypoints."""
        if self.obstacle_pid_controller is None or self.obstacle_path_manager is None:
            # Path manager not initialized, use basic controls
            self._apply_basic_obstacle_control()
            return
            
        if self.obstacle_path_manager.is_completed():
            # Path completed, maintain current speed and direction
            control = carla.VehicleControl(throttle=0.2, brake=0.0, steer=0.0)
            self.obstacle_vehicle.apply_control(control)
            return
        
        # Get current vehicle state
        transform = self.obstacle_vehicle.get_transform()
        velocity = self.obstacle_vehicle.get_velocity()
        current_speed = get_speed(velocity)
        
        # Create current state object
        class CurrentState:
            def __init__(self, x, y, angle):
                self.x = x
                self.y = y
                self.angle = angle
        
        current_state = CurrentState(
            transform.location.x,
            transform.location.y,
            math.radians(transform.rotation.yaw)
        )
        
        # Get target waypoint with sync mode awareness
        current_pos = (transform.location.x, transform.location.y)
        sync_mode = self.config['simulation'].get('sync_mode', False)
        target_waypoint = self.obstacle_path_manager.get_target_waypoint(current_pos, sync_mode)
        
        if target_waypoint is None:
            # No target waypoint available, use basic controls
            self._apply_basic_obstacle_control()
            return
        
        # Apply PID control
        try:
            # carla_waypoint = CarlaWaypoint(target_waypoint.x, target_waypoint.y, target_waypoint.angle)
            
            control = self.obstacle_pid_controller.run_step(
                current_speed,
                target_waypoint.speed,
                current_state,
                target_waypoint
            )
            self.obstacle_vehicle.apply_control(control)
        except Exception as e:
            print(f"Error in obstacle PID control: {e}")
            self._apply_basic_obstacle_control()
    
    def _apply_basic_obstacle_control(self):
        """Apply basic tick-based control to obstacle vehicle as fallback."""
        obstacle_control = carla.VehicleControl()
        if self.tick < self.config['obstacle_vehicle']['go_straight_ticks']:
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['straight']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['straight']
        elif self.tick < (self.config['obstacle_vehicle']['go_straight_ticks'] + 
                         self.config['obstacle_vehicle']['turn_ticks']):
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['turn']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['turn']
        else:
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['after_turn']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['after_turn']
        
        self.obstacle_vehicle.apply_control(obstacle_control)
    
    def get_states(self):
        """Get current obstacle position as observation."""
        obs_transform = self.obstacle_vehicle.get_transform()
        
        # Return obstacle position as observation
        observation = np.array([
            obs_transform.location.x,
            obs_transform.location.y,
            obs_transform.rotation.yaw  # in degrees
        ])
        
        return {
            'observation': observation,
            'tick': self.tick,
            'has_collided': self.has_collided
        }
    
    def step(self):
        """Step CARLA simulation forward."""
        self.world.tick()
        self.tick += 1
        
        # Track actual vehicle positions
        if self.ego_vehicle:
            ego_transform = self.ego_vehicle.get_transform()
            self.ego_actual_trajectory.append([
                ego_transform.location.x,
                ego_transform.location.y, 
                math.radians(ego_transform.rotation.yaw)
            ])
            
        if self.obstacle_vehicle:
            obs_transform = self.obstacle_vehicle.get_transform()
            self.obstacle_actual_trajectory.append([
                obs_transform.location.x,
                obs_transform.location.y,
                math.radians(obs_transform.rotation.yaw)
            ])
    
    def render(self, camera_name='bev'):
        """Process and return camera frames."""
        if camera_name in self.frame_queues and self.frame_queues[camera_name]:
            return self.frame_queues[camera_name][-1]
        return None
    
    def save_all_camera_frames(self, tick, output_dir):
        """Save BEV camera frame."""
        if 'bev' in self.frame_queues and self.frame_queues['bev']:
            frame_bgr = self.frame_queues['bev'][-1]
            bev_output_dir = os.path.join(output_dir, 'bev_images')
            os.makedirs(bev_output_dir, exist_ok=True)
            
            # Get scenario type from config
            scenario_type = self.config.get('scenario_type', 'unknown')
            sync_mode = self.config['simulation'].get('sync_mode', False)
            
            # Generate descriptive filename
            filename = generate_descriptive_filename(
                f'frame_{tick}',
                'png',
                sync_mode=sync_mode,
                scenario_type=scenario_type
            )
            
            cv2.imwrite(os.path.join(bev_output_dir, filename), frame_bgr)
    
    def get_latest_frames(self):
        """Get the latest BEV frame."""
        latest_frames = {}
        if 'bev' in self.frame_queues and self.frame_queues['bev']:
            latest_frames['bev'] = self.frame_queues['bev'][-1]
        return latest_frames
    
    def plot_trajectory_comparison(self, save_plots=True):
        """Plot reference vs actual trajectories for both vehicles."""
        if not save_plots:
            return
            
        output_dir = os.path.dirname(self.collision_prob_file)
        
        # Load reference trajectories
        ego_ref = []
        obstacle_ref = []
        
        if os.path.exists(self.config['trajectories']['ego']):
            ego_ref = load_trajectory(self.config['trajectories']['ego'])
        
        if os.path.exists(self.config['trajectories']['obstacle']):
            obstacle_ref = load_trajectory(self.config['trajectories']['obstacle'])
        
        # Get configuration parameters for plotting
        sync_mode = self.config['simulation'].get('sync_mode', False)
        scenario_type = self.config.get('scenario_type', None)
        
        # Plot individual vehicle comparisons
        if ego_ref or self.ego_actual_trajectory:
            ego_plot_path = os.path.join(output_dir, 'ego_trajectory_comparison.png')
            plot_trajectory_comparison(ego_ref, self.ego_actual_trajectory, 
                                     "Ego Vehicle", ego_plot_path,
                                     sync_mode=sync_mode,
                                     scenario_type=scenario_type)
        
        if obstacle_ref or self.obstacle_actual_trajectory:
            obs_plot_path = os.path.join(output_dir, 'obstacle_trajectory_comparison.png')
            plot_trajectory_comparison(obstacle_ref, self.obstacle_actual_trajectory, 
                                     "Obstacle Vehicle", obs_plot_path,
                                     sync_mode=sync_mode,
                                     scenario_type=scenario_type)
        
        # Plot dual vehicle comparison
        if any([ego_ref, self.ego_actual_trajectory, obstacle_ref, self.obstacle_actual_trajectory]):
            dual_plot_path = os.path.join(output_dir, 'dual_vehicle_trajectory_comparison.png')
            sync_mode = self.config['simulation'].get('sync_mode', False)
            scenario_type = self.config.get('scenario_type', None)
            plot_dual_vehicle_trajectories(ego_ref, self.ego_actual_trajectory,
                                         obstacle_ref, self.obstacle_actual_trajectory,
                                         dual_plot_path,
                                         sync_mode=sync_mode,
                                         scenario_type=scenario_type)
    
    def close(self):
        """Clean up CARLA resources."""
        # Plot trajectories before cleanup
        self.plot_trajectory_comparison(save_plots=True)
        
        if self.collision_sensor:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
        
        for camera in self.cameras.values():
            if camera:
                camera.stop()
                camera.destroy()
        
        if self.video_writer:
            self.video_writer.release()
        
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        
        if self.obstacle_vehicle:
            self.obstacle_vehicle.destroy()
        
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)
        
        # if self.client:
        #     self.client.reload_world()
    
    def get_extra(self):
        """Get extra metadata for collision probability calculation."""
        return {
            'obstacle_tracker': self.obstacle_tracker,
            'ground_truth_tracker': self.ground_truth_tracker,
            'ego_trajectory': self.ego_trajectory,
            'current_delta_k': self.current_delta_k,
            'collision_prob_file': self.collision_prob_file,
            'frame_queues': self.frame_queues,
            'video_writer': self.video_writer
        }
    
    def reset(self):
        """Reset the simulation and return initial observation."""
        # Reset state
        self.tick = 0
        self.has_collided = False
        if 'bev' in self.frame_queues:
            self.frame_queues['bev'] = []
        
        # Get initial observation
        if self.obstacle_vehicle:
            obs_transform = self.obstacle_vehicle.get_transform()
            observation = np.array([
                obs_transform.location.x,
                obs_transform.location.y,
                obs_transform.rotation.yaw
            ])
        else:
            observation = np.array([0.0, 0.0, 0.0])
        
        info = {
            'tick': 0,
            'has_collided': False
        }
        
        return observation, info
    
    def step_with_action(self, action):
        """
        Step the simulation with an action and return results.
        
        Returns:
            observation, reward, success, termination, timeout, info
        """
        # Apply action (brake decision)
        self.set_states(action=action)
        
        # Step the simulation
        self.step()
        
        # Get new state
        states = self.get_states()
        observation = states['observation']
        
        # Calculate reward (negative if collision, 0 otherwise)
        reward = -100.0 if self.has_collided else 0.0
        
        # Check termination conditions
        termination = self.has_collided
        timeout = self.tick >= (self.config['ego_vehicle']['go_straight_ticks'] +
                               self.config['ego_vehicle']['turn_ticks'] +
                               self.config['ego_vehicle']['after_turn_ticks'])
        success = not termination and timeout  # Success if completed without collision
        
        info = {
            'tick': self.tick,
            'has_collided': self.has_collided
        }
        
        return observation, reward, success, termination, timeout, info

class BaseTracker(ABC):

    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.05):
        # Get vehicle dimensions from CARLA
        ego_bbox = ego_vehicle.bounding_box
        obs_bbox = obstacle_vehicle.bounding_box

        # Store vehicle dimensions
        self.ego_length = ego_bbox.extent.x * 2
        self.ego_width = ego_bbox.extent.y * 2
        self.obs_length = obs_bbox.extent.x * 2
        self.obs_width = obs_bbox.extent.y * 2

        self.dt = dt
        self.history = []
        self.history_ticks = []

    @abstractmethod
    def update(self, state, tick_number):
        pass

    @abstractmethod
    def predict_future_position(self, steps_ahead):
        pass

    def calculate_collision_probability(self, ego_state, obstacle_state):
        """
        Calculate collision probability between ego vehicle and obstacle
        ego_state: [x, y, theta in radians]
        obstacle_state: [x, y, theta in radians]
        """
        # %%
        import matplotlib.pyplot as plt
        import numpy as np
        # %%
        from scipy.spatial import ConvexHull
        from scipy.spatial import distance
        # lo = 2
        # wo = 1
        # so = [-1.75, 2.0, 0.7853981633974483]
        # corner_dir = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
        # shape_matrix = np.array([[wo/2,0],[0,lo/2]])
        # # center_ori_matrix = np.array([ [np.cos(so[2]), np.sin(so[2]),so[0]],
        # #                       [np.sin(so[2]), np.cos(so[2]),so[1]],
        # #                       [0,0,1] ])
        # relative_rad = so[2]-np.pi/2
        # ori_matrix = np.array([ [np.cos(relative_rad), -np.sin(relative_rad)],
        #                     [np.sin(relative_rad), np.cos(relative_rad)] ])
        # print(ori_matrix)
        # center_matrix = np.array([so[0],so[1]])
        # print(np.matmul(ori_matrix,shape_matrix))
        # np.matmul(np.matmul(ori_matrix,shape_matrix),corner_dir.T) +np.tile(center_matrix,(4,1)).T
        # print(np.matmul(np.matmul(ori_matrix,shape_matrix),corner_dir.T) )
        # %%
        from sympy import Point
        from sympy import Polygon

        def Sort_List(l):
            # reverse = None (Sorts in Ascending order)
            # key is set to sort using second element of
            # sublist lambda has been used
            l.sort(key=lambda x: x[2])
            return l

        def collision_case(corner, w, l):
            x, y = corner
            d = 100
            close_point_list = [(x, l / 2), (x, -l / 2), (w / 2, y),
                                (w / 2, -y)]
            for pts in close_point_list:
                dis = distance.euclidean(corner, pts)
                if dis < d:
                    d = dis
                    close_point = pts
            return (close_point, -d)

        def collision_dis_dir(corner, w, l):
            #     print('corner',corner)
            closet_point = (0, 0)
            d = 100
            ego_corner = [(w / 2, l / 2), (w / 2, -l / 2), (-w / 2, l / 2),
                          (-w / 2, -l / 2)]
            cor_x, cor_y = corner

            if np.abs(cor_x) < w / 2 and np.abs(cor_y) < l / 2:
                closet_point, d = collision_case(corner, w, l)

            elif np.abs(cor_y) < l / 2 and np.abs(cor_x) >= w / 2:
                d = -w / 2 - cor_x if cor_x < -w / 2 else cor_x - w / 2
                closet_point = (-w / 2, cor_y) if cor_x < -w / 2 else (w / 2,
                                                                       cor_y)
            elif np.abs(cor_x) < w / 2 and np.abs(cor_y) >= l / 2:
                d = -l / 2 - cor_y if cor_y < -l / 2 else cor_y - l / 2
                closet_point = (cor_x, -l / 2) if cor_y < -l / 2 else (cor_x,
                                                                       l / 2)
            else:
                for ego_cor in ego_corner:
                    cor_dis = distance.euclidean(ego_cor, corner)
                    if cor_dis < d:
                        d = cor_dis
                        closet_point = ego_cor

        #     print(closet_point,d)
            return [closet_point, d]

        def object_tranformation(s1, s2):
            """
            kc's(orignal) frame {0}
            object s1's center as coordinate center {1}
            Transformation from {0} to {1} for object s2
            """
            relative_rad = np.pi / 2 - s1[2]
            R = np.array([[np.cos(relative_rad), -np.sin(relative_rad)],
                          [np.sin(relative_rad),
                           np.cos(relative_rad)]])
            #     print(R)
            # Obstacle coordinate transformation from {0} to {1}
            obs_center_homo = np.array([s2[0] - s1[0], s2[1] - s1[1]])
            obs_x, obs_y = np.matmul(R, obs_center_homo)
            obs_theta = s2[2] + relative_rad
            so_f1 = [obs_x, obs_y, obs_theta]
            return so_f1

        def point_transformation(s1, p):
            """
            transfer point p back to the orginal coordinate from coordinate frame s1
            """

            relative_rad = np.pi / 2 - s1[2]
            R = np.array([[np.cos(-relative_rad), -np.sin(-relative_rad)],
                          [np.sin(-relative_rad),
                           np.cos(-relative_rad)]])
            p = np.matmul(R, p)
            p = np.array([p[0] + s1[0], p[1] + s1[1]])
            return p

        def corners_cal(so_f1, lo, wo, corner_dir):
            obs_center_matrix = np.array([so_f1[0], so_f1[1]])
            shape_matrix = np.array([[wo / 2, 0], [0, lo / 2]])

            relative_rad = so_f1[2] - np.pi / 2
            #     print(relative_rad)
            ori_matrix = np.array(
                [[np.cos(relative_rad), -np.sin(relative_rad)],
                 [np.sin(relative_rad),
                  np.cos(relative_rad)]])

            obs_corners = np.matmul(np.matmul(ori_matrix, shape_matrix),
                                    corner_dir.T) + np.tile(
                                        obs_center_matrix, (4, 1)).T  #2*4
            return obs_corners

        def collision_point_rect(se,
                                 so,
                                 we=1.5,
                                 le=4,
                                 wo=1.4,
                                 lo=4,
                                 plot_flag=0):
            """
            Input:
            - ego vehicle's state se = (x_e,y_e,theta_e)  at time k and shape (w_e,l_e)
            - obstacle's state mean so = (x_o,y_o,theta_o) at time k and shape prior (w_o,l_o)

            Output:
            - collision point and collision direction
            """
            # check theta is in radians and with in -pi to pi
            if not isinstance(se[2], (int, float)) or not isinstance(
                    so[2], (int, float)):
                raise ValueError("Theta values must be numeric.")
            # if se[2] < -np.pi*2 or se[2] > np.pi*2 or so[2] < -np.pi*2 or so[2] > np.pi*2:
            #     #raise ValueError(f"Theta values {se[2]} and {so[2]} must be within -pi and pi.")
            #     print(f"Theta values {se[2]} and {so[2]} must be within -pi and pi.")

            corner_dir = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

            # Transfer matrix from kc's frame to ego center
            so_f1 = object_tranformation(se, so)

            # 4 obstacle corner point to ego vehicle distance
            obs_corners = corners_cal(so_f1, lo, wo, corner_dir)
            obs_corners = obs_corners.T
            closest_point_dis = [[tuple(corner)] +
                                 collision_dis_dir(corner, we, le)
                                 for corner in obs_corners]
            closest_point_dis = (Sort_List(closest_point_dis))
            # print('dis:',closest_point_dis[0][2])

            if plot_flag == 1:
                hull = ConvexHull(obs_corners)
                obs_corner_cov = obs_corners[hull.vertices]
                obs_corner_cov = np.append(obs_corner_cov, [obs_corner_cov[0]],
                                           axis=0)
                plt.plot(obs_corner_cov[:, 0],
                         obs_corner_cov[:, 1],
                         'b--',
                         lw=2)

                ego_corners = np.array([(we / 2, le / 2), (we / 2, -le / 2),
                                        (-we / 2, le / 2), (-we / 2, -le / 2)])
                hull = ConvexHull(ego_corners)
                ego_corners_cov = ego_corners[hull.vertices]
                ego_corners_cov = np.append(ego_corners_cov,
                                            [ego_corners_cov[0]],
                                            axis=0)
                plt.plot(ego_corners_cov[:, 0],
                         ego_corners_cov[:, 1],
                         'r--',
                         lw=2)

            # Transfer matrix from kc's frame to obstacle center frame
            se_f1 = object_tranformation(so, se)

            # 4 obstacle corner point to ego vehicle distance
            ego_corners = corners_cal(se_f1, le, we, corner_dir)
            ego_corners = ego_corners.T
            closest_point_dis2 = [[tuple(corner)] +
                                  collision_dis_dir(corner, wo, lo)
                                  for corner in ego_corners]
            closest_point_dis2 = (Sort_List(closest_point_dis2))
            #     print('dis:',closest_point_dis2[0][2])

            #     if closest_point_dis2[0][2] <0 or closest_point_dis[0][2] <0:
            #         print(se,so)

            # print(closest_point_dis2)

            if closest_point_dis[0][2] <= closest_point_dis2[0][2]:
                #transfer back to original coordinates closest_point_dis[0][1]
                obstacle_point = point_transformation(se,
                                                      closest_point_dis[0][0])
                ego_point = point_transformation(se, closest_point_dis[0][1])
                return (obstacle_point, ego_point, closest_point_dis[0][2])
            else:
                #transfer back to original coordinates closest_point_dis[0][1]
                obstacle_point = point_transformation(so,
                                                      closest_point_dis2[0][1])
                ego_point = point_transformation(so, closest_point_dis2[0][0])

                #define Matplotlib figure and axis

            #display plot

            return (obstacle_point, ego_point, closest_point_dis2[0][2])

        # %%
        from scipy.stats import norm

        def collision_probablity(V, P, dis, obs_cov):
            PV = V - P
            VP = P - V
            theta_pv = np.arccos(
                np.dot(PV, np.array([1, 0])) / np.linalg.norm(PV))
            R = np.array([[np.cos(theta_pv), -np.sin(theta_pv)],
                          [np.sin(theta_pv), np.cos(theta_pv)]])
            #     print(np.matmul(R, R.T))
            den = np.matmul(np.matmul(R, obs_cov), R.T)
            #     print(np.cos(theta_pv)* VP[0]+ np.sin(theta_pv) * VP[1])
            point_dis = np.cos(theta_pv) * VP[0] + np.sin(theta_pv) * VP[1]
            col_prob = norm.cdf(-dis / np.sqrt(den[0, 0]))

            return col_prob

        # Get collision points and distance
        obstacle_point, ego_point, distance = collision_point_rect(
            ego_state,
            obstacle_state,
            we=self.ego_width,
            le=self.ego_length,
            wo=self.obs_width,
            lo=self.obs_length)

        if distance == -1:  # Already colliding
            return 1.0

        # Calculate collision probability
        col_prob = collision_probablity(np.array(obstacle_point),
                                        np.array(ego_point), distance,
                                        self.obs_cov)

        return col_prob

    def calculate_collision_probability_with_trajectory(self,
                                                        ego_trajectory_point,
                                                        obstacle_state):
        return self.calculate_collision_probability(ego_trajectory_point,
                                                    obstacle_state)


class KFObstacleTracker(BaseTracker):

    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.05):
        super().__init__(ego_vehicle, obstacle_vehicle, dt)
        self.kf = self._initialize_kalman_filter()
        self.obs_cov = np.identity(2) * 0.04

    def _initialize_kalman_filter(self):
        # State: [x, y, theta, vx, vy, omega], Measurement: [x, y, theta]
        kf = KalmanFilter(dim_x=6, dim_z=3)

        # Use dt to construct F:
        kf.F = np.array([[1, 0, 0, self.dt, 0, 0], [0, 1, 0, 0, self.dt, 0],
                         [0, 0, 1, 0, 0, self.dt], [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # measure x
            [0, 1, 0, 0, 0, 0],  # measure y
            [0, 0, 1, 0, 0, 0]  # measure theta
        ])

        # Measurement noise
        kf.R = np.eye(3) * 0.1

        # Process noise
        kf.Q = np.eye(6) * 0.1

        # Initial state covariance
        kf.P *= 1000

        return kf

    def update(self, state, tick_number):
        """Update tracker with new state measurement (x, y, theta in radians)"""
        x, y, yaw_deg = state
        # Convert yaw to radians:
        theta = math.radians(yaw_deg)

        self.history.append((x, y, theta))
        self.history_ticks.append(tick_number)
        measurement = np.array([x, y, theta])
        self.kf.predict()
        self.kf.update(measurement)

    def predict_future_position(self, steps_ahead):
        """Predict future position using Kalman filter"""
        all_predicted_states = []
        state = self.kf.x.copy()
        for _ in range(steps_ahead):
            state = np.dot(self.kf.F, state)
            all_predicted_states.append([state[0][0], state[1][0], state[2][0]])
        return all_predicted_states


class EKFObstacleTracker(BaseTracker):

    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.05):
        super().__init__(ego_vehicle, obstacle_vehicle, dt)
        self.state = np.zeros((5, 1))  # [x, y, theta, v, omega]
        self.P = np.eye(5) * 1000  # Initial state covariance
        self.Q = np.eye(5) * 0.1  # Process noise
        self.R = np.eye(3) * 0.1  # Measurement noise
        self.obs_cov = np.identity(2) * 0.04

    def _f(self, x, dt):
        """State transition function"""
        F = np.array([
            [x[0, 0] + x[3, 0] * np.cos(x[2, 0]) * dt],  # x + v*cos(theta)*dt
            [x[1, 0] + x[3, 0] * np.sin(x[2, 0]) * dt],  # y + v*sin(theta)*dt
            [x[2, 0] + x[4, 0] * dt],  # theta + omega*dt
            [x[3, 0]],  # v
            [x[4, 0]]  # omega
        ])
        return F

    def _F(self, x, dt):
        """Jacobian of state transition function"""
        F = np.array(
            [[1, 0, -x[3, 0] * np.sin(x[2, 0]) * dt,
              np.cos(x[2, 0]) * dt, 0],
             [0, 1, x[3, 0] * np.cos(x[2, 0]) * dt,
              np.sin(x[2, 0]) * dt, 0], [0, 0, 1, 0, dt], [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]])
        return F

    def _h(self, x):
        """Measurement function"""
        H = np.array([
            [x[0, 0]],  # x
            [x[1, 0]],  # y
            [x[2, 0]]  # theta
        ])
        return H

    def _H(self, x):
        """Jacobian of measurement function"""
        H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        return H

    def update(self, state, tick_number):
        """Update tracker with new state measurement"""
        x, y, yaw_deg = state
        theta = math.radians(yaw_deg)

        self.history.append((x, y, theta))
        self.history_ticks.append(tick_number)

        # Prediction step
        x_pred = self._f(self.state, self.dt)
        F = self._F(self.state, self.dt)
        self.P = F @ self.P @ F.T + self.Q

        # Update step
        z = np.array([[x], [y], [theta]])
        H = self._H(x_pred)
        y = z - self._h(x_pred)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = x_pred + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    def predict_future_position(self, steps_ahead):
        """Predict future positions using EKF"""
        predicted_states = []
        current_state = self.state.copy()

        for _ in range(steps_ahead):
            current_state = self._f(current_state, self.dt)
            predicted_states.append([
                current_state[0, 0],  # x
                current_state[1, 0],  # y
                current_state[2, 0]  # theta
            ])

        return predicted_states


def save_trajectory(vehicle, filename):
    """Save vehicle transform to file"""
    with open(filename, 'a') as f:
        transform = vehicle.get_transform()
        f.write(
            f"{transform.location.x},{transform.location.y},{transform.rotation.yaw}\n"
        )


def load_trajectory(filename):
    """Load trajectory from file"""
    trajectory = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, yaw = map(float, line.strip().split(','))
            trajectory.append([x, y, math.radians(yaw)])
    return trajectory


def plot_trajectory_comparison(reference_trajectory, actual_trajectory, vehicle_name="Vehicle", output_path=None, show_arrows=True, arrow_spacing=10, sync_mode=False, scenario_type=None):
    """
    Plot reference trajectory vs actual vehicle movement.
    
    Args:
        reference_trajectory: List of [x, y, yaw] points from CSV (reference path)
        actual_trajectory: List of [x, y, yaw] points from actual simulation
        vehicle_name: Name for the vehicle (ego or obstacle)
        output_path: Path to save the plot (if None, shows the plot)
        show_arrows: Whether to show direction arrows
        arrow_spacing: Spacing between direction arrows (every N points)
        sync_mode: Whether running in synchronous mode
        scenario_type: Type of scenario being run
    """
    sns.set_context("talk")
    plt.figure(figsize=(12, 10))
    
    if reference_trajectory:
        ref_x = [point[0] for point in reference_trajectory]
        ref_y = [point[1] for point in reference_trajectory]
        ref_yaw = [point[2] for point in reference_trajectory]
        
        # Plot with flipped axes (y, x) and dotted line for reference
        plt.plot(ref_y, ref_x, 'b:', linewidth=2, label=f'{vehicle_name} Reference Trajectory', alpha=0.7)
        
        if show_arrows and len(reference_trajectory) > arrow_spacing:
            for i in range(0, len(reference_trajectory), arrow_spacing):
                # Flip the arrow directions for rotated plot
                dx = 2.0 * np.sin(ref_yaw[i])
                dy = 2.0 * np.cos(ref_yaw[i])
                plt.arrow(ref_y[i], ref_x[i], dx, dy, head_width=1.0, head_length=1.0, 
                         fc='blue', ec='blue', alpha=0.6)
    
    if actual_trajectory:
        actual_x = [point[0] for point in actual_trajectory]
        actual_y = [point[1] for point in actual_trajectory]
        actual_yaw = [point[2] for point in actual_trajectory]
        
        # Plot with flipped axes (y, x) and solid line for actual
        plt.plot(actual_y, actual_x, 'r-', linewidth=2, label=f'{vehicle_name} Actual Movement', alpha=0.8)
        
        if show_arrows and len(actual_trajectory) > arrow_spacing:
            for i in range(0, len(actual_trajectory), arrow_spacing):
                # Flip the arrow directions for rotated plot
                dx = 2.0 * np.sin(actual_yaw[i])
                dy = 2.0 * np.cos(actual_yaw[i])
                plt.arrow(actual_y[i], actual_x[i], dx, dy, head_width=1.0, head_length=1.0, 
                         fc='red', ec='red', alpha=0.6)
        
        # Mark start and end points with flipped axes
        plt.plot(actual_y[0], actual_x[0], 'go', markersize=8, label=f'{vehicle_name} Start')
        plt.plot(actual_y[-1], actual_x[-1], 'ro', markersize=8, label=f'{vehicle_name} End')
    
    plt.xlabel('Y Position (m)')
    plt.ylabel('X Position (m)')
    plt.title(f'{vehicle_name} Trajectory Comparison: Reference vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    all_x = []
    all_y = []
    if reference_trajectory:
        all_x.extend([point[0] for point in reference_trajectory])
        all_y.extend([point[1] for point in reference_trajectory])
    if actual_trajectory:
        all_x.extend([point[0] for point in actual_trajectory])
        all_y.extend([point[1] for point in actual_trajectory])
    
    if all_x and all_y:
        padding = 5  # meters of padding
        plt.xlim(min(all_y) - padding, max(all_y) + padding)
        plt.ylim(min(all_x) - padding, max(all_x) + padding)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    if output_path:
        # Generate descriptive filenames for both PNG and PDF
        if sync_mode is not None or scenario_type is not None:
            # Extract directory and base name from output_path
            dir_path = os.path.dirname(output_path)
            
            # Determine base name from vehicle name
            if "Ego" in vehicle_name:
                base_name = "ego_trajectory"
            elif "Obstacle" in vehicle_name:
                base_name = "obstacle_trajectory"
            else:
                base_name = "trajectory"
            
            # Generate PNG filename
            png_filename = generate_descriptive_filename(
                base_name,
                'png',
                sync_mode=sync_mode,
                scenario_type=scenario_type
            )
            png_path = os.path.join(dir_path, png_filename)
            
            # Generate PDF filename  
            pdf_filename = generate_descriptive_filename(
                base_name,
                'pdf',
                sync_mode=sync_mode,
                scenario_type=scenario_type
            )
            pdf_path = os.path.join(dir_path, pdf_filename)
            
            # Save both formats
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            print(f"{vehicle_name} trajectory plots saved to:")
            print(f"  PNG: {png_path}")
            print(f"  PDF: {pdf_path}")
        else:
            # Fallback to original behavior if no mode/scenario info
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dual_vehicle_trajectories(ego_ref, ego_actual, obs_ref, obs_actual, output_path=None, show_arrows=True, sync_mode=False, scenario_type=None, run_number=None):
    """
    Plot trajectories for both ego and obstacle vehicles in the same figure.
    
    Args:
        ego_ref: Reference trajectory for ego vehicle
        ego_actual: Actual trajectory for ego vehicle  
        obs_ref: Reference trajectory for obstacle vehicle
        obs_actual: Actual trajectory for obstacle vehicle
        output_path: Path to save the plot
        show_arrows: Whether to show direction arrows
    """
    sns.set_context("talk", font_scale=1.5)
    plt.figure(figsize=(15, 12))
    
    # Plot ego vehicle trajectories with flipped axes
    if ego_ref:
        ego_ref_x = [point[0] for point in ego_ref]
        ego_ref_y = [point[1] for point in ego_ref]
        # Dotted line for reference, flipped axes
        plt.plot(ego_ref_y, ego_ref_x, 'b:', linewidth=8, label='Ego Reference', alpha=0.2)
        
    if ego_actual:
        ego_actual_x = [point[0] for point in ego_actual]
        ego_actual_y = [point[1] for point in ego_actual]
        # Solid line for actual, flipped axes
        plt.plot(ego_actual_y, ego_actual_x, 'b-', linewidth=4, label='Ego Actual', alpha=0.6)
        plt.plot(ego_actual_y[0], ego_actual_x[0], 'bo', markersize=8, label='Ego Start')
        
    # Plot obstacle vehicle trajectories with flipped axes
    if obs_ref:
        obs_ref_x = [point[0] for point in obs_ref]
        obs_ref_y = [point[1] for point in obs_ref]
        # Dotted line for reference, flipped axes
        plt.plot(obs_ref_y, obs_ref_x, 'r:', linewidth=8, label='Obstacle Reference', alpha=0.2)
        
    if obs_actual:
        obs_actual_x = [point[0] for point in obs_actual]
        obs_actual_y = [point[1] for point in obs_actual]
        # Solid line for actual, flipped axes
        plt.plot(obs_actual_y, obs_actual_x, 'r-', linewidth=4, label='Obstacle Actual', alpha=0.6)
        plt.plot(obs_actual_y[0], obs_actual_x[0], 'ro', markersize=8, label='Obstacle Start')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    # plt.legend()
    plt.grid(True, alpha=0.3)

    # if all_x and all_y:
    plt.xlim(-180, -110)
    plt.ylim(-30, 20)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    if output_path:
        # Generate descriptive filenames for both PNG and PDF
        if sync_mode is not None or scenario_type is not None:
            # Extract directory and base name from output_path
            dir_path = os.path.dirname(output_path)
            base_name = "dual_vehicle_trajectory"
            
            # Generate PNG filename
            png_filename = generate_descriptive_filename(
                base_name,
                'png',
                sync_mode=sync_mode,
                scenario_type=scenario_type,
                run_number=run_number
            )
            png_path = os.path.join(dir_path, png_filename)
            
            # Generate PDF filename  
            pdf_filename = generate_descriptive_filename(
                base_name,
                'pdf',
                sync_mode=sync_mode,
                scenario_type=scenario_type,
                run_number=run_number
            )
            pdf_path = os.path.join(dir_path, pdf_filename)
            
            # Save both formats
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            print(f"Dual vehicle trajectory plots saved to:")
            print(f"  PNG: {png_path}")
            print(f"  PDF: {pdf_path}")
        else:
            # Fallback to original behavior if no mode/scenario info
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.savefig(output_path.replace("png", "pdf"), dpi=300, bbox_inches='tight')
            print(f"Dual vehicle trajectory plot saved to {output_path}")
    else:
        plt.show()
        
    plt.close()


def run_first_simulation(config, ego_trajectory_file=None, obstacle_trajectory_file=None):
    """Run the first simulation to generate both ego and obstacle vehicle trajectories"""
    # Use trajectory files from config if none provided
    if ego_trajectory_file is None:
        ego_trajectory_file = config['trajectories']['ego']
    if obstacle_trajectory_file is None:
        obstacle_trajectory_file = config['trajectories']['obstacle']

    client = carla.Client(config['simulation']['host'],
                          config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Always use synchronous mode for consistent waypoint collection
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    
    print(f"Using synchronous mode for trajectory collection (delta_seconds={settings.fixed_delta_seconds})")

    blueprint_library = world.get_blueprint_library()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Setup ego vehicle spawn point
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    # Setup obstacle vehicle spawn point
    obstacle_spawn_point = spawn_points[1]
    obstacle_spawn_point.location.x = ego_spawn_point.location.x + config['obstacle_vehicle']['spawn_offset']['x']
    obstacle_spawn_point.location.y = ego_spawn_point.location.y + config['obstacle_vehicle']['spawn_offset']['y']
    obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle']['spawn_offset']['yaw']

    # Spawn both vehicles
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    ego_vehicle.set_autopilot(False)
    
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)
    obstacle_vehicle.set_autopilot(False)

    # Clear existing trajectory files
    if os.path.exists(ego_trajectory_file):
        os.remove(ego_trajectory_file)
    if os.path.exists(obstacle_trajectory_file):
        os.remove(obstacle_trajectory_file)

    try:
        # Use ORIGINAL tick-based control to record accurate waypoints for collision scenarios
        total_ticks = (config['ego_vehicle']['go_straight_ticks'] +
                       config['ego_vehicle']['turn_ticks'] +
                       config['ego_vehicle']['after_turn_ticks'])
        
        for tick in range(total_ticks):
            world.tick()
            
            # Apply ORIGINAL tick-based controls to ego vehicle
            ego_control = carla.VehicleControl()
            if tick < config['ego_vehicle']['go_straight_ticks']:
                # Initial straight phase
                ego_control.throttle = config['ego_vehicle']['throttle']['straight']
                ego_control.steer = 0.0
            elif tick < config['ego_vehicle']['go_straight_ticks'] + config['ego_vehicle']['turn_ticks']:
                # Turning phase
                ego_control.throttle = config['ego_vehicle']['throttle']['turn']
                ego_control.steer = config['ego_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                ego_control.throttle = config['ego_vehicle']['throttle']['after_turn']
                ego_control.steer = 0.0

            ego_vehicle.apply_control(ego_control)
            
            # Apply ORIGINAL tick-based controls to obstacle vehicle
            obstacle_control = carla.VehicleControl()
            if tick < config['obstacle_vehicle']['go_straight_ticks']:
                # Initial straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['straight']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['straight']
            elif tick < config['obstacle_vehicle']['go_straight_ticks'] + config['obstacle_vehicle']['turn_ticks']:
                # Turning phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['after_turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['after_turn']

            obstacle_vehicle.apply_control(obstacle_control)
            
            # Save both trajectories every tick
            save_trajectory(ego_vehicle, ego_trajectory_file)
            save_trajectory(obstacle_vehicle, obstacle_trajectory_file)

        print(f"Ego trajectory saved to {ego_trajectory_file}")
        print(f"Obstacle trajectory saved to {obstacle_trajectory_file}")

    finally:
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()
        client.reload_world()
        world.apply_settings(original_settings)


def run_adaptive_simulation_fogsim(config, output_dir, no_risk_eval=False):
    """
    Run simulation with FogSim handling network delays.
    
    This version uses FogSim to simulate latency between observation and action,
    replacing the manual buffering approach with proper network simulation.
    """
    # Initialize return variables
    has_collided = False
    current_delta_k = config['simulation']['delta_k']
    
    try:
        # Create network configuration based on delta_k
        network_delay = config['simulation']['delta_k'] * config['simulation']['delta_seconds']
        network_config = NetworkConfig(
            source_rate=1e6,  # 1 Mbps
            topology={'link_delay': network_delay}
        )
        
        # Create handler
        handler = CollisionHandler(config, output_dir, no_risk_eval=no_risk_eval)
        
        # Create FogSim environment
        print(f"Creating FogSim with network delay: {network_delay}s")
        env = Env(
            handler=handler,
            network_config=network_config,
            enable_network=True,
            timestep=config['simulation']['delta_seconds']
        )
        print(f"FogSim mode: {env.fogsim.mode}")
        print(f"FogSim network initialized: {env.fogsim.network is not None}")
        
        # Initialize environment
        print("Initializing FogSim environment...")
        obs, info = env.reset()
        print(f"Initial observation: {obs}")
        
        # Run first simulation to generate trajectories if they don't exist
        if not os.path.exists(config['trajectories']['ego']) or not os.path.exists(config['trajectories']['obstacle']):
            print("Generating trajectories...")
            run_first_simulation(config, 
                               ego_trajectory_file=config['trajectories']['ego'],
                               obstacle_trajectory_file=config['trajectories']['obstacle'])
        
        # Reload ego trajectory in handler
        if os.path.exists(config['trajectories']['ego']):
            handler.ego_trajectory = load_trajectory(config['trajectories']['ego'])
        
        # Initialize variables for tracking
        max_collision_prob = 0.0
        ground_truth_collision_prob = 0.0
        tick = 0
        
        #     # Attach camera
#     camera_bp = blueprint_library.find('sensor.camera.rgb')
#     camera_bp.set_attribute('image_size_x', str(config['video']['width']))
#     camera_bp.set_attribute('image_size_y', str(config['video']['height']))
#     camera_bp.set_attribute('fov', config['camera']['fov'])

#     camera_transform = carla.Transform(
#         carla.Location(
#             x=ego_spawn_point.location.x + config['camera']['offset']['x'],
#             y=ego_spawn_point.location.y + config['camera']['offset']['y'],
#             z=config['camera']['height']), carla.Rotation(pitch=-90))
#     camera = world.spawn_actor(camera_bp, camera_transform, attach_to=None)

        
        # Observation buffer for delayed tracking
        observation_buffer = []
        
        # Total simulation steps
        total_steps = (config['ego_vehicle']['go_straight_ticks'] +
                      config['ego_vehicle']['turn_ticks'] +
                      config['ego_vehicle']['after_turn_ticks'])
        
        print(f"Starting simulation with {total_steps} steps...")
        for step in range(total_steps):
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}")
            
            # Get current observation (delayed by network)
            current_obs = obs  # This is already delayed by FogSim
            
            # Buffer observations
            observation_buffer.append(current_obs)
            if len(observation_buffer) > config['simulation']['l_max']:
                observation_buffer.pop(0)
            
            # Update trackers with current and historical observations
            if not no_risk_eval and len(observation_buffer) >= config['simulation']['l_max']:
                # Update with delayed observation (l_max steps ago)
                historical_obs = observation_buffer[0]
                handler.obstacle_tracker.update(
                    (historical_obs[0], historical_obs[1], historical_obs[2]), tick)
                
                # Update ground truth with current observation
                handler.ground_truth_tracker.update(
                    (current_obs[0], current_obs[1], current_obs[2]), tick)
                
                # Calculate collision probabilities
                predicted_positions = handler.obstacle_tracker.predict_future_position(
                    int(config['simulation']['prediction_steps'] / current_delta_k))
                
                max_collision_prob, collision_time, collision_probabilities = \
                    calculate_collision_probabilities(
                        handler.obstacle_tracker, predicted_positions,
                        handler.ego_trajectory, tick)
                
                # Ground truth predictions
                ground_truth_predictions = handler.ground_truth_tracker.predict_future_position(
                    int(config['simulation']['prediction_steps'] / current_delta_k))
                
                ground_truth_max_prob, _, _ = calculate_collision_probabilities(
                    handler.ground_truth_tracker, ground_truth_predictions,
                    handler.ego_trajectory, tick)
                
                ground_truth_collision_prob = ground_truth_max_prob
                
                # Adaptive behavior based on collision probability
                if max_collision_prob > config['simulation']['emergency_brake_threshold']:
                    # Emergency brake
                    action = 1  # Brake
                    print(f"Emergency brake activated! Collision probability: {max_collision_prob:.4f}")
                    
                elif max_collision_prob > config['simulation']['cautious_threshold']:
                    # Increase tracking frequency (decrease delta_k)
                    new_delta_k = config['simulation']['cautious_delta_k']
                    if new_delta_k != current_delta_k:
                        print(f"Adjusting delta_k from {current_delta_k} to {new_delta_k}")
                        current_delta_k = new_delta_k
                        # Update network delay
                        new_delay = new_delta_k * config['simulation']['delta_seconds']
                        env.fogsim.network.link_delay = new_delay
                        # Adjust buffer
                        while len(observation_buffer) > new_delta_k:
                            observation_buffer.pop(0)
                    action = 0  # No brake
                else:
                    action = 0  # No brake
            else:
                action = 0  # No brake (either during initial buffering or when risk eval is disabled)
            
            # Step environment with action
            obs, reward, success, termination, timeout, info = env.step(action)
            
            # Log network delay info periodically
            if step % 50 == 0 and 'network_delay_active' in info:
                print(f"  Network active: {info.get('network_delay_active', False)}, "
                      f"Sim time: {info.get('simulation_time', 0):.3f}s")
            
            # Check for collision
            if handler.has_collided:
                has_collided = True
                print("Collision detected! Stopping simulation.")
                break
            
            # Process frames for video (async-safe version)
            if 'bev' in handler.frame_queues and handler.frame_queues['bev'] and (config['save_options']['save_video'] or config['save_options']['save_images']):
                try:
                    # Process only the most recent frame to avoid blocking in async mode
                    if len(handler.frame_queues['bev']) > 0:
                        frame_bgr = handler.frame_queues['bev'][-1].copy()  # Get latest frame and copy it
                        handler.frame_queues['bev'].clear()  # Clear queue to prevent buildup in async mode
                        
                        # Add text overlays
                        collision_text = f"Predicted Collision Probability: {max_collision_prob:.4f}"
                        cv2.putText(frame_bgr, collision_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        ground_truth_text = f"Groundtruth Collision Probability: {ground_truth_collision_prob:.4f}"
                        cv2.putText(frame_bgr, ground_truth_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        latency_text = f"Current Latency: {current_delta_k * 10} ms"
                        cv2.putText(frame_bgr, latency_text, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        if handler.video_writer and config['save_options']['save_video']:
                            handler.video_writer.write(frame_bgr)
                        
                        if config['save_options']['save_images']:
                            # Save BEV camera with overlays
                            os.makedirs(os.path.join(output_dir, 'bev_images'), exist_ok=True)
                            cv2.imwrite(
                                os.path.join(output_dir, f'bev_images/frame_{tick}.png'),
                                frame_bgr)
                            
                            # Save BEV view without overlays
                            handler.save_all_camera_frames(tick, output_dir)
                except Exception as e:
                    # Silently handle frame processing errors in async mode
                    pass
            
            # Log collision probabilities
            timestamp = tick * config['simulation']['delta_seconds']
            with open(handler.collision_prob_file, 'a') as f:
                f.write(f'{timestamp:.2f},{tick},{current_delta_k},'
                       f'{max_collision_prob:.4f},{ground_truth_collision_prob:.4f}\n')
            
            tick += 1
    
    except Exception as e:
        print(f"Error in FogSim simulation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'env' in locals():
            env.close()
    
    return has_collided, current_delta_k


def run_obstacle_only_simulation(config, trajectory_file=None):
    """Run a simulation with only the obstacle vehicle to record its trajectory"""
    # Use trajectory file from config if none provided
    if trajectory_file is None:
        trajectory_file = config['trajectories']['obstacle']

    # Connect to CARLA
    client = carla.Client(config['simulation']['host'],
                          config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Set synchronous mode (ALWAYS use sync mode for trajectory collection)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True  # Force sync mode for consistent waypoint collection
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    
    print(f"Using synchronous mode for obstacle trajectory collection (delta_seconds={settings.fixed_delta_seconds})")

    blueprint_library = world.get_blueprint_library()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Setup obstacle vehicle spawn point based on ego vehicle's position
    # (since obstacle spawn is defined relative to ego in config)
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    obstacle_spawn_point = spawn_points[1]
    obstacle_spawn_point.location.x = ego_spawn_point.location.x + config[
        'obstacle_vehicle']['spawn_offset']['x']
    obstacle_spawn_point.location.y = ego_spawn_point.location.y + config[
        'obstacle_vehicle']['spawn_offset']['y']
    obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + config[
        'obstacle_vehicle']['spawn_offset']['yaw']

    # Spawn only the obstacle vehicle
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)

    if obstacle_vehicle is None:
        raise RuntimeError("Failed to spawn obstacle vehicle")

    try:
        # Use ORIGINAL tick-based control to record accurate waypoints for collision scenarios
        total_ticks = (config['ego_vehicle']['go_straight_ticks'] +
                       config['ego_vehicle']['turn_ticks'] +
                       config['ego_vehicle'].get('after_turn_ticks', 0))

        for tick in range(total_ticks):
            world.tick()

            # Apply ORIGINAL tick-based control to obstacle vehicle
            obstacle_control = carla.VehicleControl()

            if tick < config['obstacle_vehicle']['go_straight_ticks']:
                # Initial straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['straight']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['straight']
            elif tick < config['obstacle_vehicle']['go_straight_ticks'] + config['obstacle_vehicle']['turn_ticks']:
                # Turning phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['after_turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['after_turn']

            obstacle_vehicle.apply_control(obstacle_control)
            save_trajectory(obstacle_vehicle, trajectory_file)

    finally:
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()

        # Restore original settings
        world.apply_settings(original_settings)
        print(f"Obstacle trajectory saved to {trajectory_file}")


def calculate_collision_probabilities(obstacle_tracker, predicted_positions,
                                      ego_trajectory, tick):
    """
    Calculate collision probabilities for predicted positions against ego trajectory.
    
    Args:
        obstacle_tracker: The tracker object used for collision probability calculation
        predicted_positions: List of predicted future positions of the obstacle
        ego_trajectory: List of ego vehicle trajectory points
        tick: Current simulation tick
    
    Returns:
        tuple: (max_collision_prob, collision_time, collision_probabilities)
            - max_collision_prob: Maximum collision probability across all predictions
            - collision_time: Time step at which maximum collision probability occurs
            - collision_probabilities: List of all calculated collision probabilities
    """
    collision_probabilities = []
    for step, predicted_pos in enumerate(predicted_positions):
        if tick + step < len(ego_trajectory):
            ego_trajectory_point = ego_trajectory[tick + step]
            predicted_pos = [
                predicted_pos[0], predicted_pos[1], predicted_pos[2]
            ]
            collision_prob = obstacle_tracker.calculate_collision_probability_with_trajectory(
                ego_trajectory_point, predicted_pos)
            collision_probabilities.append(collision_prob)

    max_collision_prob = max(
        collision_probabilities) if collision_probabilities else 0.0
    collision_time = collision_probabilities.index(
        max_collision_prob) if collision_probabilities else 0

    # print(
    #     f"Tick {tick}: Max collision probability: {max_collision_prob:.4f} at time step {collision_time}"
    # )

    return max_collision_prob, collision_time, collision_probabilities


def get_monte_carlo_spawn_point(config, ego_spawn_point, std_dev=0):
    """
    Generate a randomized spawn point for the obstacle vehicle using Monte Carlo sampling.
    
    Args:
        config: Configuration dictionary
        ego_spawn_point: Base ego vehicle spawn point
        std_dev: Standard deviation for the normal distribution (in meters)
    
    Returns:
        carla.Transform: Randomized spawn point
    """
    base_x = ego_spawn_point.location.x + config['obstacle_vehicle'][
        'spawn_offset']['x']
    base_y = ego_spawn_point.location.y + config['obstacle_vehicle'][
        'spawn_offset']['y']
    base_yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle'][
        'spawn_offset']['yaw']

    # Sample from normal distribution for position and yaw
    x = np.random.normal(base_x, std_dev)
    y = np.random.normal(base_y, std_dev)
    yaw = np.random.normal(base_yaw,
                           std_dev * 2)  # Larger variation in orientation

    spawn_point = carla.Transform(carla.Location(x=x, y=y, z=0.0),
                                  carla.Rotation(yaw=yaw))

    return spawn_point


def run_monte_carlo_simulation(config, num_samples=10, output_dir='./results', use_fogsim=False, no_risk_eval=False):
    """
    Run multiple simulations with Monte Carlo sampling of obstacle spawn points.
    
    Args:
        config: Configuration dictionary
        num_samples: Number of Monte Carlo samples to run
        use_fogsim: Whether to use FogSim version with network simulation
    
    Returns:
        dict: Statistics about collisions and spawn points
    """
    collision_stats = {
        'num_collisions': 0,
        'spawn_points': [],
        'collision_cases': []
    }

    # Connect to CARLA
    client = carla.Client(config['simulation']['host'],
                          config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Get base spawn point for ego vehicle
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    for sample in range(num_samples):
        print(f"\nRunning Monte Carlo sample {sample + 1}/{num_samples}")

        # Generate random spawn point
        obstacle_spawn_point = get_monte_carlo_spawn_point(
            config, ego_spawn_point)

        # Update config with new spawn point
        sample_config = dict(config)
        sample_config['obstacle_vehicle']['spawn_offset'] = {
            'x': obstacle_spawn_point.location.x - ego_spawn_point.location.x,
            'y': obstacle_spawn_point.location.y - ego_spawn_point.location.y,
            'yaw': obstacle_spawn_point.rotation.yaw -
                   ego_spawn_point.rotation.yaw
        }

        # Update trajectory files for this sample
        sample_config['trajectories'] = {
            'ego': f'./ego_trajectory_sample_{sample}.csv',
            'obstacle': f'./obstacle_trajectory_sample_{sample}.csv'
        }
        # try:
        #     # Generate obstacle trajectory
        #     run_obstacle_only_simulation(sample_config)
        # except Exception as e:
        #     # obstacle spawn point is invalid
        #     print(f"Error in sample {sample}: {e}")
        #     continue

        try:
            # Generate trajectories for both ego and obstacle vehicles
            run_first_simulation(sample_config,
                               ego_trajectory_file=sample_config['trajectories']['ego'],
                               obstacle_trajectory_file=sample_config['trajectories']['obstacle'])

            # Run simulation and check for collision
            has_collided, current_delta_k = run_adaptive_simulation_fogsim(
                    sample_config, output_dir, no_risk_eval)
            # else:
            #     has_collided, current_delta_k = run_adaptive_simulation(
            #         sample_config, output_dir, no_risk_eval)

            # Check if collision occurred
            if has_collided:
                collision_stats['num_collisions'] += 1
                collision_stats['collision_cases'].append({
                    'spawn_point': obstacle_spawn_point,
                    'sample_num': sample
                })
            else:
                collision_stats['spawn_points'].append(obstacle_spawn_point)

        except Exception as e:
            print(f"Error in sample {sample}: {e}")
            collision_stats['num_collisions'] += 1
            collision_stats['collision_cases'].append({
                'spawn_point': obstacle_spawn_point,
                'sample_num': sample,
                'error': str(e)
            })

        if not os.path.exists(
                os.path.join(output_dir, 'monte_carlo_results/statistics.csv')):
            with open(
                    os.path.join(output_dir,
                                 'monte_carlo_results/statistics.csv'),
                    'w') as f:
                f.write(f"scenario,lmax,delta_k,collision,delta_k_used\n")
        # save and append the stats to a csv file
        # with open(os.path.join(output_dir, 'monte_carlo_results/statistics.csv'), 'a') as f:
        # doesn't support 'a', read and write instead
        with open(
                os.path.join(output_dir, 'monte_carlo_results/statistics.csv'),
                'r') as f:
            lines = f.readlines()
        with open(
                os.path.join(output_dir, 'monte_carlo_results/statistics.csv'),
                'w') as f:
            for line in lines:
                f.write(line)
            # scenario, lmax, delta_k, collision yes or no, delta_k_used
            scenario_type = config['video']['filename'].split('/')[-1].split(
                '_collision')[0]
            # f.write(
            #     f"{scenario_type},{config['simulation']['l_max']},{current_delta_k},{has_collided},{config['simulation']['delta_k']}\n"
            # )

    return collision_stats


def restart_carla_docker():
    """Restart the CARLA Docker container"""
    import subprocess
    import time

    try:
        # Stop existing CARLA container
        subprocess.run(['docker', 'stop', 'carla'], check=False)
        subprocess.run(['docker', 'rm', 'carla'], check=False)

        # Start new CARLA container with audio disabled
        subprocess.run(
            [
                'docker',
                'run',
                '-d',
                '--name=carla',
                '--privileged',
                '--gpus',
                'all',
                '--net=host',
                '-e',
                'PULSE_SERVER=/dev/null',  # Disable PulseAudio
                '-e',
                'ALSA_CONFIG_PATH=/dev/null',  # Disable ALSA
                '-v',
                '/tmp/.X11-unix:/tmp/.X11-unix:rw',
                'carlasim/carla:0.9.15',
                '/bin/bash',
                './CarlaUE4.sh',
                '-RenderOffScreen',
                '-nosound'  # Add -nosound flag
            ],
            check=True)

        # Wait for CARLA to initialize
        time.sleep(10)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restarting CARLA container: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run CARLA simulation with configurable parameters')
    parser.add_argument('--cautious_delta_k',
                        type=int,
                        default=-1,
                        help='Value for cautious_delta_k parameter')
    parser.add_argument('--config_type',
                        type=str,
                        choices=['right_turn', 'left_turn', 'merge'],
                        default='merge',
                        help='Type of configuration to use')
    parser.add_argument('--emergency_brake_threshold',
                        type=float,
                        default=1.1,
                        help='Threshold for emergency braking')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results',
                        help='Directory to store results')
    parser.add_argument('--use_fogsim',
                        action='store_true',
                        help='Use FogSim for network delay simulation')
    parser.add_argument('--no_risk_eval',
                        action='store_true',
                        help='Disable EKF tracking and risk evaluation')
    parser.add_argument('--sync_mode',
                        action='store_true',
                        help='Use synchronous mode for CARLA simulation (default: asynchronous)')
    parser.add_argument('--carla_port',
                        type=int,
                        default=2000,
                        help='CARLA server port (default: 2000)')
    args = parser.parse_args()

    # Select configuration based on argument
    config_map = {
        'right_turn': unprotected_right_turn_config,
        'left_turn': unprotected_left_turn_config,
        'merge': opposite_direction_merge_config
    }

    base_config = config_map[args.config_type]

    # Update configuration with command line parameters
    if args.cautious_delta_k != -1:
        base_config['simulation']['cautious_delta_k'] = args.cautious_delta_k
        base_config['simulation']['l_max'] = args.cautious_delta_k
        base_config['simulation']['delta_k'] = args.cautious_delta_k

    # Update emergency brake threshold
    base_config['simulation'][
        'emergency_brake_threshold'] = args.emergency_brake_threshold
    
    # Add sync_mode to configuration
    base_config['simulation']['sync_mode'] = args.sync_mode
    
    # Add scenario type to configuration for filename generation
    base_config['scenario_type'] = args.config_type
    
    # Update CARLA port
    base_config['simulation']['port'] = args.carla_port

    # Update output paths based on output directory
    base_config['video']['filename'] = os.path.join(args.output_dir,
                                                    'simulation.mp4')
    base_config['trajectories']['ego'] = os.path.join(args.output_dir,
                                                      'ego_trajectory.csv')
    base_config['trajectories']['obstacle'] = os.path.join(
        args.output_dir, 'obstacle_trajectory.csv')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'bev_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'monte_carlo_results'),
                exist_ok=True)

    max_retries = 3
    retry_count = 0

    # Clean old files in the output directory
    if os.path.exists(os.path.join(args.output_dir, 'bev_images')):
        shutil.rmtree(os.path.join(args.output_dir, 'bev_images'))
    os.makedirs(os.path.join(args.output_dir, 'bev_images'))

    collision_prob_file = os.path.join(args.output_dir,
                                       'collision_probabilities.csv')
    if os.path.exists(collision_prob_file):
        os.remove(collision_prob_file)

    # Base configuration for Monte Carlo simulation
    num_samples = 1

    while retry_count < max_retries:
        try:
            # Run Monte Carlo simulation
            stats = run_monte_carlo_simulation(base_config, num_samples,
                                               args.output_dir, use_fogsim=args.use_fogsim, 
                                               no_risk_eval=args.no_risk_eval)

            # Calculate and save statistics
            collision_rate = stats['num_collisions'] / num_samples

            # Save results
            stats_file = os.path.join(args.output_dir, 'monte_carlo_results',
                                      'statistics.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Monte Carlo Simulation Results\n")
                f.write(f"Configuration:\n")
                f.write(f"  Config Type: {args.config_type}\n")
                f.write(f"  Using FogSim: {args.use_fogsim}\n")
                f.write(
                    f"  Cautious Delta K: {base_config['simulation']['cautious_delta_k']}\n"
                )
                f.write(
                    f"  Emergency Brake Threshold: {base_config['simulation']['emergency_brake_threshold']}\n\n"
                )
                f.write(f"Results:\n")
                f.write(f"  Number of samples: {num_samples}\n")
                f.write(f"  Number of collisions: {stats['num_collisions']}\n")
                f.write(f"  Collision rate: {collision_rate:.2%}\n\n")

                f.write("Collision cases:\n")
                for case in stats['collision_cases']:
                    f.write(f"Sample {case['sample_num']}:\n")
                    f.write(
                        f"  Spawn point: x={case['spawn_point'].location.x:.2f}, "
                        f"y={case['spawn_point'].location.y:.2f}, "
                        f"yaw={case['spawn_point'].rotation.yaw:.2f}\n")
                    if 'error' in case:
                        f.write(f"  Error: {case['error']}\n")

            print(f"\nMonte Carlo simulation completed.")
            print(
                f"Collision rate: {collision_rate:.2%} ({stats['num_collisions']}/{num_samples} collisions)"
            )
            print(f"Results saved to {args.output_dir}")
            break

        except Exception as e:
            retry_count += 1
            print(f"\nError occurred: {e}")
            print(f"Retry {retry_count}/{max_retries}")

            print(
                "CARLA connection issue detected. Attempting to restart CARLA..."
            )
            if restart_carla_docker():
                print("CARLA successfully restarted")
                time.sleep(5)  # Give additional time for CARLA to stabilize
            else:
                print("Failed to restart CARLA")

            if retry_count >= max_retries:
                print("Max retries reached. Exiting...")
                break

            time.sleep(5)  # Wait before retrying


if __name__ == '__main__':
    main()
