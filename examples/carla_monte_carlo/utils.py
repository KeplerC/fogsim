from abc import ABC
from abc import abstractmethod
import argparse
import math
import os
import shutil
import time

import carla
import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.stats import norm

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


class GroundTruthTracker(BaseTracker):

    def __init__(self,
                 ego_vehicle,
                 obstacle_vehicle,
                 dt=0.05,
                 trajectory_file='./obstacle_trajectory.csv'):
        super().__init__(ego_vehicle, obstacle_vehicle, dt)
        self.trajectory_file = trajectory_file
        self.trajectory = self.load_trajectory()
        self.current_index = 0
        self.obs_cov = np.identity(
            2) * 0.001  # Very small uncertainty for ground truth

    def load_trajectory(self):
        """Load pre-recorded trajectory from file"""
        trajectory = []
        with open(self.trajectory_file, 'r') as f:
            for line in f:
                x, y, yaw = map(float, line.strip().split(','))
                trajectory.append([x, y, math.radians(yaw)])
        return trajectory

    def update(self, state, tick_number):
        """Update tracker with new state measurement"""
        x, y, yaw_deg = state
        theta = math.radians(yaw_deg)
        self.history.append((x, y, theta))
        self.history_ticks.append(tick_number)
        self.current_index = tick_number

    def predict_future_position(self, steps_ahead):
        """Predict future position using ground truth trajectory"""
        predicted_states = []
        for i in range(steps_ahead):
            future_index = self.current_index + i
            if future_index < len(self.trajectory):
                predicted_states.append(self.trajectory[future_index])
            else:
                # If we run out of trajectory, use the last known position
                predicted_states.append(self.trajectory[-1])
        return predicted_states


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

    print(
        f"Tick {tick}: Max collision probability: {max_collision_prob:.4f} at time step {collision_time}"
    )

    return max_collision_prob, collision_time, collision_probabilities
