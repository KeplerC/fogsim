# %%
import numpy as np
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
from sympy import Point, Polygon

# %%
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def Sort_List(l):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    l.sort(key = lambda x: x[2])
    return l

def collision_case(corner,w,l):
    x,y = corner
    d = 100
    close_point_list = [(x,l/2),(x,-l/2),(w/2,y),(w/2,-y)]
    for pts in close_point_list:
        dis = distance.euclidean(corner,pts)
        if dis<d:
            d = dis
            close_point = pts
    return (close_point,-d)

def collision_dis_dir(corner,w,l):
#     print('corner',corner)
    closet_point = (0,0)
    d = 100
    ego_corner = [(w/2,l/2),(w/2,-l/2),(-w/2,l/2),(-w/2,-l/2)]
    cor_x,cor_y = corner
    
    if np.abs(cor_x) < w/2 and np.abs(cor_y) < l/2 :
        closet_point,d = collision_case(corner,w,l)

    elif np.abs(cor_y) < l/2 and np.abs(cor_x) >= w/2 :
        d = -w/2-cor_x if cor_x<-w/2 else cor_x-w/2
        closet_point =  (-w/2,cor_y)  if cor_x<-w/2 else (w/2,cor_y)
    elif np.abs(cor_x) < w/2 and np.abs(cor_y) >= l/2 :
        d = -l/2-cor_y if cor_y<-l/2 else cor_y-l/2
        closet_point =  (cor_x,-l/2)  if cor_y<-l/2 else (cor_x,l/2)
    else:
        for ego_cor in ego_corner:
            cor_dis = distance.euclidean(ego_cor, corner)
            if cor_dis<d:
                d = cor_dis
                closet_point = ego_cor
#     print(closet_point,d)
    return [closet_point,d]

def object_tranformation(s1,s2):
    """
    kc's(orignal) frame {0}
    object s1's center as coordinate center {1}
    Transformation from {0} to {1} for object s2
    """ 
    relative_rad = np.pi/2 - s1[2]
    R = np.array([ [np.cos(relative_rad), -np.sin(relative_rad)],
                    [np.sin(relative_rad), np.cos(relative_rad)]])
#     print(R)
    # Obstacle coordinate transformation from {0} to {1}
    obs_center_homo = np.array([s2[0]-s1[0],s2[1]-s1[1]])
    obs_x,obs_y = np.matmul(R ,obs_center_homo)
    obs_theta = s2[2] + relative_rad
    so_f1 = [obs_x,obs_y,obs_theta]
    return so_f1


def point_transformation(s1,p):
    """
    transfer point p back to the orginal coordinate from coordinate frame s1
    """

    relative_rad = np.pi/2 - s1[2]
    R = np.array([ [np.cos(-relative_rad), -np.sin(-relative_rad)],
                    [np.sin(-relative_rad), np.cos(-relative_rad)]])
    p = np.matmul(R,p)
    p= np.array([p[0]+s1[0],p[1]+s1[1]])
    return p

def corners_cal(so_f1,lo,wo,corner_dir):
    obs_center_matrix = np.array([so_f1[0],so_f1[1]])
    shape_matrix = np.array([[wo/2,0],[0,lo/2]]) 

    relative_rad = so_f1[2]-np.pi/2
#     print(relative_rad)
    ori_matrix = np.array([ [np.cos(relative_rad), -np.sin(relative_rad)],
                        [np.sin(relative_rad), np.cos(relative_rad)] ])

    obs_corners = np.matmul(np.matmul(ori_matrix,shape_matrix),corner_dir.T) +np.tile(obs_center_matrix,(4,1)).T #2*4
    return obs_corners

def collision_point_rect(se, so, we = 1.5, le = 4, wo = 1.4, lo = 4,plot_flag = 0):
    """
    Input:
    - ego vehicle's state se = (x_e,y_e,theta_e)  at time k and shape (w_e,l_e)
    - obstacle's state mean so = (x_o,y_o,theta_o) at time k and shape prior (w_o,l_o)

    Output:
    - collision point and collision direction
    """
    # check theta is in radians and with in -pi to pi   
    if not isinstance(se[2], (int, float)) or not isinstance(so[2], (int, float)):
        raise ValueError("Theta values must be numeric.")
    # if se[2] < -np.pi*2 or se[2] > np.pi*2 or so[2] < -np.pi*2 or so[2] > np.pi*2:
    #     #raise ValueError(f"Theta values {se[2]} and {so[2]} must be within -pi and pi.")
    #     print(f"Theta values {se[2]} and {so[2]} must be within -pi and pi.")
    
    corner_dir = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])

    # Transfer matrix from kc's frame to ego center
    so_f1 = object_tranformation(se,so)

    # 4 obstacle corner point to ego vehicle distance
    obs_corners = corners_cal(so_f1,lo,wo,corner_dir)
    obs_corners = obs_corners.T
    closest_point_dis = [ [tuple(corner)]+collision_dis_dir(corner,we,le) for corner in obs_corners]
    closest_point_dis = (Sort_List(closest_point_dis))
    # print('dis:',closest_point_dis[0][2])
    
    
    if plot_flag == 1:
        hull = ConvexHull(obs_corners)
        obs_corner_cov = obs_corners[hull.vertices]
        obs_corner_cov = np.append(obs_corner_cov,[obs_corner_cov[0]], axis=0)
        plt.plot(obs_corner_cov[:,0], obs_corner_cov[:,1], 'b--', lw=2)

        ego_corners = np.array([(we/2,le/2),(we/2,-le/2),(-we/2,le/2),(-we/2,-le/2)])
        hull = ConvexHull(ego_corners)
        ego_corners_cov = ego_corners[hull.vertices]
        ego_corners_cov = np.append(ego_corners_cov,[ego_corners_cov[0]], axis=0)
        plt.plot(ego_corners_cov[:,0], ego_corners_cov[:,1], 'r--', lw=2)
    
    
    # Transfer matrix from kc's frame to obstacle center frame
    se_f1 = object_tranformation(so,se)

    # 4 obstacle corner point to ego vehicle distance
    ego_corners = corners_cal(se_f1,le,we,corner_dir)
    ego_corners = ego_corners.T
    closest_point_dis2 = [ [tuple(corner)]+collision_dis_dir(corner,wo,lo) for corner in ego_corners]
    closest_point_dis2 = (Sort_List(closest_point_dis2))
#     print('dis:',closest_point_dis2[0][2])
    
#     if closest_point_dis2[0][2] <0 or closest_point_dis[0][2] <0:
#         print(se,so)
    
    # print(closest_point_dis2)
    
    if closest_point_dis[0][2]<=closest_point_dis2[0][2]:
        #transfer back to original coordinates closest_point_dis[0][1]
        obstacle_point = point_transformation(se,closest_point_dis[0][0])
        ego_point = point_transformation(se,closest_point_dis[0][1])
        return (obstacle_point, ego_point, closest_point_dis[0][2])
    else:
        #transfer back to original coordinates closest_point_dis[0][1]
        obstacle_point = point_transformation(so,closest_point_dis2[0][1])
        ego_point = point_transformation(so,closest_point_dis2[0][0])
        

    
    
        #define Matplotlib figure and axis


    
    

    
    
    #display plot
   
    return (obstacle_point, ego_point, closest_point_dis2[0][2])



 

# %%
from scipy.stats import norm
def collision_probablity(V, P, dis, obs_cov):
    PV = V - P
    VP = P - V
    theta_pv =  np.arccos(np.dot(PV,np.array([1,0]))/np.linalg.norm(PV))
    R = np.array([ [np.cos(theta_pv), -np.sin(theta_pv)],
                    [np.sin(theta_pv), np.cos(theta_pv)]])
#     print(np.matmul(R, R.T))
    den = np.matmul(np.matmul(R, obs_cov),R.T)
#     print(np.cos(theta_pv)* VP[0]+ np.sin(theta_pv) * VP[1])
    point_dis = np.cos(theta_pv)* VP[0]+ np.sin(theta_pv) * VP[1]
    col_prob =  norm.cdf( -dis/ np.sqrt(den[0,0]))

    return col_prob

# fig, ax = plt.subplots(figsize=(6, 6))

# obs_cov = np.identity(2)*0.04
# # 0.0174532925
# ori_var = 2/180*np.pi
# obs_state_cov = np.identity(3)*0.04
# obs_state_cov[2,2] = ori_var

# se = [1.75,4,np.pi/2]
# so_list = [[3.2,7.2,0],[2.4,7.1,np.pi/16], [1.6,7,np.pi/8],[0.8,6.8,3*np.pi/16], [0,6,np.pi/4]   ]
# mc_cp_ori = []
# for i in range(1):
#     print('---------------------------------------------')
#     print('iteration:' , i )
#     so_list = [[3.2,7.2,0],[2.4,7.1,np.pi/16], [1.6,7,np.pi/8],[0.8,6.8,3*np.pi/16], [0,6,np.pi/4]   ]
#     so_list = [[0,6,np.pi/4]   ]
#     for obs in so_list[:1]:
#         print('case', obs)
#         obstacle_col_point, ego_col_point, dis = collision_point_rect(se, obs, we = 1.8, le = 4, wo = 1.4, lo = 4, plot_flag = 1) 
#         print(ego_col_point,dis)
#         if dis == -1:
#             print('collision')
#         else:
#             col_prob = collision_probablity(obstacle_col_point, ego_col_point,dis,  obs_cov)
#             print('collision probability:',col_prob)
    
#         point_samples = np.random.multivariate_normal(obs[:2], obs_cov, 10000)
#         collision_count = 0
#         for s_sample in point_samples:
#             obs_state = obs
#             obs_state[:2] = s_sample
#             obstacle_col_point, ego_col_point, dis = collision_point_rect(se, obs_state, we = 1.5, le = 4, wo = 1.4, lo = 4) 
#             if dis == -1:
#                 collision_count += 1
#         mc_cp = collision_count/10000
#         print('MC collision probability no orientation uncertainty:',collision_count/10000)
#         cp.append([i, col_prob, mc_cp])
#     #     point_samples = np.random.multivariate_normal(so[:2], obs_cov, 100000)
#     #     ori_samples = np.random.normal(so[2], ori_var, 100000)
#     #     ori_samples = ori_samples[:,None]
#     #     state_samples = np.hstack((point_samples, ori_samples))
#         sample_num = 1000000
#         state_samples = np.random.multivariate_normal(obs, obs_state_cov, sample_num)
#         collision_count = 0
#         for obs_state in state_samples:
#             obstacle_col_point, ego_col_point, dis = collision_point_rect(se, obs_state, we = 1.5, le = 4, wo = 1.4, lo = 4) 
#             if dis == -1:
#                 collision_count += 1
#         print('MC collision probability with orientation uncertainty:',collision_count/sample_num)
#         mc_ori = collision_count/sample_num
#         mc_cp_ori.append(mc_ori)
# plt.show()  

# # %%
# norm.cdf(-0.2626/0.163)

# # %%
# res = cp_np.reshape(10,5,3)
# mc_res = []
# cp_res = []
# for i in range(5):
#     mc_res.append(np.average(res[:,i,2]))
#     cp_res.append(np.average(res[:,i,1]))

# # %%

# np.zeros(3)

# # %%
# mc_ori = [0.020661,0.18252,0.502203, 0.29935, 0.503334]

# # %%
# fig, ax = plt.subplots(figsize=(6, 6))
# plt.plot(mc_ori, 'm-', lw=2,label="mc-with orientation uncertainty")
# plt.plot(mc_res, 'y-', lw=2, label="mc-no orientation uncertainty")
# plt.plot(cp_res, 'g-', lw=2,label="our method")
# plt.legend(loc="lower right")
# plt.xlabel("TimeStamp")
# plt.ylabel("Collision probablity")

# # %%
# PV = np.array([0.3,0.4])
# VP = np.array([-0.3,-0.4])
# theta_pv =  np.arccos(np.dot(PV,np.array([1,0]))/np.linalg.norm(PV))
# R = np.array([ [np.cos(theta_pv), -np.sin(theta_pv)],
#                 [np.sin(theta_pv), np.cos(theta_pv)]])
# print(np.cos(theta_pv)* VP[0]+ np.sin(theta_pv) * VP[1])
# den = np.matmul(np.matmul(R, obs_cov),R.T)
# print(den[0,0])
# col_prob =  norm.cdf(np.cos(theta_pv)* VP[0]+ np.sin(theta_pv) * VP[1] / den[0,0])
# print(col_prob)


