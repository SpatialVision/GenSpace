import random
from itertools import combinations

import numpy as np
from osdsynth.processor.pointcloud import calculate_distances_between_point_clouds, human_like_distance
from osdsynth.processor.prompt_template import *
from osdsynth.processor.prompt_utils import *
from osdsynth.processor.prompt_spatitalbench_template import *



def get_upper(theta):
    for i in [-1,0,1,2]:
        if theta < i * np.pi / 2 - np.pi / 4:
            return i * np.pi / 2 - np.pi / 4
    return 3 * np.pi / 2 - np.pi / 4



def CR_two_front(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    
    B_rotation_matrix = A_rotation_matrix.T @ B_rotation_matrix # 在A的坐标系下，B的旋转矩阵
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    A_P_B_last = A['A_P_B']

    
    theta1 = np.arctan2(A_P_B[2], A_P_B[0])
    theta2 = np.arctan2(A_P_B_last[2], A_P_B_last[0])
    theta1_upper = get_upper(theta1)
    
    if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
        theta1_lower = theta1_upper - np.pi / 2
        position = True if theta1_upper > theta2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position = True if theta2 < -3/4*np.pi or theta2 > 3/4*np.pi else False
        
    
    max_angle = 30

    
    angle_rad_A = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    direction = angle_rad_A < max_angle / 180 * np.pi
    
    check = position and direction
    
    question_template = f"Are [A] and [B] maintaining their original relative relationship when viewed from the front of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position:
            w1 = 1
        else:
            if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
                w1 = 1 - np.min([np.abs(theta1_upper-theta2), np.abs(theta2-theta1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta2 = theta2 + 2 * np.pi if theta2 < -3/4*np.pi else theta2 # 转化到0~2pi
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta2), np.abs(theta2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w2 = 1
        else:
            w2 = 1 - np.abs(angle_rad_A - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 else w1 * w2

    return question, answer, check, score

def CR_two_back(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    
    B_rotation_matrix = A_rotation_matrix.T @ B_rotation_matrix # 在A的坐标系下，B的旋转矩阵
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    A_P_B_last = A['A_P_B']

    
    theta1 = np.arctan2(A_P_B[2], A_P_B[0]) 
    theta2 = np.arctan2(A_P_B_last[2], A_P_B_last[0]) 
    theta1_upper = get_upper(theta1)
    if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
        theta1_lower = theta1_upper - np.pi / 2
        position = True if theta1_upper > theta2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position = True if theta2 < -3/4*np.pi or theta2 > 3/4*np.pi else False
    
    
    max_angle = 30
    
    angle_rad_A = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,1])), -1.0, 1.0))
    direction = angle_rad_A < max_angle / 180 * np.pi
    
    check = position and direction
    
    question_template = f"Are [A] and [B] maintaining their original relative relationship when viewed from the back of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position:
            w1 = 1
        else:
            if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
                w1 = 1 - np.min([np.abs(theta1_upper-theta2), np.abs(theta2-theta1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta2 = theta2 + 2 * np.pi if theta2 < -3/4*np.pi else theta2 # 转化到0~2pi
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta2), np.abs(theta2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w2 = 1
        else:
            w2 = 1 - np.abs(angle_rad_A - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 else w1 * w2

    return question, answer, check, score

def CR_two_left(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    
    B_rotation_matrix = A_rotation_matrix.T @ B_rotation_matrix # 在A的坐标系下，B的旋转矩阵
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    A_P_B_last = A['A_P_B']
    
    theta1 = np.arctan2(A_P_B[2], A_P_B[0]) 
    theta2 = np.arctan2(A_P_B_last[2], A_P_B_last[0]) 
    theta1_upper = get_upper(theta1)
    if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
        theta1_lower = theta1_upper - np.pi / 2
        position = True if theta1_upper > theta2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position = True if theta2 < -3/4*np.pi or theta2 > 3/4*np.pi else False
    
    
    max_angle = 30

    
    angle_rad_A = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([-1,0,0])), -1.0, 1.0))
    direction = angle_rad_A < max_angle / 180 * np.pi
    
    check = position and direction
    
    question_template = f"Are [A] and [B] maintaining their original relative relationship when viewed from the left of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position:
            w1 = 1
        else:
            if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
                w1 = 1 - np.min([np.abs(theta1_upper-theta2), np.abs(theta2-theta1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta2 = theta2 + 2 * np.pi if theta2 < -3/4*np.pi else theta2 # 转化到0~2pi
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta2), np.abs(theta2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w2 = 1
        else:
            w2 = 1 - np.abs(angle_rad_A - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 else w1 * w2

    return question, answer, check, score

def CR_two_right(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    
    B_rotation_matrix = A_rotation_matrix.T @ B_rotation_matrix # 在A的坐标系下，B的旋转矩阵
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    A_P_B_last = A['A_P_B']

    
    theta1 = np.arctan2(A_P_B[2], A_P_B[0]) 
    theta2 = np.arctan2(A_P_B_last[2], A_P_B_last[0]) 
    theta1_upper = get_upper(theta1)
    if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
        theta1_lower = theta1_upper - np.pi / 2
        position = True if theta1_upper > theta2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position = True if theta2 < -3/4*np.pi or theta2 > 3/4*np.pi else False
    
    
    max_angle = 30
    # angle_rad_AB = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    # same_direction = angle_rad_AB < max_angle / 180 * np.pi
    
    angle_rad_A = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    direction = angle_rad_A < max_angle / 180 * np.pi
    
    check = position and direction
    
    question_template = f"Are [A] and [B] maintaining their original relative relationship when viewed from the right of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position:
            w1 = 1
        else:
            if theta1_upper < np.pi and theta1_upper > -np.pi/2: 
                w1 = 1 - np.min([np.abs(theta1_upper-theta2), np.abs(theta2-theta1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta2 = theta2 + 2 * np.pi if theta2 < -3/4*np.pi else theta2 # 转化到0~2pi
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta2), np.abs(theta2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w2 = 1
        else:
            w2 = 1 - np.abs(angle_rad_A - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 else w1 * w2

    return question, answer, check, score


def CR_three_front(A, B, C):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    C_desc, C_cloud = C["caption"], C["pcd"]
    C_desc = C_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    C_pos = C_cloud.get_center()
    C_pos[0] = -C_pos[0]; C_pos[1] = -C_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    C_rotation_matrix = C["rotation_matrix"]
    
    A_rotation_matrix = C_rotation_matrix.T @ A_rotation_matrix # 在C的坐标系下，A的旋转矩阵
    B_rotation_matrix = C_rotation_matrix.T @ B_rotation_matrix # 在C的坐标系下，B的旋转矩阵
    
    C_P_A = C_rotation_matrix.T @ (A_pos - C_pos) # 在C的坐标系下，A相对于C的位置
    C_P_B = C_rotation_matrix.T @ (B_pos - C_pos) # 在C的坐标系下，B相对于C的位置
    C_P_A_last = C['C_P_A']
    C_P_B_last = C['C_P_B']
    
    theta_CA1 = np.arctan2(C_P_A[2], C_P_A[0])
    theta_CB1 = np.arctan2(C_P_B[2], C_P_B[0])
    theta_CA2 = np.arctan2(C_P_A_last[2], C_P_A_last[0])
    theta_CB2 = np.arctan2(C_P_B_last[2], C_P_B_last[0])
    
    
    theta_CA1_upper = get_upper(theta_CA1)
    theta_CA1_lower = theta_CA1_upper - np.pi / 2
    if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2: 
        theta1_lower = theta_CA1_upper - np.pi / 2
        position_CA = True if theta_CA1_upper > theta_CA2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CA = True if theta_CA2 < -3/4*np.pi or theta_CA2 > 3/4*np.pi else False
        
    
    theta_CB1_upper = get_upper(theta_CB1)
    theta_CB1_lower = theta_CB1_upper - np.pi / 2
    if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2: 
        theta1_lower = theta_CB1_upper - np.pi / 2
        position_CB = True if theta_CB1_upper > theta_CB2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CB = True if theta_CB2 < -3/4*np.pi or theta_CB2 > 3/4*np.pi else False
    
    
    position = position_CA and position_CB
    
    
    max_angle = 30

    
    angle_rad_C = np.arccos(np.clip(np.dot(C_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    direction = angle_rad_C < max_angle / 180 * np.pi


    check = position and direction
    question_template = f"Are [A], [B] and [C] maintaining their original relative relationship when viewed from the front of [C]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc).replace("[C]", C_desc)
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position_CA:
            w1 = 1
        else:
            if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2:                
                w1 = 1 - np.min([np.abs(theta_CA1_upper-theta_CA2), np.abs(theta_CA2-theta_CA1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta_CA2 = theta_CA2 + 2 * np.pi if theta_CA2 < -3/4*np.pi else theta_CA2
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta_CA2), np.abs(theta_CA2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
                
        if position_CB:
            w2 = 1
        else:
            if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2:                
                w2 = 1 - np.min([np.abs(theta_CB1_upper-theta_CB2), np.abs(theta_CB2-theta_CB1_lower)]) / (np.pi / 6)
            else: # 135~180和-180~-135的情况
                theta_CB2 = theta_CB2 + 2 * np.pi if theta_CB2 < -3/4*np.pi else theta_CB2
                w2 = 1 - np.min([np.abs(1.25*np.pi-theta_CB2), np.abs(theta_CB2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w3 = 1
        else:
            w3 = 1 - np.abs(angle_rad_C - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3
        
    return question, answer, check, score

def CR_three_back(A, B, C):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    C_desc, C_cloud = C["caption"], C["pcd"]
    C_desc = C_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    C_pos = C_cloud.get_center()
    C_pos[0] = -C_pos[0]; C_pos[1] = -C_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    C_rotation_matrix = C["rotation_matrix"]
    
    A_rotation_matrix = C_rotation_matrix.T @ A_rotation_matrix # 在C的坐标系下，A的旋转矩阵
    B_rotation_matrix = C_rotation_matrix.T @ B_rotation_matrix # 在C的坐标系下，B的旋转矩阵
    
    C_P_A = C_rotation_matrix.T @ (A_pos - C_pos) # 在C的坐标系下，A相对于C的位置
    C_P_B = C_rotation_matrix.T @ (B_pos - C_pos) # 在C的坐标系下，B相对于C的位置
    C_P_A_last = C['C_P_A']
    C_P_B_last = C['C_P_B']
    
    theta_CA1 = np.arctan2(C_P_A[2], C_P_A[0])
    theta_CB1 = np.arctan2(C_P_B[2], C_P_B[0])
    theta_CA2 = np.arctan2(C_P_A_last[2], C_P_A_last[0])
    theta_CB2 = np.arctan2(C_P_B_last[2], C_P_B_last[0])
    
    theta_CA1_upper = get_upper(theta_CA1)
    theta_CA1_lower = theta_CA1_upper - np.pi / 2
    if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2: 
        theta1_lower = theta_CA1_upper - np.pi / 2
        position_CA = True if theta_CA1_upper > theta_CA2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CA = True if theta_CA2 < -3/4*np.pi or theta_CA2 > 3/4*np.pi else False
        
    
    theta_CB1_upper = get_upper(theta_CB1)
    theta_CB1_lower = theta_CB1_upper - np.pi / 2
    if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2: 
        theta1_lower = theta_CB1_upper - np.pi / 2
        position_CB = True if theta_CB1_upper > theta_CB2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CB = True if theta_CB2 < -3/4*np.pi or theta_CB2 > 3/4*np.pi else False
    
    
    position = position_CA and position_CB
    
    
    max_angle = 30
    # angle_rad_AC = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    # angle_rad_BC = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    # same_direction_AC = angle_rad_AC < max_angle / 180 * np.pi
    # same_direction_BC = angle_rad_BC < max_angle / 180 * np.pi
    # same_direction = same_direction_AC and same_direction_BC
    
    
    angle_rad_C = np.arccos(np.clip(np.dot(C_rotation_matrix.T[0], np.array([0,0,1])), -1.0, 1.0))
    direction = angle_rad_C < max_angle / 180 * np.pi


    check = position and direction
    question_template = f"Are [A], [B] and [C] maintaining their original relative relationship when viewed from the back of [C]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc).replace("[C]", C_desc)
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position_CA:
            w1 = 1
        else:
            if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2:                
                w1 = 1 - np.min([np.abs(theta_CA1_upper-theta_CA2), np.abs(theta_CA2-theta_CA1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta_CA2 = theta_CA2 + 2 * np.pi if theta_CA2 < -3/4*np.pi else theta_CA2
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta_CA2), np.abs(theta_CA2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
                
        if position_CB:
            w2 = 1
        else:
            if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2:                
                w2 = 1 - np.min([np.abs(theta_CB1_upper-theta_CB2), np.abs(theta_CB2-theta_CB1_lower)]) / (np.pi / 6)
            else: # 135~180和-180~-135的情况
                theta_CB2 = theta_CB2 + 2 * np.pi if theta_CB2 < -3/4*np.pi else theta_CB2
                w2 = 1 - np.min([np.abs(1.25*np.pi-theta_CB2), np.abs(theta_CB2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w3 = 1
        else:
            w3 = 1 - np.abs(angle_rad_C - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3
        
    return question, answer, check, score

def CR_three_left(A, B, C):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    C_desc, C_cloud = C["caption"], C["pcd"]
    C_desc = C_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    C_pos = C_cloud.get_center()
    C_pos[0] = -C_pos[0]; C_pos[1] = -C_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    C_rotation_matrix = C["rotation_matrix"]
    
    A_rotation_matrix = C_rotation_matrix.T @ A_rotation_matrix # 在C的坐标系下，A的旋转矩阵
    B_rotation_matrix = C_rotation_matrix.T @ B_rotation_matrix # 在C的坐标系下，B的旋转矩阵
    
    C_P_A = C_rotation_matrix.T @ (A_pos - C_pos) # 在C的坐标系下，A相对于C的位置
    C_P_B = C_rotation_matrix.T @ (B_pos - C_pos) # 在C的坐标系下，B相对于C的位置
    C_P_A_last = C['C_P_A']
    C_P_B_last = C['C_P_B']
    
    theta_CA1 = np.arctan2(C_P_A[2], C_P_A[0])
    theta_CB1 = np.arctan2(C_P_B[2], C_P_B[0])
    theta_CA2 = np.arctan2(C_P_A_last[2], C_P_A_last[0])
    theta_CB2 = np.arctan2(C_P_B_last[2], C_P_B_last[0])
    
    theta_CA1_upper = get_upper(theta_CA1)
    theta_CA1_lower = theta_CA1_upper - np.pi / 2
    if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2: 
        theta1_lower = theta_CA1_upper - np.pi / 2
        position_CA = True if theta_CA1_upper > theta_CA2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CA = True if theta_CA2 < -3/4*np.pi or theta_CA2 > 3/4*np.pi else False
        
    
    theta_CB1_upper = get_upper(theta_CB1)
    theta_CB1_lower = theta_CB1_upper - np.pi / 2
    if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2: 
        theta1_lower = theta_CB1_upper - np.pi / 2
        position_CB = True if theta_CB1_upper > theta_CB2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CB = True if theta_CB2 < -3/4*np.pi or theta_CB2 > 3/4*np.pi else False
    
    
    position = position_CA and position_CB
    
    
    max_angle = 30
    # angle_rad_AC = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    # angle_rad_BC = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    # same_direction_AC = angle_rad_AC < max_angle / 180 * np.pi
    # same_direction_BC = angle_rad_BC < max_angle / 180 * np.pi
    # same_direction = same_direction_AC and same_direction_BC
    
    
    angle_rad_C = np.arccos(np.clip(np.dot(C_rotation_matrix.T[0], np.array([-1,0,0])), -1.0, 1.0))
    direction = angle_rad_C < max_angle / 180 * np.pi


    check = position and direction
    question_template = f"Are [A], [B] and [C] maintaining their original relative relationship when viewed from the left of [C]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc).replace("[C]", C_desc)
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position_CA:
            w1 = 1
        else:
            if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2:                
                w1 = 1 - np.min([np.abs(theta_CA1_upper-theta_CA2), np.abs(theta_CA2-theta_CA1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta_CA2 = theta_CA2 + 2 * np.pi if theta_CA2 < -3/4*np.pi else theta_CA2
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta_CA2), np.abs(theta_CA2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
                
        if position_CB:
            w2 = 1
        else:
            if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2:                
                w2 = 1 - np.min([np.abs(theta_CB1_upper-theta_CB2), np.abs(theta_CB2-theta_CB1_lower)]) / (np.pi / 6)
            else: # 135~180和-180~-135的情况
                theta_CB2 = theta_CB2 + 2 * np.pi if theta_CB2 < -3/4*np.pi else theta_CB2
                w2 = 1 - np.min([np.abs(1.25*np.pi-theta_CB2), np.abs(theta_CB2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w3 = 1
        else:
            w3 = 1 - np.abs(angle_rad_C - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3
        
    return question, answer, check, score

def CR_three_right(A, B, C):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    B_desc, B_cloud = B["caption"], B["pcd"]
    B_desc = B_desc.lower()
    C_desc, C_cloud = C["caption"], C["pcd"]
    C_desc = C_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
    C_pos = C_cloud.get_center()
    C_pos[0] = -C_pos[0]; C_pos[1] = -C_pos[1]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"]
    C_rotation_matrix = C["rotation_matrix"]
    
    A_rotation_matrix = C_rotation_matrix.T @ A_rotation_matrix # 在C的坐标系下，A的旋转矩阵
    B_rotation_matrix = C_rotation_matrix.T @ B_rotation_matrix # 在C的坐标系下，B的旋转矩阵
    
    C_P_A = C_rotation_matrix.T @ (A_pos - C_pos) # 在C的坐标系下，A相对于C的位置
    C_P_B = C_rotation_matrix.T @ (B_pos - C_pos) # 在C的坐标系下，B相对于C的位置
    C_P_A_last = C['C_P_A']
    C_P_B_last = C['C_P_B']
    
    theta_CA1 = np.arctan2(C_P_A[2], C_P_A[0])
    theta_CB1 = np.arctan2(C_P_B[2], C_P_B[0])
    theta_CA2 = np.arctan2(C_P_A_last[2], C_P_A_last[0])
    theta_CB2 = np.arctan2(C_P_B_last[2], C_P_B_last[0])
    
    theta_CA1_upper = get_upper(theta_CA1)
    theta_CA1_lower = theta_CA1_upper - np.pi / 2
    if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2: 
        theta1_lower = theta_CA1_upper - np.pi / 2
        position_CA = True if theta_CA1_upper > theta_CA2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CA = True if theta_CA2 < -3/4*np.pi or theta_CA2 > 3/4*np.pi else False
        
    
    theta_CB1_upper = get_upper(theta_CB1)
    theta_CB1_lower = theta_CB1_upper - np.pi / 2
    if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2: 
        theta1_lower = theta_CB1_upper - np.pi / 2
        position_CB = True if theta_CB1_upper > theta_CB2 > theta1_lower else False
    else: # 135~180和-180~-135的情况
        position_CB = True if theta_CB2 < -3/4*np.pi or theta_CB2 > 3/4*np.pi else False
    
    
    position = position_CA and position_CB
    
    
    max_angle = 30
    # angle_rad_AC = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    # angle_rad_BC = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    # same_direction_AC = angle_rad_AC < max_angle / 180 * np.pi
    # same_direction_BC = angle_rad_BC < max_angle / 180 * np.pi
    # same_direction = same_direction_AC and same_direction_BC
    
    
    angle_rad_C = np.arccos(np.clip(np.dot(C_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    direction = angle_rad_C < max_angle / 180 * np.pi


    check = position and direction
    question_template = f"Are [A], [B] and [C] maintaining their original relative relationship when viewed from the right of [C]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc).replace("[C]", C_desc)
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if position_CA:
            w1 = 1
        else:
            if theta_CA1_upper < np.pi and theta_CA1_upper > -np.pi/2:                
                w1 = 1 - np.min([np.abs(theta_CA1_upper-theta_CA2), np.abs(theta_CA2-theta_CA1_lower)]) / (np.pi / 6) # 30度的阈值
            else: # 135~180和-180~-135的情况
                theta_CA2 = theta_CA2 + 2 * np.pi if theta_CA2 < -3/4*np.pi else theta_CA2
                w1 = 1 - np.min([np.abs(1.25*np.pi-theta_CA2), np.abs(theta_CA2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
                
        if position_CB:
            w2 = 1
        else:
            if theta_CB1_upper < np.pi and theta_CB1_upper > -np.pi/2:                
                w2 = 1 - np.min([np.abs(theta_CB1_upper-theta_CB2), np.abs(theta_CB2-theta_CB1_lower)]) / (np.pi / 6)
            else: # 135~180和-180~-135的情况
                theta_CB2 = theta_CB2 + 2 * np.pi if theta_CB2 < -3/4*np.pi else theta_CB2
                w2 = 1 - np.min([np.abs(1.25*np.pi-theta_CB2), np.abs(theta_CB2-0.75*np.pi)]) / (np.pi / 6) # 30度的阈值
            
        if direction:
            w3 = 1
        else:
            w3 = 1 - np.abs(angle_rad_C - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3
        
    return question, answer, check, score