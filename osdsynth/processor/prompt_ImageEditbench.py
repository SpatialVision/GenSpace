import random
from itertools import combinations

import numpy as np
from osdsynth.processor.pointcloud import calculate_distances_between_point_clouds, human_like_distance
from osdsynth.processor.prompt_template import *
from osdsynth.processor.prompt_utils import *
from osdsynth.processor.prompt_spatitalbench_template import *

def camera_to_front_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_front_view = angle_rad < max_angle / 180 * np.pi

    check = is_front_view

    question_template = f"Is the camera facing the front of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score    

def camera_to_left_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_left_view = angle_rad < max_angle / 180 * np.pi

    check = is_left_view

    question_template = f"Is the camera facing the left of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score 

def camera_to_right_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_right_view = angle_rad < max_angle / 180 * np.pi

    check = is_right_view

    question_template = f"Is the camera facing the right of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score 

def camera_to_back_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_back_view = angle_rad < max_angle / 180 * np.pi

    check = is_back_view

    question_template = f"Is the camera facing the back of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score

def camera_to_front_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_front_view = angle_rad < max_angle / 180 * np.pi

    check = is_front_view

    question_template = f"Is the camera facing the front of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score    

def camera_to_left_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_left_view = angle_rad < max_angle / 180 * np.pi

    check = is_left_view

    question_template = f"Is the camera facing the left of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score 

def camera_to_right_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_right_view = angle_rad < max_angle / 180 * np.pi

    check = is_right_view

    question_template = f"Is the camera facing the right of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score 

def camera_to_back_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()
    
    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_back_view = angle_rad < max_angle / 180 * np.pi

    check = is_back_view

    question_template = f"Is the camera facing the back of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 else score

    return question, answer, check, score

def object_insert_side_by_side_same_orientation(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"] 
    B_P_A = B_rotation_matrix.T @ (A_pos - B_pos) # 在B物体参考系下，A物体的位置
    A_rotation_matrix = B_rotation_matrix.T @ A["rotation_matrix"]

    is_side_by_side = np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) > np.pi * 1 / 3

    # 比较X轴的夹角
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    is_same_orientation = angle_rad < max_angle / 180 * np.pi
    
    check = is_same_orientation and is_side_by_side
    
    question_template = f"Is [A] and [B] side by side, facing the same direction?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if is_same_orientation:
            w1 = 1
        else:
            w1 = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        
        if is_side_by_side:
            w2 = 1
        else:
            w2 = 1 - 1*np.abs((np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) - np.pi * 1 / 3) / (np.pi/12))
        score = 0 if w1 < 0 or w2<0 else w1*w2

    return question, answer, check, score   

def object_insert_side_by_side_opposite_orientation(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"] 
    B_P_A = B_rotation_matrix.T @ (A_pos - B_pos) # 在B物体参考系下，A物体的位置
    A_rotation_matrix = B_rotation_matrix.T @ A["rotation_matrix"]

    is_side_by_side = np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) > np.pi * 1 / 3

    # 比较X轴的夹角
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([-1,0,0])), -1.0, 1.0))
    is_opposite_orientation = angle_rad < max_angle / 180 * np.pi

    check = is_opposite_orientation and is_side_by_side

    question_template = f"Is [A] and [B] side by side, facing the opposite direction?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        if is_opposite_orientation:
            w1 = 1
        else:
            w1 = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        if is_side_by_side:
            w2 = 1
        else:
            w2 = 1 - 1*np.abs((np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) - np.pi * 1 / 3) / (np.pi/12))
        score = 0 if w1 < 0 or w2<0 else w1*w2

    return question, answer, check, score  

def object_insert_face_to_face(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"] 
    B_P_A = B_rotation_matrix.T @ (A_pos - B_pos) # 在B物体参考系下，A物体的位置
    A_rotation_matrix = B_rotation_matrix.T @ A["rotation_matrix"]
    
    
    is_line =  B_P_A[0] > 0 and np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) < np.pi/3# 在一条线上，且A在B的前面
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], [-1,0,0]), -1.0, 1.0))
    is_opposite_orientation = angle_rad < max_angle / 180 * np.pi
    
    check = is_opposite_orientation and is_line
    
    question_template = f"Is [A] and [B] face to face?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if is_opposite_orientation:
            w1 = 1
        else:
            w1 = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        if is_line:
            w2 = 1
        else:
            w2 = 1 - 1*np.abs((np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) - np.pi * 1 / 3) / (np.pi/12))
            if B_P_A[0] < 0 or np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) > np.pi/3:
                w2 = 0
        score = 0 if w1<0 or w2<0 else w1*w2

    return question, answer, check, score  

def object_insert_back_to_back(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    B_rotation_matrix = B["rotation_matrix"] 
    B_P_A = B_rotation_matrix.T @ (A_pos - B_pos) # 在B物体参考系下，A物体的位置
    A_rotation_matrix = B_rotation_matrix.T @ A["rotation_matrix"]


    is_line =  B_P_A[0] < 0 and np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) < np.pi/3# 在一条线上，且A在B的前面
    
    max_angle = 30
    angle_rad =  np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], [-1,0,0]), -1.0, 1.0))
    is_opposite_orientation = angle_rad < max_angle / 180 * np.pi
    
    check = is_opposite_orientation and is_line
    
    question_template = f"Is [A] and [B] back to back?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        if is_opposite_orientation:
            w1 = 1
        else:
            w1 = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        if is_line:
            w2 = 1
        else:
            w2 = 1 - 1*np.abs((np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) - np.pi * 1 / 3) / (np.pi/12))
            if B_P_A[0] > 0 or np.abs(np.arctan(B_P_A[2]/ B_P_A[0])) > np.pi/3:
                w2 = 0
        score = 0 if w1<0 or w2<0 else w1*w2

    return question, answer, check, score  

def object_insert_front_object_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    A_rotation_matrix = A["rotation_matrix"]
    # B_rotation_matrix = B["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，A物体的位置


    is_front =  A_P_B[0] > 0
        
    check = is_front
    
    question_template = f"Is [B] in front of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
        
    return question, answer, check, score

def object_insert_left_object_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    A_rotation_matrix = A["rotation_matrix"]
    # B_rotation_matrix = B["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，A物体的位置
    
    max_angle = 30
    A_P_B_direcetion = A_P_B / np.linalg.norm(A_P_B)
    angle_rad = np.arccos(np.clip(np.dot(A_P_B_direcetion, np.array([0,0,-1])), -1.0, 1.0))
    B_is_in_left_A = A_P_B[2] < 0 and angle_rad < max_angle / 180 * np.pi# 在一条线上，且A在B的前面
    
    is_left =  A_P_B[2] < 0 and B_is_in_left_A
        
    check = is_left
    
    question_template = f"Is [B] in the left of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 or A_P_B[2] > 0 else score

    return question, answer, check, score  

def object_insert_right_object_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    A_rotation_matrix = A["rotation_matrix"]
    # B_rotation_matrix = B["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，A物体的位置
    
    max_angle = 30
    A_P_B_direcetion = A_P_B / np.linalg.norm(A_P_B)
    angle_rad = np.arccos(np.clip(np.dot(A_P_B_direcetion, np.array([0,0,1])), -1.0, 1.0))
    B_is_in_right_A = A_P_B[2] > 0 and angle_rad < max_angle / 180 * np.pi# 在一条线上，且A在B的前面
    
    is_right =  A_P_B[2] > 0 and B_is_in_right_A
        
    check = is_right
    
    question_template = f"Is [B] in the right of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45))
        score = 0 if score < 0 or A_P_B[2] < 0 else score

    return question, answer, check, score  

def object_insert_behind_object_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    A_rotation_matrix = A["rotation_matrix"]
    # B_rotation_matrix = B["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，A物体的位置

    is_behind =  A_P_B[0] < 0
        
    check = is_behind
    
    question_template = f"Is [B] behind [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    
    return question, answer, check, score


## 描述物体
# 定性

def object_insert_front_camera_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    is_front =  B_pos[2] < A_pos[2]
        
    check = is_front
    
    question_template = f"Is [B] in front of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
        
    return question, answer, check, score

def object_insert_left_camera_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]


    is_left =  B_pos[0] < A_pos[0]
        
    check = is_left
    
    question_template = f"Is [B] in left of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
        
    return question, answer, check, score

def object_insert_right_camera_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    is_right =  B_pos[0] > A_pos[0]
        
    check = is_right
    
    question_template = f"Is [B] in right of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
        
    return question, answer, check, score

def object_insert_behind_camera_center(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]


    is_behind =  B_pos[2] > A_pos[2]
        
    check = is_behind
    
    question_template = f"Is [B] in behind of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
        
    return question, answer, check, score

def objectmove_close_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[2] -A['last_pos'][2]
    
    delta = 1.0/3
    gt_distance = -1
    
    check = (1+delta)*gt_distance < distance and distance < (1-delta)*gt_distance

    question_template = f"Does [A] move 1 meter close to the camera?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def objectmove_far_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[2] -A['last_pos'][2]
    
    delta = 1.0/3
    gt_distance = 1
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Does [A] move 1 meter far to the camera?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def objectmove_left_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[0] -A['last_pos'][0]
    
    delta = 1.0/3
    gt_distance = -1
    
    check = (1+delta)*gt_distance < distance and distance < (1-delta)*gt_distance

    question_template = f"Does [A] move 1 meter left?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def objectmove_right_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[0] -A['last_pos'][0]
    
    delta = 1.0/3
    gt_distance = 1
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Does [A] move 1 meter right?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score


def camera_forward_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[2] - A['last_pos'][2]
    
    delta = 1.0/3
    gt_distance = -1
    
    check = (1+delta)*gt_distance < distance and distance < (1-delta)*gt_distance

    question_template = f"Does camera move 1 meter forward? [A]"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def camera_leftward_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[0] - A['last_pos'][0]
    
    delta = 1.0/3
    gt_distance = 1
    
    check = (1+delta)*gt_distance < distance and distance < (1-delta)*gt_distance

    question_template = f"Does camera move 1 meter leftward? [A]"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def camera_rightward_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[0] - A['last_pos'][0]
    
    delta = 1.0/3
    gt_distance = -1
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Does camera move 1 meter rightward? [A]"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def camera_backward_1meter(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    distance = A_pos[2] - A['last_pos'][2]
    
    delta = 1.0/3
    gt_distance = 1
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Does camera move 1 meter backward? [A]"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score


# 定量
def object_make_12bigger(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    
    
    A_rotation_matrix = A["rotation_matrix"]
    theta_A = np.arctan2(A_rotation_matrix.T[0][2], A_rotation_matrix.T[0][0])
    A_center = A["pcd"].get_center()
    R = A["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_A)) 
    A["pcd"] = A["pcd"].rotate(R)
    A_length = A["pcd"].get_axis_aligned_bounding_box().get_extent()[0]
    A_height = A["pcd"].get_axis_aligned_bounding_box().get_extent()[1]
    A_width = A["pcd"].get_axis_aligned_bounding_box().get_extent()[2]
    volume = A_length * A_height * A_width
    
    distance = volume / A['last_volume'] - 1
    
    delta = 1.0/3
    gt_distance = 0.2
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Does [A] become 1.5 times its initial dimensions.?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def object_make_20cm_higher(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 计算距离
        
    height = A["pcd"].get_axis_aligned_bounding_box().get_extent()[1]
    last_height = A["last_height"]
    distance = height-last_height 
    
    delta = 1.0/3
    gt_distance = 0.2
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] higher 20cm than [B]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def object_make_50cm_longer(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 计算距离
        
    A_rotation_matrix = A["rotation_matrix"]
    theta_A = np.arctan2(A_rotation_matrix.T[0][2], A_rotation_matrix.T[0][0])
    A_center = A["pcd"].get_center()
    R = A["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_A)) 
    A["pcd"] = A["pcd"].rotate(R)
    length = A["pcd"].get_axis_aligned_bounding_box().get_extent()[0]
    
    last_length = A["last_length"]
    distance = length-last_length 
    
    delta = 1.0/3
    gt_distance = 0.5
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] higher 20cm than [B]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def object_make_40cm_wider(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 计算距离
        
    A_rotation_matrix = A["rotation_matrix"]
    theta_A = np.arctan2(A_rotation_matrix.T[0][2], A_rotation_matrix.T[0][0])
    A_center = A["pcd"].get_center()
    R = A["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_A)) 
    A["pcd"] = A["pcd"].rotate(R)
    width = A["pcd"].get_axis_aligned_bounding_box().get_extent()[2]
    
    last_width = A["last_width"]
    distance = width-last_width 
    
    delta = 1.0/3
    gt_distance = 0.4
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] higher 20cm than [B]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score