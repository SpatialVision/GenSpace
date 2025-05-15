import random
from itertools import combinations

import numpy as np
from osdsynth.processor.pointcloud import calculate_distances_between_point_clouds, human_like_distance
from osdsynth.processor.prompt_template import *
from osdsynth.processor.prompt_utils import *
from osdsynth.processor.prompt_spatitalbench_template import *

def camera_front_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 15
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_front = angle_rad < max_angle / 180 * np.pi

    check = is_front

    question_template = f"Does the camera face the front of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(45-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_back_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 15
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_back = angle_rad < max_angle / 180 * np.pi

    check = is_back

    question_template = f"Does the camera face the back of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(45-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_left_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_left = angle_rad < max_angle / 180 * np.pi

    check = is_left

    question_template = f"Does the camera face the left of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(60-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_right_camera_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_right = angle_rad < max_angle / 180 * np.pi

    check = is_right

    question_template = f"Does the camera face the right of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(60-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  



def camera_front_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 15
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_front = angle_rad < max_angle / 180 * np.pi

    check = is_front

    question_template = f"Does the camera face the front of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(45-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_back_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 15
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_back = angle_rad < max_angle / 180 * np.pi

    check = is_back

    question_template = f"Does the camera face the back of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(45-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_left_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(-A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_left = angle_rad < max_angle / 180 * np.pi

    check = is_left

    question_template = f"Does the camera face the left of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(60-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_right_object_center(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    A_rotation_matrix = A["rotation_matrix"]
    
    max_angle = 30
    angle_rad = np.arccos(np.clip(np.dot(A_rotation_matrix.T[2], np.array([0,0,-1])), -1.0, 1.0))
    is_right = angle_rad < max_angle / 180 * np.pi

    check = is_right

    question_template = f"Does the camera face the right of [A]?"
    question = question_template.replace("[A]", A_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi-max_angle)/(60-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  



def object_side_by_side_same_direction(A, B):
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

    max_angle = 30


    side_by_side_radius = np.abs(np.arctan(B_P_A[2]/ B_P_A[0]))
    is_side_by_side = side_by_side_radius > (90 - max_angle) / 180 * np.pi

    # 比较X轴的夹角
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
        w1 = 1 - np.abs(side_by_side_radius - (90 - max_angle) / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w2 = 1 - np.abs(angle_rad - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        score = 0 if w1 < 0 or w2 < 0 else w1 * w2
        score = 0 if score < 0 else score

    return question, answer, check, score    

def object_side_by_side_opposite_direction(A, B):
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

    max_angle = 30
    

    side_by_side_radius = np.abs(np.arctan(B_P_A[2]/ B_P_A[0]))
    is_side_by_side = side_by_side_radius > (90 - max_angle) / 180 * np.pi

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
        w1 = 1 - np.abs(side_by_side_radius - (90 - max_angle) / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w2 = 1 - np.abs(angle_rad - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        score = 0 if w1 < 0 or w2 < 0 else w1 * w2
        score = 0 if score < 0 else score

    return question, answer, check, score  

def object_face_to_face(A, B):
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
    
    max_angle = 30
    
    
    face_to_face_radius = np.abs(np.arctan(B_P_A[2]/ B_P_A[0]))
    is_line =  B_P_A[0] > 0 and face_to_face_radius < max_angle * np.pi# 在一条线上，且A在B的前面
    
    
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
        w1 = 1 if  B_P_A[0] > 0 else -1
        w2 = 1 - np.abs(face_to_face_radius - max_angle * np.pi) / (np.pi / 12) # 15度的阈值
        w3 = 1 - np.abs(angle_rad - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        score = 0 if w1 < 0 or w2 < 0 or w3 < 0 else w1 * w2 * w3
        score = 0 if score < 0 else score

    return question, answer, check, score  
    
def object_back_to_back(A, B):
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

    max_angle = 30

    face_to_face_radius = np.abs(np.arctan(B_P_A[2]/ B_P_A[0]))
    is_line =  B_P_A[0] < 0 and face_to_face_radius < max_angle / 180 * np.pi# 在一条线上，且A在B的前面
    
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
        w1 = 1 if  B_P_A[0] < 0 else -1
        w2 = 1 - np.abs(face_to_face_radius - max_angle * np.pi) / (np.pi / 12) # 15度的阈值
        w3 = 1 - np.abs(angle_rad - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        score = 0 if w1 < 0 or w2 < 0 or w3 < 0 else w1 * w2 * w3
        score = 0 if score < 0 else score

    return question, answer, check, score  



def object_front(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    A_rotation_matrix = A["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，B物体的位置

    max_angle = 15
    A_P_B_direcetion = A_P_B / np.linalg.norm(A_P_B)
    angle_rad = np.arccos(np.clip(np.dot(A_P_B_direcetion, np.array([1,0,0])), -1.0, 1.0))
    
    B_is_in_front_A = A_P_B[0] > 0 and angle_rad < max_angle / 180 * np.pi# 在一条线上，且A在B的前面
    
    check = B_is_in_front_A
    
    question_template = f"Is [B] in front of [A], from the view of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score  

def object_back(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    A_rotation_matrix = A["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，B物体的位置

    max_angle = 15
    A_P_B_direcetion = A_P_B / np.linalg.norm(A_P_B)
    angle_rad = np.arccos(np.clip(np.dot(A_P_B_direcetion, np.array([-1,0,0])), -1.0, 1.0))
    B_is_in_back_A = A_P_B[0] < 0 and angle_rad < max_angle / 180 * np.pi# 在一条线上，且A在B的前面
    
    check = B_is_in_back_A
    
    question_template = f"Is [B] in back of [A], from the view of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (45-max_angle))
        score = 0 if score < 0 else score

    return question, answer, check, score

def object_left(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    A_rotation_matrix = A["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，B物体的位置

    max_angle = 30
    A_P_B_direcetion = A_P_B / np.linalg.norm(A_P_B)
    angle_rad = np.arccos(np.clip(np.dot(A_P_B_direcetion, np.array([0,0,-1])), -1.0, 1.0))
    B_is_in_left_A = A_P_B[2] < 0 and angle_rad < max_angle / 180 * np.pi# 在一条线上，且A在B的前面
    
    check = B_is_in_left_A
    
    question_template = f"Is [B] on the left of [A], from the view of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (60-max_angle))
        score = 0 if score < 0 or A_P_B[2] > 0 else score

    return question, answer, check, score  

def object_right(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # A_rotation_matrix = A["rotation_matrix"]
    A_rotation_matrix = A["rotation_matrix"] 
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A物体参考系下，B物体的位置


    max_angle = 30
    A_P_B_direcetion = A_P_B / np.linalg.norm(A_P_B)
    angle_rad = np.arccos(np.clip(np.dot(A_P_B_direcetion, np.array([0,0,1])), -1.0, 1.0))
    B_is_in_right_A = A_P_B[2] > 0 and angle_rad < max_angle / 180 * np.pi# 在一条线上，且A在B的前面
    
    check = B_is_in_right_A
    
    question_template = f"Is [B] on the right of [A], from the view of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs((angle_rad*180/np.pi - max_angle) / (60-max_angle))
        score = 0 if score < 0 or A_P_B[2] < 0 else score

    return question, answer, check, score  



def camera_two_objects_closer(A,B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # 计算距离
    distance_a = np.linalg.norm(A_pos)
    distance_b = np.linalg.norm(B_pos)
    
    
    check = distance_a < distance_b

    question_template = f"Is [A] closer to the camera than [B]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    score = 1 if check else 0
    
    return question, answer, check, score

def camera_two_objects_farther(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # 计算距离
    distance_a = np.linalg.norm(A_pos)
    distance_b = np.linalg.norm(B_pos)
    
    check = distance_a > distance_b

    question_template = f"Is [A] farther to the camera than [B]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    score = 1 if check else 0
    
    return question, answer, check, score

def camera_two_objects_left(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    
    check = B_pos[0] - A_pos[0] > 0 

    question_template = f"Is [A] on the left of [B], from the view of the camera?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 1 if check else 0
    
    return question, answer, check, score

def camera_two_objects_right(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    
    check = B_pos[0] - A_pos[0] < 0 

    question_template = f"Is [A] on the right of [B], from the view of the camera?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 1 if check else 0
    
    return question, answer, check, score



def object_apart_0_5meter(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # 计算距离
    distance = np.linalg.norm(A_pos - B_pos)
    
    delta = 1.0/3
    gt_distance = 0.5
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] apart from [B] about 0.5 meter?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def object_apart_1meter(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # 计算距离
    distance = np.linalg.norm(A_pos - B_pos)
    
    delta = 1.0/3
    gt_distance = 1
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] apart from [B] about 1 meter?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def object_apart_1_5meter(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # 计算距离
    distance = np.linalg.norm(A_pos - B_pos)
    
    delta = 1.0/3
    gt_distance = 1.5
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] apart from [B] about 0.5 meter?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score

def object_apart_2meter(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    B_pos = B_cloud.get_center()
    B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]

    # 计算距离
    distance = np.linalg.norm(A_pos - B_pos)
    
    delta = 1.0/3
    gt_distance = 2
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] apart from [B] about 2 meters?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  


def camera_1meter_away(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    delta = 1.0/3
    gt_distance = 1

    distance = np.linalg.norm(A_pos)

    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is the camera about 1 meter away from [A]?"
    question = question_template.replace("[A]", A_desc)
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_2meter_away(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    delta = 1.0/3
    gt_distance = 2

    distance = np.linalg.norm(A_pos)

    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is the camera about 2 meter away from [A]?"
    question = question_template.replace("[A]", A_desc)
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_3meter_away(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    delta = 0.3
    gt_distance = 3

    distance = np.linalg.norm(A_pos)

    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is the camera about 3 meter away from [A]?"
    question = question_template.replace("[A]", A_desc)
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def camera_4meter_away(A):
    A_desc, A_cloud = A["caption"], A["pcd"]
    A_desc = A_desc.lower()

    # 从PyTorch3D的坐标系转换到OpenCV的坐标系
    A_pos = A_cloud.get_center()
    A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
    
    delta = 0.2
    gt_distance = 4

    distance = np.linalg.norm(A_pos)

    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is the camera about 4 meter away from [A]?"
    question = question_template.replace("[A]", A_desc)
    answer = "Yes" if check else "No"

    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  



def object_bigger_than1_2(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 计算距离
    A_rotation_matrix = A["rotation_matrix"]
    theta_A = np.arctan2(A_rotation_matrix.T[0][2], A_rotation_matrix.T[0][0])
    A_center = A["pcd"].get_center()
    R = A["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_A)) 
    A["pcd"] = A["pcd"].rotate(R)
    A_length = A["pcd"].get_axis_aligned_bounding_box().get_extent()[0]
    A_height = A["pcd"].get_axis_aligned_bounding_box().get_extent()[1]
    A_width = A["pcd"].get_axis_aligned_bounding_box().get_extent()[2]
    A_volume = A_length * A_height * A_width
    
    B_rotation_matrix = B["rotation_matrix"]
    theta_B = np.arctan2(B_rotation_matrix.T[0][2], B_rotation_matrix.T[0][0])
    B_center = B["pcd"].get_center()
    R = B["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_B))
    B["pcd"] = B["pcd"].rotate(R)
    B_length = B["pcd"].get_axis_aligned_bounding_box().get_extent()[0]
    B_height = B["pcd"].get_axis_aligned_bounding_box().get_extent()[1]
    B_width = B["pcd"].get_axis_aligned_bounding_box().get_extent()[2]
    B_volume = B_length * B_height * B_width
    
    if A_volume > B_volume:
        distance = A_volume / B_volume
    else:
        distance = B_volume / A_volume
    
    delta = 1.0/3
    gt_distance = 1.2
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] bigger than [B] about 0.2 times?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def object_higher_20cm(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 计算距离
        
    A_height = A["pcd"].get_axis_aligned_bounding_box().get_extent()[1]
    B_height = B["pcd"].get_axis_aligned_bounding_box().get_extent()[1]
    distance = np.abs(A_height-B_height)
    
    delta = 1.0/3
    gt_distance = 0.2
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] higher 20cm than [B]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def object_longer_50cm(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 计算距离
        
    A_rotation_matrix = A["rotation_matrix"]
    theta_A = np.arctan2(A_rotation_matrix.T[0][2], A_rotation_matrix.T[0][0])
    A_center = A["pcd"].get_center()
    R = A["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_A)) 
    A["pcd"] = A["pcd"].rotate(R)
    A_length = A["pcd"].get_axis_aligned_bounding_box().get_extent()[0]
    
    B_rotation_matrix = B["rotation_matrix"]
    theta_B = np.arctan2(B_rotation_matrix.T[0][2], B_rotation_matrix.T[0][0])
    B_center = B["pcd"].get_center()
    R = B["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_B))
    B["pcd"] = B["pcd"].rotate(R)
    B_length = B["pcd"].get_axis_aligned_bounding_box().get_extent()[0]
    
    distance = np.abs(A_length-B_length)
    
    delta = 1.0/3
    gt_distance = 0.5
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] longer 50cm than [B]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  

def object_wider_30cm(A, B):
    A_desc, A_cloud = A["caption"], A["pcd"]
    B_desc, B_cloud = B["caption"], B["pcd"]
    A_desc, B_desc = A_desc.lower(), B_desc.lower()

    # 计算距离
        
    A_rotation_matrix = A["rotation_matrix"]
    theta_A = np.arctan2(A_rotation_matrix.T[0][2], A_rotation_matrix.T[0][0])
    A_center = A["pcd"].get_center()
    R = A["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_A)) 
    A["pcd"] = A["pcd"].rotate(R)
    A_width = A["pcd"].get_axis_aligned_bounding_box().get_extent()[2]
    
    B_rotation_matrix = B["rotation_matrix"]
    theta_B = np.arctan2(B_rotation_matrix.T[0][2], B_rotation_matrix.T[0][0])
    B_center = B["pcd"].get_center()
    R = B["pcd"].get_rotation_matrix_from_xyz((0, 0, theta_B))
    B["pcd"] = B["pcd"].rotate(R)
    B_width = B["pcd"].get_axis_aligned_bounding_box().get_extent()[2]
    
    distance = np.abs(A_width-B_width)
    
    delta = 1.0/3
    gt_distance = 0.3
    
    check = (1-delta)*gt_distance < distance and distance < (1+delta)*gt_distance

    question_template = f"Is [A] wider 30cm than [B]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        score = 1 - 1*np.abs(((distance - gt_distance) / gt_distance)- delta)/delta
        score = 0 if score < 0 else score

    return question, answer, check, score  


def side_by_side_front(A, B):
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
    
    max_angle = 30
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    side_by_side_radius = np.abs(np.arctan(A_P_B[2]/ A_P_B[0]))
    is_side_by_side = side_by_side_radius > (90 - max_angle) * np.pi / 180
    
    same_direction_radius = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    is_same_direction = same_direction_radius < max_angle * np.pi / 180 # 30度的阈值
    
    front_radius = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,-1])), -1.0, 1.0))
    is_front = front_radius < max_angle * np.pi / 180 # 30度的阈值
    
    check = is_side_by_side and is_same_direction and is_front
    
    question_template = f"Is [A] and [B] side-by-side and same-orientation with viewed from the front of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        w1 = 1 - np.abs(side_by_side_radius - (90 - max_angle) / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w2 = 1 - np.abs(same_direction_radius - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w3 = 1 - np.abs(front_radius - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3

    return question, answer, check, score

def side_by_side_left(A, B):
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
    
    max_angle = 30
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    side_by_side_radius = np.abs(np.arctan(A_P_B[2]/ A_P_B[0]))
    is_side_by_side = side_by_side_radius > (90 - max_angle) * np.pi / 180
    
    same_direction_radius = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    is_same_direction = same_direction_radius < max_angle * np.pi / 180 # 30度的阈值
    
    left_radius = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([-1,0,0])), -1.0, 1.0))
    is_left = left_radius < max_angle * np.pi / 180 # 30度的阈值
    
    check = is_side_by_side and is_same_direction and is_left
    
    question_template = f"Is [A] and [B] side-by-side and same-orientation with viewed from the left of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        w1 = 1 - np.abs(side_by_side_radius - (90 - max_angle) / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w2 = 1 - np.abs(same_direction_radius - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w3 = 1 - np.abs(left_radius - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3

    return question, answer, check, score

def side_by_side_right(A, B):
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
    
    max_angle = 30
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    side_by_side_radius = np.abs(np.arctan(A_P_B[2]/ A_P_B[0]))
    is_side_by_side = side_by_side_radius > (90 - max_angle) * np.pi / 180
    
    same_direction_radius = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    is_same_direction = same_direction_radius < max_angle * np.pi / 180 # 30度的阈值
    
    right_radius = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    is_right = right_radius < max_angle * np.pi / 180 # 30度的阈值
    
    check = is_side_by_side and is_same_direction and is_right
    
    question_template = f"Is [A] and [B] side-by-side and same-orientation with viewed from the right of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        w1 = 1 - np.abs(side_by_side_radius - (90 - max_angle) / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w2 = 1 - np.abs(same_direction_radius - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w3 = 1 - np.abs(right_radius - max_angle / 180 * np.pi) / (np.pi / 6) # 30度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3

    return question, answer, check, score


def side_by_side_back(A, B):
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
    
    max_angle = 30
    
    A_P_B = A_rotation_matrix.T @ (B_pos - A_pos) # 在A的坐标系下，B相对于A的位置
    side_by_side_radius = np.abs(np.arctan(A_P_B[2]/ A_P_B[0]))
    is_side_by_side = side_by_side_radius > (90 - max_angle) * np.pi / 180
    
    same_direction_radius = np.arccos(np.clip(np.dot(B_rotation_matrix.T[0], np.array([1,0,0])), -1.0, 1.0))
    is_same_direction = same_direction_radius < max_angle * np.pi / 180 # 30度的阈值
    
    back_radius = np.arccos(np.clip(np.dot(A_rotation_matrix.T[0], np.array([0,0,1])), -1.0, 1.0))
    is_back = back_radius < max_angle * np.pi / 180 # 30度的阈值
    
    check = is_side_by_side and is_same_direction and is_back
    
    question_template = f"Is [A] and [B] side-by-side and same-orientation with viewed from the back of [A]?"
    question = question_template.replace("[A]", A_desc).replace("[B]", B_desc)
    
    answer = "Yes" if check else "No"
    
    score = 0
    if check:
        score = 1
    else:
        w1 = 1 - np.abs(side_by_side_radius - (90 - max_angle) / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w2 = 1 - np.abs(same_direction_radius - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        w3 = 1 - np.abs(back_radius - max_angle / 180 * np.pi) / (np.pi / 12) # 15度的阈值
        score = 0 if w1<0 or w2<0 or w3<0 else w1 * w2 * w3

    return question, answer, check, score
