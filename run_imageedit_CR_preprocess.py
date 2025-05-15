import argparse
import glob
import json
import os
import pickle
import random
import time
import warnings
import torch
from multiprocessing.pool import Pool

import cv2
import numpy as np

from mmengine import Config
from osdsynth.processor.captions import CaptionImage
from osdsynth.processor.pointcloud import PointCloudReconstruction
from osdsynth.processor.prompt import ImageEditPromptGenerator

# from osdsynth.processor.filter import FilterImage
from osdsynth.processor.segment import SegmentImage
from osdsynth.Orient_Anything.Rotation import Rotation
from osdsynth.utils.logger import SkipImageException, save_detection_list_to_json, setup_logger
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as T

import re

# Suppressing all warnings
warnings.filterwarnings("ignore")


def main(args):
    """Main function to control the flow of the program."""
    # Parse arguments
    cfg = Config.fromfile(args.config)
    exp_name = args.name if args.name else args.timestamp

    # Create log folder
    cfg.log_folder = os.path.join(args.log_dir, exp_name)
    os.makedirs(os.path.abspath(cfg.log_folder), exist_ok=True)

    # Create Wis3D folder
    cfg.vis = args.vis
    cfg.wis3d_folder = os.path.join(args.log_dir, "Wis3D")
    os.makedirs(os.path.abspath(cfg.wis3d_folder), exist_ok=True)

    # Init the logger and log some basic info
    cfg.log_file = os.path.join(cfg.log_folder, f"{exp_name}_{args.timestamp}.log")
    logger = setup_logger()  # cfg.log_file
    logger.info(f"Config:\n{cfg.pretty_text}")

    # Dump config to log
    cfg.dump(os.path.join(cfg.log_folder, os.path.basename(args.config)))

    # Create output folder
    cfg.exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(os.path.abspath(cfg.exp_dir), exist_ok=True)

    # Create folder for output json
    cfg.json_folder = os.path.join(cfg.exp_dir, "json")
    os.makedirs(os.path.abspath(cfg.json_folder), exist_ok=True)

    device = "cuda"

    segmenter = SegmentImage(cfg, logger, device)
    rotation = Rotation(cfg, logger, device)
    reconstructor = PointCloudReconstruction(cfg, logger, device)
    captioner = CaptionImage(cfg, logger, device)
    prompter = ImageEditPromptGenerator(cfg, logger, device)


    for class_index in range(4):
        if not os.path.exists(f"{args.input}/{class_index}"):
            continue
        
        # if class_index not in [0]:
        #     continue

        global_data = glob.glob(f"{args.input}/{class_index}/*.jpg") + glob.glob(f"{args.input}/{class_index}/*.png")

        annotate(cfg, segmenter, rotation, reconstructor, captioner, prompter, global_data, logger, device, class_index)


preprocess_base_path = '/data2/projects/SpatialRGPT/dataset_pipeline/output_file/CR/preprocess'

def annotate(cfg, segmenter, rotation, reconstructor, captioner, prompter, global_data, logger, device, class_index):


    random.shuffle(global_data)

    all_sum = 0 


    for i, filepath in tqdm(enumerate(global_data), ncols=25):
        
        all_sum = all_sum+1
        postfix = filepath.split("/")[-3]
        filename = filepath.split("/")[-1].split(".")[0]
        if filename != '8':
            continue
        print(f"\n\nProcessing file: {filename}")

        progress_file_path = os.path.join(cfg.log_folder, f"{filename}.progress")
        if os.path.exists(progress_file_path) and cfg.check_exist:
            continue
        
        preprocess_path = os.path.join(preprocess_base_path, filepath.split('/')[-3], str(class_index))
        os.makedirs(preprocess_path, exist_ok=True)
        preprocess_path = os.path.join(preprocess_path, f'{filename}.npy')

        image_bgr = cv2.imread(filepath)
        image_bgr = cv2.resize(image_bgr, (int(640 / (image_bgr.shape[0]) * (image_bgr.shape[1])), 640))

        try:
            is_three = False if int(filename)%2 == 0 else True
            
           
            # two_class = filepath.split("/")[-1].split('_')[:2]
            with open (f"{filepath.replace('png','txt').replace('jpg','txt')}", "r") as f:
                s = f.read()
                two_class = re.findall(r'<(.*?)>', s)
                

            # Run tagging model and get openworld detections
            
            vis_som, detection_list = segmenter.process(image_bgr, two_class)
            
            detection_list = rotation.process(detection_list)

            # Lift 2D to 3D, 3D bbox informations are included in detection_list
            detection_list, angles, xz_max_min = reconstructor.process(filename, image_bgr, detection_list)

            is_one = False

            tmp_detection_list = []
            for i in range(len(detection_list)):
                for j in range(len(detection_list)):
                    if two_class[i] == detection_list[j]['class_name'][:-1]:
                        tmp_detection_list.append(detection_list[j])
                        break
                
            detection_list = tmp_detection_list
            
            detection_list = captioner.process_local_caption(detection_list, is_one=is_one, is_three=is_three)


            if is_three:
                A, B, C = detection_list[0], detection_list[1], detection_list[2]
                
                A_pos = A['pcd'].get_center()
                A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
                B_pos = B['pcd'].get_center()
                B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
                C_pos = C['pcd'].get_center()
                C_pos[0] = -C_pos[0]; C_pos[1] = -C_pos[1]
                

                C_rotation_matrix = C["rotation_matrix"]
                
                C_P_A = C_rotation_matrix.T @ (A_pos - C_pos)
                C_P_B = C_rotation_matrix.T @ (B_pos - C_pos)
                
                npy_to_save = np.array([C_P_A, C_P_B])
                np.save(preprocess_path, npy_to_save)
                
                # pass
            else:
                A,B = detection_list[0], detection_list[1]

                A_rotation_matrix = A["rotation_matrix"]
                
                A_pos = A['pcd'].get_center()
                A_pos[0] = -A_pos[0]; A_pos[1] = -A_pos[1]
                B_pos = B['pcd'].get_center()
                B_pos[0] = -B_pos[0]; B_pos[1] = -B_pos[1]
                
                A_P_B = A_rotation_matrix.T @ (B_pos - A_pos)
                
                # npy_to_save = np.zeros((len(detection_list),3))
                npy_to_save = np.array([A_P_B])
                np.save(preprocess_path, npy_to_save)
                
                
                
                
            

        
        
            print(f"{filepath.split('/')[-1]} finished")


        except SkipImageException as e:
            # Meet skip image condition
            logger.info(f"Skipping processing {filename}: {e}.")
            continue
    


def parse_vqa_results(vqa_results):
    func_names = []
    conversations = []
    for i, instruction in enumerate(vqa_results):
        conversations.append(instruction)
        # func_names.append(funct_name)
    return conversations


def parse_args():
    """Command-line argument parser."""
    parser = argparse.ArgumentParser(description="Generate 3D SceneGraph for an image.")
    parser.add_argument("--config", default="configs/v2.py", help="Annotation config file path.")
    parser.add_argument(
        "--input",
        default="./demo_images",
        help="Path to input, can be json of folder of images.",
    )
    parser.add_argument("--output-dir", default="./demo_out", help="Path to save the scene-graph JSON files.")
    parser.add_argument("--name", required=False, default=None, help="Specify, otherwise use timestamp as nameing.")
    parser.add_argument("--log-dir", default="./demo_out/log", help="Path to save logs and visualization results.")
    parser.add_argument("--vis", required=False, default=True, help="Wis3D visualization for reconstruted pointclouds.")
    parser.add_argument("--overwrite", required=False, action="store_true", help="Overwrite previous.")
    return parser.parse_args()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.timestamp = timestamp
    main(args)
