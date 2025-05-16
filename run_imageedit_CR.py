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
import pandas as pd

from mmengine import Config
from osdsynth.processor.captions import CaptionImage
from osdsynth.processor.pointcloud import PointCloudReconstruction
from osdsynth.processor.prompt import CRPromptGenerator

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
    
    cfg['mask_conf_threshold']=0.15
    segmenter = SegmentImage(cfg, logger, device)
    rotation = Rotation(cfg, logger, device)
    reconstructor = PointCloudReconstruction(cfg, logger, device)
    captioner = CaptionImage(cfg, logger, device)
    prompter = CRPromptGenerator(cfg, logger, device)
    # f_caculator, f_caculator_tansform = depth_pro.create_model_and_transforms()
    # f_caculator.eval()
    

        
    for class_index in range(4):
        if not os.path.exists(f"{args.input}/{class_index}"):
            continue
        
        if class_index not in [0, 1]:
            continue


        global_data = glob.glob(f"{args.input}/{class_index}/*.jpg") + glob.glob(f"{args.input}/{class_index}/*.png")
       

        annotate(cfg, segmenter, rotation, reconstructor, captioner, prompter, global_data, logger, device, class_index)


preprocess_base_path = 'output_file/CR/preprocess'



def annotate(cfg, segmenter, rotation, reconstructor, captioner, prompter, global_data, logger, device, class_index):


    random.shuffle(global_data)

    all_sum = 0 
    correct_sum = 0
    correct_before_spatialQA_total_sum = 0 
    score_sum = 0
    
    
    
    gt_list=[]
    not_pass_segment_list = []

    for i, filepath in tqdm(enumerate(global_data), ncols=25):
        
        all_sum = all_sum+1
        postfix = f'CR_{filepath.split("/")[-3]}'
        filename = filepath.split("/")[-1].split(".")[0]
        # if filename != '1':
        #     continue
        print(f"\n\nProcessing file: {filename}")

        progress_file_path = os.path.join(cfg.log_folder, f"{filename}.progress")
        if os.path.exists(progress_file_path) and cfg.check_exist:
            continue

        # preprocess_path = os.path.join(preprocess_base_path, 'unedit', str(class_index))
        preprocess_path = os.path.join(preprocess_base_path, 'unedit_CR', str(class_index))
        os.makedirs(preprocess_path, exist_ok=True)
        preprocess_path = os.path.join(preprocess_path, f'{filename}.npy')
        


        image_bgr = cv2.imread(filepath)
        image_bgr = cv2.resize(image_bgr, (int(640 / (image_bgr.shape[0]) * (image_bgr.shape[1])), 640))

        try:
            is_three = False if int(filename)%2 == 0 else True
            
            detection_list = None
            # two_class = filepath.split("/")[-1].split('_')[:2]
            with open (f"{filepath.replace('png','txt').replace('jpg','txt')}", "r") as f:
                s = f.read()
                two_class = re.findall(r'<(.*?)>', s)

            
            vis_som, detection_list = segmenter.process(image_bgr, two_class)
            
            detection_list = rotation.process(detection_list)

            # Lift 2D to 3D, 3D bbox informations are included in detection_list
            detection_list, angles, xz_max_min = reconstructor.process(filename, image_bgr, detection_list)

            # Get LLaVA local caption for each region, however, currently just use a <region> placeholder
            is_one = False

            
            
            tmp_detection_list = []
            for i in range(len(detection_list)):
                for j in range(len(detection_list)):
                    if two_class[i] == detection_list[j]['class_name'][:-1]:
                        tmp_detection_list.append(detection_list[j])
                        break
                
            detection_list = tmp_detection_list
            
            detection_list = captioner.process_local_caption(detection_list, is_one=is_one, is_three=is_three)
                
            if not os.path.exists(preprocess_path):
                raise SkipImageException(f"There is no .npy file for {filename} !")

            if is_three:
                read_data = np.load(preprocess_path)
                for i in range(len(detection_list)):
                    detection_list[i]['C_P_A'] = read_data[0]
                    detection_list[i]['C_P_B'] = read_data[1]
            else:
                read_data = np.load(preprocess_path)
                for i in range(len(detection_list)):
                    detection_list[i]['A_P_B'] = read_data[0]
            # Save detection list to json
            detection_list_path = os.path.join(cfg.json_folder, f"{filename}.json")
            # save_detection_list_to_json(detection_list, detection_list_path)

            
            

            # Generate QAs based on templates
            
            vqa_results, correct, score = prompter.evaluate_predicates_on_pairs(detections=detection_list, spatial_choice=class_index, is_three=is_three)
            correct_sum += correct
            correct_before_spatialQA_total_sum += 1
            score_sum += score
            
            if correct > 0:
                gt_list.append(filename)

            for sample in vqa_results:
                print(f"Q: {sample[0][0]}")
                print(f"A: {sample[0][1]}")
                print("-----------------------")
            
            print(f"{filepath.split('/')[-1]} finished")


        except SkipImageException as e:
            # Meet skip image condition


            logger.info(f"Skipping processing {filename}: {e}.")

            not_pass_segment_list.append(f'{class_index}\t{filename}')
            continue
    
    # posfix = 
    
    if correct_before_spatialQA_total_sum > 0:
        acc = 1.0 * correct_sum / correct_before_spatialQA_total_sum
        # 把acc写到txt文件中
        with open(f'output_file/acc/acc_{postfix}.txt', 'a') as f:
            f.write(f"{class_index}: {acc}\n")
    else:
        with open(f'output_file/acc/acc_{postfix}.txt', 'a') as f:
            f.write(f"{class_index}: {0}\n")
        
    # 把gt_list写到txt中，每行一个
    score_avg = 1.0 * score_sum / correct_before_spatialQA_total_sum if correct_before_spatialQA_total_sum > 0 else 0
    with open(f'output_file/gt_list/gt_list_{postfix}.txt', 'a') as f:
        f.write(f"all: {all_sum}, total(pass the segment and reconstructor):{correct_before_spatialQA_total_sum}\ncorrect:{correct_sum}, error(not pass the segment or reconstructor):{all_sum-correct_before_spatialQA_total_sum}\nscores_avg:{score_avg}\n")
    
    with open(f'output_file/gt_list/gt_list_{postfix}.txt', 'a') as f:
        for item in gt_list:
            f.write(f"{item}\n")
            
    with open(f'output_file/notpass/not_pass_segment_{postfix}.txt', 'a') as f:
        for item in not_pass_segment_list:
            f.write(f"{item}\n")

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
