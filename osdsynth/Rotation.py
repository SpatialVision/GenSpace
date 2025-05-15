from osdsynth.Orient_Anything.paths import *
from osdsynth.Orient_Anything.vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from PIL import Image

import torch.nn.functional as F
from osdsynth.Orient_Anything.utils import *
from osdsynth.Orient_Anything.inference import *
from osdsynth.utils.logger import SkipImageException

ckpt_path = '/data2/projects/SpatialRGPT/dataset_pipeline/osdsynth/Orient_Anything/dino_weight.pt'

class Rotation:
    def __init__(self, cfg, logger, device):
        self.rotation_model = DINOv2_MLP(
                    dino_mode   = 'large',
                    in_dim      = 1024,
                    out_dim     = 360+180+180+2,
                    evaluate    = True,
                    mask_dino   = False,
                    frozen_back = False
                ).eval()
        self.rotation_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.rotation_model = self.rotation_model.to(device)
        self.val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')
        self.logger = logger
        self.device = device
        
    def get_R(self, phi, theta, gamma):
        R = np.array([[-math.sin(phi), 0, -math.cos(phi)],
                        [math.cos(phi) * math.sin(theta), -math.cos(theta), -math.sin(phi) * math.sin(theta)],
                        [-math.cos(phi) * math.cos(theta), -math.sin(theta), math.sin(phi) * math.cos(theta)]
                    ]) # 相机看来，物体的位姿
        
        # R绕世界坐标系的Z轴旋转gamma
        gamma = gamma * math.pi / 180
        R = np.dot(R, np.array([[math.cos(gamma), -math.sin(gamma), 0],
                                [math.sin(gamma), math.cos(gamma), 0],
                                [0, 0, 1]
                                ]))
        
        
        
        return R

    def process_single(self, image_segment_wbg, do_rm_bkg=True, do_infer_aug=False):
        origin_img_wbg = Image.fromarray(image_segment_wbg) if isinstance(image_segment_wbg, np.ndarray) else image_segment_wbg
        if do_infer_aug:
            rm_bkg_img = background_preprocess(origin_img_wbg, True)
            angles = get_3angle_infer_aug(origin_img_wbg, rm_bkg_img, self.rotation_model, self.val_preprocess, self.device)
        else:
            rm_bkg_img = background_preprocess(origin_img_wbg, False)
            angles = get_3angle(rm_bkg_img, self.rotation_model, self.val_preprocess, self.device)
            confidence = float(angles[3])
            
            if confidence < 0.5:
                do_rm_bkg_img = background_preprocess(origin_img_wbg, True)
                do_angles = get_3angle(do_rm_bkg_img, self.rotation_model, self.val_preprocess, self.device)
                do_confidence = float(do_angles[3])
                if do_confidence >0.5 :
                    rm_bkg_img = do_rm_bkg_img
                    angles = do_angles
        
        
        phi   = np.radians(angles[0])
        theta = np.radians(angles[1])
        gamma = angles[2]
        confidence = float(angles[3])
        


        render_axis = render_3D_axis(phi, theta, gamma)
        res_img = overlay_images_with_scaling(render_axis, rm_bkg_img)
        rotation_matrix = self.get_R(phi=phi, theta=theta, gamma=gamma)
        return rotation_matrix, confidence


    def process(self, detection_list, do_rm_bkg=True, do_infer_aug=False):
        skip_index = []
        for i in range(len(detection_list)):
            # image_segment_wobg = detection_list[i]['image_segment']
            image_segment_wbg = detection_list[i]['image_crop']
            
            # 物体为中心的参考系，相机的位姿
            rotation_matrix, confidence = self.process_single(image_segment_wbg, do_rm_bkg=do_rm_bkg, do_infer_aug=do_infer_aug)
            
            # 对相机而言，物体的位姿
            detection_list[i]['rotation_matrix'] = rotation_matrix 
            detection_list[i]['rotation_matrix_confidence'] = confidence
            
        # 删除不符合条件的项
        for i in skip_index[::-1]:
            del detection_list[i]
        
            
        return detection_list