import cv2
import torch
import torchvision
from osdsynth.processor.wrappers.grounding_dino import get_grounding_dino_model
from osdsynth.processor.wrappers.ram import get_tagging_model, run_tagging_model
from osdsynth.processor.wrappers.sam import (
    convert_detections_to_dict,
    convert_detections_to_list,
    crop_detections_with_xyxy,
    filter_detections,
    get_sam_predictor,
    get_sam_segmentation_from_xyxy,
    mask_subtract_contained,
    post_process_mask,
    sort_detections_by_area,
)
from osdsynth.utils.logger import SkipImageException
from osdsynth.visualizer.som import draw_som_on_image
from PIL import Image
import numpy as np

class SegmentImage:
    """Class to segment the image."""

    def __init__(self, cfg, logger, device, init_gdino=True, init_tagging=True, init_sam=True):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        if init_gdino:
            # Initialize the Grounding Dino Model
            self.grounding_dino_model = get_grounding_dino_model(cfg, device)
        else:
            self.grounding_dino_model = None

        if init_tagging:
            # Initialize the tagging Model
            self.tagging_transform, self.tagging_model = get_tagging_model(cfg, device)
        else:
            self.tagging_transform = self.tagging_model = None

        if init_sam:
            # Initialize the SAM Model
            self.sam_predictor = get_sam_predictor(cfg.sam_variant, device)
        else:
            self.sam_predictor = None

        pass

    def process(self, image_bgr, two_class ,plot_som=True):
        """Segment the image."""

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_pil = Image.fromarray(image_rgb)
        
        # image_rgb_pil.save('tmp.png')

        img_tagging = image_rgb_pil.resize((384, 384))
        img_tagging = self.tagging_transform(img_tagging).unsqueeze(0).to(self.device)

        # Tag2Text
        if two_class is None:    
            classes = run_tagging_model(self.cfg, img_tagging, self.tagging_model)
        else:
            classes = two_class

        if len(classes) == 0:
            raise SkipImageException("No foreground objects detected by tagging model.")

        # Using GroundingDINO to detect and SAM to segment
        detections = self.grounding_dino_model.predict_with_classes(
            image=image_bgr,  # This function expects a BGR image...
            classes=classes,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
        )
        image_rgb_pil.save('tmp.png')
        print('two_class:', two_class)
        if len(detections.class_id) != len(classes):
            print("DO HUMAN CHECK")

        if len(detections.class_id) < 1:
            raise SkipImageException("No object detected.")
        


        # Non-maximum suppression
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.cfg.nms_threshold,
            )
            .numpy()
            .tolist()
        )

        print(f"Before NMS: {len(detections.xyxy)} detections")
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        print(f"After NMS: {len(detections.xyxy)} detections")

        # Somehow some detections will have class_id=-1, remove them
        valid_idx = detections.class_id != -1
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]

        # Segment Anything
        detections.mask = get_sam_segmentation_from_xyxy(
            sam_predictor=self.sam_predictor, image=image_rgb, xyxy=detections.xyxy
        )

        # Convert the detection to a dict. Elements are np.ndarray
        detections_dict = convert_detections_to_dict(detections, classes)
        
        # Filter out the objects based on various criteria
        detections_dict = filter_detections(self.cfg, detections_dict, image_rgb)

        if len(detections_dict["xyxy"]) < 1:
            raise SkipImageException("No object detected after filtering.")

        # Subtract the mask of bounding boxes that are contained by it
        detections_dict["subtracted_mask"], mask_contained = mask_subtract_contained(
            detections_dict["xyxy"], detections_dict["mask"], th1=0.05, th2=0.05
        )

        # Sort the dets by area
        detections_dict = sort_detections_by_area(detections_dict)

        # Add RLE to dict
        detections_dict = post_process_mask(detections_dict)

        # Convert the detection to a list. Each element is a dict
        detections_list = convert_detections_to_list(detections_dict, classes)
        
        # Skip objects with confidence lower than 0.4
        # detections_list = skipbyconfidence(detections_list)

        detections_list = crop_detections_with_xyxy(self.cfg, image_rgb_pil, detections_list)

        detections_list = segmentImage(detections_list, image_rgb_pil)
        
        detections_list = add_index_to_class(detections_list)
        
        if two_class is not None:
            if len(two_class)==2 and len(detections_list) != 2:
                raise SkipImageException("Not all objects detected.")
            
            if len(two_class)==1 and len(detections_list) != 1:
                raise SkipImageException("Not all objects detected.")
            
            if len(two_class)==3 and len(detections_list) != 3:
                raise SkipImageException("Not all objects detected.")
            
            if len(two_class)==2:
                detections_two_class = [detections_list[0]['class_name'][:-1], detections_list[1]['class_name'][:-1]]
                if two_class[0] not in detections_two_class or two_class[1] not in detections_two_class:
                    raise SkipImageException("Not all objects detected.")
                
            if len(two_class)==3:
                detections_two_class = [detections_list[0]['class_name'][:-1], detections_list[1]['class_name'][:-1], detections_list[2]['class_name'][:-1]]
                if two_class[0] not in detections_two_class or two_class[1] not in detections_two_class or two_class[2] not in detections_two_class:
                    raise SkipImageException("Not all objects detected.")
            
        
        # image_tagged = add_bbox_and_taggingtext_to_image(image_bgr, detections_list)
        # save image
        # cv2.imwrite("tmp_tagged.jpg", image_tagged)
        

        if plot_som:
            # Visualize with SoM
            vis_som = draw_som_on_image(
                detections_dict,
                image_rgb,
                label_mode="1",
                alpha=0.4,
                anno_mode=["Mask", "Mark", "Box"],
            )
        else:
            vis_som = None
        
        
            
        return vis_som, detections_list


# 从mask_crop和image_crop中分割出不含背景的物体
def segmentImage(detections_list, image_rgb_pil):
    
    for i in range(len(detections_list)):
        image_pil = detections_list[i]['image_crop']
        mask_pil = Image.fromarray(detections_list[i]['mask_crop'])
        
        image_rgba = image_pil.convert("RGBA")
        
        # 创建一个全透明的背景图（与原始图尺寸相同）
        transparent_bg = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))

        # 使用掩码将物体区域从原图复制到透明背景上
        segmented_image = Image.composite(
            image_rgba,       # 原图的物体区域
            transparent_bg,   # 透明背景
            mask_pil          # 掩码（白色为物体，黑色为背景）
        )
        
        detections_list[i]['image_segment'] = segmented_image
    
    return detections_list
    
def skipbyconfidence(detections_list):
    skip_index = []
    for i in range(len(detections_list)):
        if detections_list[i]['confidence'] < 0.3:
            skip_index.append(i)
    
    # 删除不符合条件的项
    for i in skip_index[::-1]:
        del detections_list[i]
    
    return detections_list
    
def add_bbox_and_taggingtext_to_image(image, detections_list):
    for i in range(len(detections_list)):
        bbox = detections_list[i]['xyxy']
        label = detections_list[i]['class_name']
        confidence = detections_list[i]['confidence']
        
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (int(bbox[0]), int((bbox[1]+bbox[3])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def add_index_to_class(detections_list):
    #如果某个class第一次出现，就把这个对象的class_name后面加上0，第二次出现就加上1，以此类推
    class_index = {}
    for detection in detections_list:
        class_name = detection['class_name']
        if class_name not in class_index:
            class_index[class_name] = 0
        else:
            class_index[class_name] += 1
        # 新增字段，保留原始类名
        detection['class_name'] = f"{class_name}{class_index[class_name]}"
    return detections_list
        