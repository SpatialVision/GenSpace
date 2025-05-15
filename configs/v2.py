rotation_model_ckpt_path = '/data/genspace/osdsynth/Orient_Anything/dino_weight.pt'

# Class related params
class_set = "ram"
add_bg_classes = False
accumu_classes = False
exp_suffix = None
rm_bg_classes = True

add_classes = []
remove_classes = [
    "room",
    "kitchen",
    "office",
    "house",
    "home",
    "building",
    "corner",
    "shadow",
    "carpet",
    "photo",
    "sea",
    "shade",
    "stall",
    "space",
    "aquarium",
    "apartment",
    "image",
    "city",
    "blue",
    "skylight",
    "hallway",
    "bureau",
    "modern",
    "salon",
    "doorway",
    "wall lamp",
    "scene",
    "sun",
    "sky",
    "smile",
    "cloudy",
    "comfort",
    "white",
    "black",
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "pink",
    "stand",
    "wear",
    "area",
    "shine",
    "lay",
    "walk",
    "lead",
    "bite",
    "sing",
]
bg_classes = ["wall", "floor", "ceiling"]
check_class = [
    "man",
    "woman",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "chair",
    "couch",
    "bed",
    "toilet",
    "tv",
    "laptop",
    "microwave",
    "oven",
    "refrigerator",
    "clock"]


# Sam related params
sam_variant = "sam-hq"

# Tag2text related params
specified_tags = "None"

# Grounding Dino related params
box_threshold = 0.25
text_threshold = 0.2
nms_threshold = 0.5

# LLaVa related params
masking_option = "none"

# Selection criteria on the 2D masks
mask_area_threshold = 25  # mask with pixel area less than this will be skipped
mask_conf_threshold = 0.2  # mask with lower confidence score will be skipped default 0.2
max_bbox_area_ratio = 1.0  # boxes with larger areas than this will be skipped
skip_bg = False
min_points_threshold = 16  # projected and sampled pcd with less points will be skipped
min_points_threshold_after_denoise = 10

# point cloud processing
downsample_voxel_size = 0.025
dbscan_remove_noise = True
dbscan_eps = 0.2  # v1 use 0.2
dbscan_min_points = 10

# bounding-box related
spatial_sim_type = "overlap"  # "iou", "giou", "overlap"

save_interval = 1
wid3d_interval = 1

use_clip = False
