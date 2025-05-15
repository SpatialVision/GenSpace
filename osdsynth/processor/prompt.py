import random
from itertools import combinations

import numpy as np
from osdsynth.processor.prompt_utils import *
from osdsynth.processor.prompt_T2Ibench import *
from osdsynth.processor.prompt_ImageEditbench import *
from osdsynth.processor.prompt_CR import *




class T2IPromptGenerator:
    def __init__(self, cfg, logger, device):
        """Initialize the class."""
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.vis = True

    def evaluate_predicates_on_pairs(self, detections, n_conv=3, spatial_choice=-1):
        # 全部是SpatialBench的prompt
        all_prompt_variants = [
            camera_front_camera_center, # 0
            camera_back_camera_center, # 1
            camera_left_camera_center, # 2
            camera_right_camera_center, # 3
            
            camera_front_object_center, # 4
            camera_back_object_center, # 5
            camera_left_object_center, # 6
            camera_right_object_center, # 7
            
            object_side_by_side_same_direction, # 8
            object_side_by_side_opposite_direction, # 9
            object_face_to_face, # 10
            object_back_to_back, # 11
            
            object_front, # 12
            object_back, # 13
            object_left, # 14
            object_right, # 15
            
            camera_two_objects_closer, # 16
            camera_two_objects_farther, # 17
            camera_two_objects_left, # 18
            camera_two_objects_right, # 19
            
            object_apart_0_5meter, # 20
            object_apart_1meter, # 21
            object_apart_1_5meter, # 22
            object_apart_2meter, # 23
            
            camera_1meter_away, # 24
            camera_2meter_away, # 25
            camera_3meter_away, # 26
            camera_4meter_away, # 27
            
            object_bigger_than1_2, # 28
            object_higher_20cm, # 29
            object_longer_50cm, # 30
            object_wider_30cm, # 31
            
            side_by_side_front, # 32
            side_by_side_left, # 33
            side_by_side_right, # 34
            side_by_side_back, # 35
        ]
        
        if spatial_choice != -1:
            all_prompt_variants = [all_prompt_variants[spatial_choice]]
        else:
            raise ValueError("spatial_choice must not be -1 for T2Ibench")
            
        
        if spatial_choice not in [0,1,2,3,4,5,6,7,24,25,26,27]:
            all_combinations = list(combinations(range(len(detections)), 2))
            random.shuffle(all_combinations)
            selected_combinations = all_combinations[:3]
            object_pairs = [(detections[i], detections[j]) for i, j in selected_combinations]
            


            results = []
            correct = 0

            for A, B in object_pairs:
                all_prompt_variants = [item for item in all_prompt_variants]
                # selected_predicates_choices = random.sample(all_prompt_variants, n_conv)
                selected_predicates_choices = random.sample(all_prompt_variants, len(all_prompt_variants))

                for prompt_func in selected_predicates_choices:
                    res = prompt_func(A, B)
                    results.append((res, A, B, prompt_func.__name__))
                    correct = correct + 1 if res[2] else correct
                    score = res[3]

            return results, correct, score
        else:
            A = detections[0]

            results = []
            correct = 0

            all_prompt_variants = [item for item in all_prompt_variants]
            selected_predicates_choices = random.sample(all_prompt_variants, len(all_prompt_variants))

            for prompt_func in selected_predicates_choices:
                res = prompt_func(A)
                results.append((res, A, prompt_func.__name__))
                correct = correct + 1 if res[2] else correct
                score = res[3]

            return results, correct, score
        
        
class ImageEditPromptGenerator:
    def __init__(self, cfg, logger, device):
        """Initialize the class."""
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.vis = True

    def evaluate_predicates_on_pairs(self, detections, n_conv=3, spatial_choice=-1):
        # 全部是SpatialBench的prompt
        all_prompt_variants = [
            camera_to_front_camera_center, # 0
            camera_to_left_camera_center, # 1
            camera_to_right_camera_center, # 2
            camera_to_back_camera_center, # 3
            
            camera_to_front_object_center, # 4
            camera_to_left_object_center, # 5
            camera_to_right_object_center, # 6
            camera_to_back_object_center, # 7
            
            object_insert_side_by_side_same_orientation, # 8
            object_insert_side_by_side_opposite_orientation, # 9
            object_insert_face_to_face, # 10
            object_insert_back_to_back, # 11
            
            object_insert_front_object_center, # 12
            object_insert_left_object_center, # 13
            object_insert_right_object_center, # 14
            object_insert_behind_object_center, # 15
            
            object_insert_front_camera_center, # 16
            object_insert_left_camera_center, # 17
            object_insert_right_camera_center, # 18
            object_insert_behind_camera_center, # 19
            
            objectmove_close_1meter, # 20
            objectmove_far_1meter, # 21
            objectmove_left_1meter, # 22
            objectmove_right_1meter, # 23
            
            camera_forward_1meter, # 24
            camera_leftward_1meter, # 25
            camera_rightward_1meter, # 26
            camera_backward_1meter, # 27
            
            object_make_12bigger, # 28
            object_make_20cm_higher, # 29
            object_make_50cm_longer, # 30
            object_make_40cm_wider, # 31
        ]
        
        if spatial_choice != -1:
            all_prompt_variants = [all_prompt_variants[spatial_choice]]
        else:
            raise ValueError("spatial_choice must not be -1 for T2Ibench")
            
        
        if spatial_choice not in [0,1,2,3,4,5,6,7,20,21,22,23,24,25,26,27,28,29,30,31]:
            object_pairs = [(detections[0], detections[1])]
            


            results = []
            correct = 0

            for A, B in object_pairs:
                all_prompt_variants = [item for item in all_prompt_variants]
                # selected_predicates_choices = random.sample(all_prompt_variants, n_conv)
                selected_predicates_choices = random.sample(all_prompt_variants, len(all_prompt_variants))

                for prompt_func in selected_predicates_choices:
                    res = prompt_func(A, B)
                    results.append((res, A, B, prompt_func.__name__))
                    correct = correct + 1 if res[2] else correct
                    score = res[3]

            return results, correct, score
        else:
            A = detections[0]

            results = []
            correct = 0

            all_prompt_variants = [item for item in all_prompt_variants]
            selected_predicates_choices = random.sample(all_prompt_variants, len(all_prompt_variants))

            for prompt_func in selected_predicates_choices:
                res = prompt_func(A)
                results.append((res, A, prompt_func.__name__))
                correct = correct + 1 if res[2] else correct
                score = res[3]

            return results, correct, score
        
class CRPromptGenerator:
    def __init__(self, cfg, logger, device):
        """Initialize the class."""
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.vis = True

    def evaluate_predicates_on_pairs(self, detections, is_three=False, spatial_choice=-1):
        # 全部是SpatialBench的prompt
        all_prompt_variants_two = [
            CR_two_front,
            CR_two_back,
            CR_two_left,
            CR_two_right,
        ]
        
        all_prompt_variants_three = [
            CR_three_front,
            CR_three_back,
            CR_three_left,
            CR_three_right,
        ]

            
        

        A = detections[0]
        B = detections[1]
        if is_three:
            C = detections[2]
            all_prompt_variants = all_prompt_variants_three
        else:
            C = None
            all_prompt_variants = all_prompt_variants_two

        

        if spatial_choice != -1:
            all_prompt_variants = [all_prompt_variants[spatial_choice]]
        else:
            raise ValueError("spatial_choice must not be -1 for T2Ibench")


        results = []
        correct = 0

        
        all_prompt_variants = [item for item in all_prompt_variants]
        # selected_predicates_choices = random.sample(all_prompt_variants, n_conv)
        selected_predicates_choices = random.sample(all_prompt_variants, len(all_prompt_variants))

        prompt_func = selected_predicates_choices[0]
        
        if is_three:
            res = prompt_func(A, B, C)
            results.append((res, A, B, C, prompt_func.__name__))

        else:
            res = prompt_func(A, B)
            results.append((res, A, B, prompt_func.__name__))

        correct = correct + 1 if res[2] else correct
        score = res[3]

        return results, correct, score