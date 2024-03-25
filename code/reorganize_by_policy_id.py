# -*- coding: utf-8 -*-
# @Author  : n13eho
# @Time    : 2024.03.25

"""
Rearrange the datasets by different type of behavior policy

Line 18 and line 20 (`src_dataset_dir_path`) need to be modified according to the train/evaluation datasets' path
"""

import os
from tqdm import tqdm
import json
import shutil

current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]
# src_dataset_dir_path = os.path.join(project_root_path, 'testbed_dataset')  # < modeify me 
# target_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'train')
src_dataset_dir_path = os.path.join(project_root_path, 'emulate_dataset')  # < modeify me 
target_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'evaluate')


if __name__ == "__main__":
    
    # put all json in a list
    all_file_path = []
    all_file_name = []
    for sub_dir_name in os.listdir(src_dataset_dir_path):
        sub_dir_path = os.path.join(src_dataset_dir_path, sub_dir_name)
        for call_file in os.listdir(sub_dir_path):
            all_file_name.append(call_file)
            e_file_path = os.path.join(sub_dir_path, call_file)
            all_file_path.append(e_file_path)
    
    print(src_dataset_dir_path)
    print(target_dataset_dir_path)
    print(len(all_file_path))
    for i in tqdm(range(len(all_file_name)), desc='Moving'):
        call_file_name = all_file_name[i]
        call_file_path = all_file_path[i]
        with open(call_file_path, "r") as file:
            call_data = json.load(file)
            policy_id = call_data['policy_id']
            target_sub_dir_path = os.path.join(target_dataset_dir_path, policy_id)
            target_file_path = os.path.join(target_sub_dir_path, call_file_name)
            # copy
            shutil.copyfile(call_file_path, target_file_path)

