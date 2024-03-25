# -*- coding: utf-8 -*-
# @Author  : n13eho
# @Time    : 2024.03.25

"""
Evaluate all models (baseline, new model, ...) on whole evaluation dataset
Metircs: error rate, mse, over-estimated rate
"""

import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt

humm = 1e6

current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]
emulate_dataset_dir_path = os.path.join(project_root_path, 'ALLdatasets', 'evaluate')
onnx_models_dir_path = os.path.join(project_root_path, 'onnx_model_for_evaluation')
onnx_model_name = ['iql_v14_520k']  # < modify your onnx model names


def get_over_estimated_rate(x, y):
    l = [max((xx - yy) / yy, 0) for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return np.nanmean(l)

def get_mse(x, y):
    l = [(xx - yy) ** 2 for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return np.nanmean(l)

def get_error_rate(x, y):
    # error rate = min(1, |x-y| / y)
    l = [min(1, abs(xx - yy) / yy) for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return np.nanmean(l)

def evaluate_every_f(e_f_path):
    # [behavior policy, baseline, m1, m2, m3, ...]
    er_perf = []
    mse_perf = []
    oer_perf = []
    with open(e_f_path, "r") as file:
        call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)

        behavior_policy = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32) / humm
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32) / humm

        # first go with behavior policy
        er_perf.append(get_error_rate(behavior_policy, true_capacity))
        mse_perf.append(get_mse(behavior_policy, true_capacity))
        oer_perf.append(get_over_estimated_rate(behavior_policy, true_capacity))
        
        # then go with these models
        for onnx_name in onnx_model_name:
            onnx_m_path = os.path.join(onnx_models_dir_path, onnx_name + '.onnx')
            ort_session = ort.InferenceSession(onnx_m_path)
            predictions = []
            hc = np.zeros((1, 1), dtype=np.float32)

            for t in range(observations.shape[0]):
                feed_dict = {'obs': observations[t:t+1,:].reshape(1,1,-1),
                            'hidden_states': hc,
                            'cell_states': hc
                            }
                bw_prediction, _, _ = ort_session.run(None, feed_dict)
                predictions.append(bw_prediction[0,0,0])
            predictions = np.asarray(predictions, dtype=np.float32) / humm

            er_perf.append(get_error_rate(predictions, true_capacity))
            mse_perf.append(get_mse(predictions, true_capacity))
            oer_perf.append(get_over_estimated_rate(predictions, true_capacity))
    return er_perf, mse_perf, oer_perf


if __name__ == "__main__":

    # put all evaluation json in a list
    all_e_file_path = []
    for sub_dir_name in os.listdir(emulate_dataset_dir_path):
        sub_dir_path = os.path.join(emulate_dataset_dir_path, sub_dir_name)
        for e_file in os.listdir(sub_dir_path):
            e_file_path = os.path.join(sub_dir_path, e_file)
            all_e_file_path.append(e_file_path)

    # prepare data
    # [behavior policy, baseline, m1, m2, m3, ...]
    error_rate = []
    mse = []
    over_estimated_rate = []
    for e_file in tqdm(all_e_file_path, desc='Evaluating'):
        er_perf, mse_perf, oer_perf = evaluate_every_f(e_file)
        error_rate.append(er_perf)
        mse.append(mse_perf)
        over_estimated_rate.append(oer_perf)
    # get avg
    error_rate = np.asarray(error_rate)
    mse = np.asarray(mse)
    over_estimated_rate = np.asarray(over_estimated_rate)
    error_rate = np.average(error_rate, axis=0)
    mse = np.average(mse, axis=0)
    over_estimated_rate = np.average(over_estimated_rate, axis=0)

    # plot
    models_names = ['behavior policy']
    models_names.extend(onnx_model_name)
    type_num = len(onnx_model_name) + 1
    x = np.arange(type_num)
    bar_width = 0.3
    
    fig, bar_rate = plt.subplots(figsize=(9, 5))
    bar_mse = bar_rate.twinx()
    rects1 = bar_rate.bar(x, error_rate, bar_width, color='c')
    rects2 = bar_rate.bar(x + bar_width, over_estimated_rate, bar_width, color='orange')
    rects3 = bar_mse.bar(x + bar_width * 2, mse, bar_width, color='green')

    bar_rate.bar_label(rects1, padding=1, fmt='%.2f')
    bar_rate.bar_label(rects2, padding=1, fmt='%.2f')
    bar_mse.bar_label(rects3, padding=1, fmt='%.2f')
    
    bar_rate.set_ylabel('Error Rate / Over-estimated Rate (%)') 
    bar_mse.set_ylabel('MSE (Mbps^2)')

    bar_rate.set_xlabel("preditive models")
    plt.xticks(x + bar_width, models_names)
    plt.legend([rects1, rects2, rects3], ['error rate', 'over-estimated rate', 'mse'], 
               bbox_to_anchor=(0.5, 1.03), ncol=3, loc='center', frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, '+'.join(models_names[1:]) + ".png"))
    plt.close()
