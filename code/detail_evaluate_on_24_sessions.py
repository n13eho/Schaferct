# -*- coding: utf-8 -*-
# @Author  : n13eho
# @Time    : 2024.03.25

"""
Evaluate models on evaluation datasets in detail.
"""

import glob
import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt


current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]

plt.rcParams.clear()
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


if __name__ == "__main__":

    data_dir = "./data"  # < modify the path to your data
    onnx_models = ['baseline', 'iql_v14_520k']  # < modify your onnx model names
    onnx_models_dir = os.path.join(project_root_path, 'onnx_model_for_evaluation')
    figs_dir = os.path.join(project_root_path, 'onnx_model_for_evaluation', ('_'.join(onnx_models[1:])))
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)
    ort_sessions = []
    for m in onnx_models:
        m_path = os.path.join(onnx_models_dir, m + '.onnx')
        ort_sessions.append(ort.InferenceSession(m_path))

    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        baseline_model_predictions = {}
        for m in onnx_models:
            baseline_model_predictions[m] = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)    
        for t in range(observations.shape[0]):
            obss = observations[t:t+1,:].reshape(1,1,-1)
            feed_dict = {'obs': obss,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            for idx, orts in enumerate(ort_sessions):
                bw_prediction, hidden_state, cell_state = orts.run(None, feed_dict)
                baseline_model_predictions[onnx_models[idx]].append(bw_prediction[0,0,0])
           
        
        for m in onnx_models:
            baseline_model_predictions[m] = np.asarray(baseline_model_predictions[m], dtype=np.float32)
            
        fig = plt.figure(figsize=(6, 3))
        time_s = np.arange(0, observations.shape[0]*60,60)/1000
        for idx, m in enumerate(onnx_models):
            plt.plot(time_s, baseline_model_predictions[m] / 1000, linestyle='-', label=['Baseline', 'Our model'][idx], color='C' + str(idx))
        plt.plot(time_s, bandwidth_predictions/1000, linestyle='--', label='Estimator ' + call_data['policy_id'], color='C' + str(len(onnx_models)))
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='black')
        plt.xlim(0, 125)
        plt.ylim(0)
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Duration (second)")
        plt.grid(True)
        
        plt.legend(bbox_to_anchor=(0.5, 1.05), ncol=4, handletextpad=0.1, columnspacing=0.5,
                    loc='center', frameon=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, os.path.basename(filename).replace(".json",".pdf")), dpi=300)
        plt.close()