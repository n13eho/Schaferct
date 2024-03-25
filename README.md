# Schaferct
This is the offical repo of team **Schaferct** in [2nd Bandwidth Prediction of MMSys'24](https://www.microsoft.com/en-us/research/academic-program/bandwidth-estimation-challenge/overview/). 


## Datasets

Use the following scripts to download training dataset and evaluation dataset

training dataset: [https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-testbed-dataset.sh](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-testbed-dataset.sh)
evaluation dataset: [https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-emulated-dataset.sh](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/download-emulated-dataset.sh)

## Hardware info

Ubuntu 20.04.6 LTS with a 12GB GPU (we use a NVIDIA GeForce RTX 3080 Ti)

## Experimentation

1. Clone repo and install packets
    
    ```bash
    git clone https://github.com/n13eho/Schaferct.git
    cd Schaferct
    pip install -r requirements.txt
    ```
    
2. Download datasets (links mentioned before)
3. Remake training datasets
    
    You can use the pickle we have made during our training at `./traning_dataset_pickle/v8.pickle` (you need to download it first, see [here](https://github.com/n13eho/Schaferct/blob/main/training_dataset_pickle/README.md) for detail), or remake a new training dataset by:
    
    1. Rearrange the datasets by different type of behavior policy. Before run this script (`reorganize_by_policy_id.py`), you need to modify the two path of your downloaded datasets.
        
        ```bash
        mkdir ALLdatasets
        mkdir ALLdatasets/train
        mkdir ALLdatasets/evaluate
        cd ./code
        python reorganize_by_policy_id.py
        ```
        
    2. Make new pickle dataset. You can modify the `K` in `make_training_dataset_pickle.py` to put more or less sessions into training dataset
        
        ```bash
        python make_training_dataset_pickle.py
        ```
        
        The dataset-making process takes about 1.5 hours.
        
4. Train model
    
    *The code we use is modified from [CORL](https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/iql.py)
    
    You can modify the variables below:
    
    1. `pickle_path`: path to training dataset, you can use your own dataset
    2. `ENUM`: how many sessions in each policy type of evaluation dataset to evaluate model every `eval_freq` steps
    3. `USE_WANDB`: trun on the [wandb](https://wandb.ai/site) or not. You can turn it on to monitor the training process by mse, error_rate, q_score, loss, etc.
    
    Then you can run:
    
    ```bash
    python v14_iql.py
    ```
    
    The training process takes about 4 hours.
    
5. Evaluate models
    
    You can use the offical [evaluation scripts](https://github.com/microsoft/RL4BandwidthEstimationChallenge/blob/main/run_baseline_model.py) to evaluate your model, and here we offer two other srcipts to help evaluation, please modify path and names of variables before runing.
    
    The detail instructions are in the comments of the scripts.
    
    1. To run a small evaluation on a [small dataset](https://github.com/microsoft/RL4BandwidthEstimationChallenge/tree/main/data): (download the 24 sessions and modify their path first)
        
        ```bash
        python detail_evaluate_on_24_sessions.py
        ```
        
    2. To evaluate the metrics (mse, errorate) over all evaluation dataset:
        
        ```bash
        python evaluate_all.py
        ```
        
        The whole evaluate process takes about 2 hours.
        
    
    !! Once again, remember to modify/adjust/rename the path/name of variables mentioned above or in the code’s comments.