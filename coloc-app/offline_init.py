# Basic libraries
import shutil 
import os
import docker
import logging
import random
import time
import concurrent.futures
from collections import defaultdict
from datetime import datetime
import csv
import yaml
import re
import pprint
import math
# Additional stuff for data handling and analysis
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Specific libraries for machine learning
# Feature extraction and preprocessing
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
# Clustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering
# Dimensionality reduction and embedding
from cca_zoo.nonparametric import KCCA
from cca_zoo.linear import CCA, MCCA
from sklearn import decomposition
from cca_zoo.preprocessing import MultiViewPreprocessing
from cca_zoo.model_selection import cross_validate
from sklearn.cross_decomposition import CCA
from cca_zoo.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Regression based learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Using this notebook's custom functions
from fastapi import FastAPI as api
from pydantic import BaseModel as model

# Logging configuration
# LEVELS = {
#     'debug': logging.DEBUG,
#     'info': logging.INFO,
#     'warning': logging.WARNING,
#     'error': logging.ERROR,
#     'critical': logging.CRITICAL,
# }
logging.basicConfig(level=logging.INFO)

# Constants
RESULTS_DIR = "/usr/local/bin/results"
SCOPED_RESULTS_DIR = "./scoped_results"
CONFIG_FILE = "/usr/local/bin/scoped_results/config.yml"
FIN_CONTAINERS = "./scoped_results/died_nextflow_containers.csv"
START_CONTAINERS = "/usr/local/bin/scoped_results/started_nextflow_containers.csv"
META_DATA = "slurm-job-exporter"
DATA_SOURCE = "all"
POWER_METERING = "ebpf-mon"
POWER_STATS= "./scoped_results/task_energy_data/ebpf-mon/container_power/containers"

# Functions
# 9
def build_container_temporal_signatures_scoped_sources(results_dir, fin_containers_file):
    """
    Build feature vectors for the scoped data sources and metrics by scanning every containers directory
    under every metric for every data source. Returns a dictionary of container temporal signatures.
    As the power consumption data of the workflow tasks will be used as labels to train models, it will be excluded from the temporal signatures.
    Each container will have a 'temporal_signatures' dict with keys like 'source/metric' for every metric from the scoped data source(s).
    """

    df = pd.read_csv(fin_containers_file)
    pattern_temporal_signatures = {}
        
    for idx, row in df.iterrows():
        if  row['Nextflow'] != '':
            pattern_temporal_signatures[row['Nextflow']] = {
                'temporal_signatures': {}
            }
        else:
            continue
    
    # Feature vectors
    for root, dirs, files in os.walk(results_dir):
        if "task_energy_data" in root.split(os.sep):
            continue
        if "task_" in os.path.basename(root):
            workload_name = os.path.basename(root)
            print("Processing workload data:", workload_name)
        if os.path.basename(root) == "containers":
            metric_name = os.path.basename(os.path.dirname(root))
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    ts_container_df = pd.read_csv(file_path)
                    if 'Nextflow' in ts_container_df.columns:
                        task_name = ts_container_df['Nextflow'].iloc[0]
                        if task_name is not None and pd.isna(task_name):
                            # print(f"Nextflow task name missing in {file_path}, skipping.")
                            continue
                    else:
                        # print(f"'Nextflow' column missing in {file_path}, skipping.")
                        continue
                    # print(f"Processing task and file :", {task_name}, {file})
                    ts_container_df['timestamp'] = pd.to_datetime(ts_container_df['timestamp'], unit='ns')
                    ts_container_df.set_index('timestamp', inplace=True)
                    value_cols = [col for col in ts_container_df.columns if col.startswith('Value')]
                    if not value_cols:
                        continue
                    resource_series = ts_container_df[value_cols[0]]  

                    # Feature extraction

                    # This only takes 10 evenly spaced samples from the time series as a simple pattern representation.
                    pattern_vector = resource_series.iloc[np.round(np.linspace(0, len(resource_series) - 1, 10)).astype(int)].to_numpy()

                    # replace each point by the Gaussian-weighted mean of the surrounding 6 samples (≈3 s window, std=2 points), drops the initial NaNs, and outputs the resulting smoothed values
                    # window=6 sets the smoothing scale to 3 s at 500 ms sampling, while std=2 makes the Gaussian weight concentrate on the central few points but still include the full window, yielding features that emphasize short-term local patterns without being dominated by noise
                    # pattern_vector = resource_series.rolling(window=6, win_type='gaussian').mean(std=2).dropna().to_numpy()

                    # # PPA 
                    # n_segments = 10  # Define the number of segments
                    # segment_size = len(resource_series) // n_segments

                    # Truncate the series to make it divisible by the number of segments
                    # truncated_series = resource_series[:segment_size * n_segments]

                    # Reshape the series into segments and calculate the mean of each segment
                    # segment_vector = truncated_series.values.reshape(n_segments, segment_size).mean(axis=1)
                    

                    # Truncate the series or pad them to fixed length if intra-task lenght variability is too high
                    # pattern_vector = np.pad(pattern_vector, (0, max(0, max_length - len(pattern_vector))), mode='constant')

                    # server_spec = {
                    #     'GHz x Cores': "",
                    #     'GFlops': "",
                    #     'RAM': "",
                    #     'IOPS': "",
                    #     'Max Network Throughput': "",
                    # }
                    
                    feature_vector = { 
                        'pattern' : pattern_vector
                    }

                    # container_name = os.path.splitext(file)[0]
                    if task_name in pattern_temporal_signatures:
                        # Validation step to account for missing feature values
                        if feature_vector is not None and feature_vector != {}:
                            expected_keys = ['pattern']
                            missing_values = [key for key in expected_keys if key not in feature_vector or feature_vector[key] is None]
                            if missing_values:
                                print(f"Warning: Missing values in feature vector for {metric_name}: {missing_values}")
                            if 'pattern_vector' in feature_vector:
                                if not isinstance(feature_vector['pattern_vector'],np.ndarray):
                                    print(f"WARNING: {metric_name} pattern_vector shape: {feature_vector['pattern_vector'].shape}")
                            if workload_name not in pattern_temporal_signatures[task_name]['temporal_signatures']:
                                pattern_temporal_signatures[task_name]['temporal_signatures'][workload_name] = {} 
                            pattern_temporal_signatures[task_name]['temporal_signatures'][workload_name][metric_name] = feature_vector
    return pattern_temporal_signatures

# 13
def buildFeatureMatriceOutput(fin_df, filtered_tasks_temporal_signatures):
    """
    Build the feature matrices for the finished containers.
    Returns the feature matrix and the container names only for containers with available power values.
    """
    task_runtime_power = {}

    fin_df['LifeTime_s'] = (
        fin_df['LifeTime']
        .str.extract(r'([0-9.]+)(ms|s)', expand=True)
        .assign(
            value=lambda x: x[0].astype(float),
            seconds=lambda x: np.where(x[1] == 'ms', x['value'] / 1000, x['value'])
        )['seconds']
    )

    for idx, row in fin_df.iterrows():
        task_runtime_power[row['Nextflow']] = {
            'runtime': row['LifeTime_s'],
            'power': row['MeanPower']
        }
        
    feature_matrix_y = []
    task_names_y = []

    for task, info in task_runtime_power.items():
        # if container not in cleaned_container_temporal_signatures:
        
        if task not in filtered_tasks_temporal_signatures:
            continue
        if pd.notna(info['runtime']) and pd.notna(info['power']):
            feature_matrix_y.append([info['runtime'], info['power']])
            task_names_y.append(task)
            
    # Transform feature matrix K_y into numpy array
    feature_matrix_y = np.array(feature_matrix_y)
    print("Shape after building output feature matrix:", feature_matrix_y.shape)
    print(f"Feature matrix shape: {feature_matrix_y.shape}")
    df = pd.DataFrame(feature_matrix_y, columns=['runtime', 'power'])

    return feature_matrix_y, task_names_y

# 10
def cleanFeatureVectors(task_temporal_signatures):
    """
    Clean the feature vectors by removing containers that have no temporal signatures.
    This function modifies the input dictionary in place.
    Works with nested structure: {'container': {'temporal_signatures': {'workload': {'metric': {...}}}}}
    """
    cleaned_task_temporal_signatures = task_temporal_signatures.copy()
    none_counter = 0
    to_delete = []
    for name, info in cleaned_task_temporal_signatures.items():
        if not info['temporal_signatures']:
            none_counter += 1
            to_delete.append(name)
    print(f"Total containers with no signature for any metric: {none_counter}")

    for name in to_delete:
        del cleaned_task_temporal_signatures[name]

    print(f"Remaining containers after cleaning: {len(cleaned_task_temporal_signatures)}")

    # Collect all (workload, metric) pairs present in the data
    all_workloads = set()
    all_metrics = set()
    all_pairs = set()
    for info in cleaned_task_temporal_signatures.values():
        for workload, metrics in info['temporal_signatures'].items():
            all_workloads.add(workload)
            for metric in metrics.keys():
                all_metrics.add(metric)
                all_pairs.add((workload, metric))
    all_workloads = sorted(all_workloads)
    all_metrics = sorted(all_metrics)
    all_pairs = sorted(all_pairs)
    print(f"All workloads found: {all_workloads}")
    print(f"All metrics found: {all_metrics}")

    all_feature_names = set()
    for info in cleaned_task_temporal_signatures.values():
        for workload_metrics in info['temporal_signatures'].values():
            for metric in workload_metrics.values():
                all_feature_names.update(metric.keys())
    all_feature_names = sorted(all_feature_names)

    containers_with_all_pairs = []
    for container, info in cleaned_task_temporal_signatures.items():
        container_pairs = set()
        for workload, metrics in info['temporal_signatures'].items():
            for metric in metrics.keys():
                container_pairs.add((workload, metric))
        if container_pairs == set(all_pairs):
            containers_with_all_pairs.append(container)
    print(f"Keeping {len(containers_with_all_pairs)} containers with all workload/metric pairs.")

    # Filtered dict: only containers in containers_with_all_pairs
    filtered_containers_temporal_signatures = {
        k: v for k, v in cleaned_task_temporal_signatures.items()
        if k in containers_with_all_pairs
    }

    return (
        containers_with_all_pairs,
        all_pairs,
        all_feature_names,
        filtered_containers_temporal_signatures,
        all_metrics
    )

    
# 11
def buildFeatureMatriceInput(tasks_with_all_pairs, filtered_tasks_temporal_signatures):
    """
    Build the feature matrices for the containers with all metrics and all workloads.
    Returns the feature matrix and the container names.
    """
    # Collect all (workload, metric, feature) triplets present in the data
    all_triplets = set()
    for info in filtered_tasks_temporal_signatures.values():
        for workload, metrics in info['temporal_signatures'].items():
            for metric, feats in metrics.items():
                for feat in feats.keys():
                    all_triplets.add((workload, metric, feat))
    all_triplets = sorted(all_triplets)

    # Build full feature names
    full_feature_names = [f"{w}_{m}_{f}" for (w, m, f) in all_triplets]
    
    # print(f"Total features: {full_feature_names}")
    
    feature_matrix_x = []
    task_names_x = []
    for task in tasks_with_all_pairs:
        info = filtered_tasks_temporal_signatures[task]
        row = []
        for workload, metric, feat in all_triplets:
            value = (
                info['temporal_signatures']
                .get(workload, {})
                .get(metric, {})
                .get(feat, None)
            )
            if isinstance(value, np.ndarray):
                row.extend(value.tolist())
            else:
                row.append(value)
        feature_matrix_x.append(row)
        task_names_x.append(task)

        
    feature_matrix_x = np.array(feature_matrix_x)
    print(f"Feature matrix shape: {feature_matrix_x.shape}")
    df = pd.DataFrame(feature_matrix_x)
    # print(df)
    return feature_matrix_x, full_feature_names, task_names_x

# 12
def addPowerToFinContainers(fin_containers, tasks_with_all_pairs, power_stats):
    """
    Add power values to the finished containers file.
    """
    
    fin_df = pd.read_csv(fin_containers)
    # power_stat_container_names = set(f[:-4] for f in os.listdir(power_stats) if f.endswith('.csv'))
    # power_stat_nextflow_names = set(
    # pd.read_csv(os.path.join(power_stats, f))['Nextflow'].iloc[0]
    # for f in os.listdir(power_stats)
    # if f.endswith('.csv') and 'Nextflow' in pd.read_csv(os.path.join(power_stats, f)).columns)

    container_to_nextflow = {}
    for f in os.listdir(power_stats):
        if f.endswith('.csv'):
            container_name = f[:-4]
            nextflow_name = pd.read_csv(os.path.join(power_stats, f))['Nextflow'].iloc[0] if 'Nextflow' in pd.read_csv(os.path.join(power_stats, f)).columns else None
            container_to_nextflow[nextflow_name] = container_name

    for task in tasks_with_all_pairs:
        power_df = pd.read_csv(os.path.join(power_stats, f"{container_to_nextflow[task]}.csv"))
        mean_power = power_df['Value (microjoules)'].mean() if 'Value (microjoules)' in power_df.columns else None
        fin_df.loc[fin_df['Name'] == container_to_nextflow[task], 'MeanPower'] = mean_power
    fin_df.to_csv(fin_containers, index=False)
    return fin_df

# Helper
def containerToNfcore(fin_containers, tasks_with_all_pairs, power_stats):
    """
    Add power values to the finished containers file.
    """
    
    fin_df = pd.read_csv(fin_containers)
    container_to_nextflow = {}
    for f in os.listdir(power_stats):
        if f.endswith('.csv'):
            container_name = f[:-4]
            nextflow_name = pd.read_csv(os.path.join(power_stats, f))['Nextflow'].iloc[0] if 'Nextflow' in pd.read_csv(os.path.join(power_stats, f)).columns else None
            container_to_nextflow[nextflow_name] = container_name

    for task in tasks_with_all_pairs:
        power_df = pd.read_csv(os.path.join(power_stats, f"{container_to_nextflow[task]}.csv"))
        mean_power = power_df['Value (microjoules)'].mean() if 'Value (microjoules)' in power_df.columns else None
        fin_df.loc[fin_df['Name'] == container_to_nextflow[task], 'MeanPower'] = mean_power
    fin_df.to_csv(fin_containers, index=False)
    return container_to_nextflow



# 14
# Only as workaorund if needed 
def make_same_dimension(feature_matrix_x_patterns, task_names_x, task_names_y):
    """
    Ensure that the feature matrix X and task names X only include tasks that are common with task names Y.
    """
    # Find the indices of common tasks
    common_tasks = set(task_names_x).intersection(set(task_names_y))
    indices_to_keep = [i for i, task in enumerate(task_names_x) if task in common_tasks]

    # Filter the feature matrix and task names
    feature_matrix_x_patterns = feature_matrix_x_patterns[indices_to_keep]
    task_names_x = [task for task in task_names_x if task in common_tasks]

    print(f"Filtered feature matrix shape: {feature_matrix_x_patterns.shape}")

    return feature_matrix_x_patterns, task_names_x

# 15
# KCCA Model as function
def fitKCCALinReg(feature_matrix_x, feature_matrix_y):
    """
    Fit a KCCA model to the provided feature matrices.
    """
    # Split the data manually
    X_train, X_test, Y_train, Y_test = train_test_split(feature_matrix_x, feature_matrix_y, test_size=0.3, random_state=42)

    # MultiView Preprocessing 
    preproc = MultiViewPreprocessing([StandardScaler(), StandardScaler()])
    X_train_scaled, Y_train_scaled = preproc.fit_transform([X_train, Y_train])

    print("X_train_scaled shape:", X_train_scaled.shape)
    print("Y_train_scaled shape:", Y_train_scaled.shape)

    # Find c by cv, try different kernel functions
    # Define an kcca instance
    kcca = KCCA(latent_dimensions=2, kernel="rbf")

    # Fit the instance
    kcca.fit((X_train_scaled, Y_train_scaled))

    # Train the regression for direct predictions
    X_train_latent, _ = kcca.fit_transform((X_train_scaled, Y_train_scaled))
    reg = LinearRegression().fit(X_train_latent, Y_train)

    return reg, kcca

# 16
# Build feature output matrix with runtime labels
def buildFeatureMatriceOutputRF(fin_df, filtered_tasks_temporal_signatures):
    """
    Build the feature matrices for the finished containers.
    Returns the feature matrix and the container names only for containers with available power values.
    """
    task_runtime_power = {}

    fin_df['LifeTime_s'] = (
        fin_df['LifeTime']
        .str.extract(r'([0-9.]+)(ms|s)', expand=True)
        .assign(
            value=lambda x: x[0].astype(float),
            seconds=lambda x: np.where(x[1] == 'ms', x['value'] / 1000, x['value'])
        )['seconds']
    )

    for idx, row in fin_df.iterrows():
        task_runtime_power[row['Nextflow']] = {
            'runtime': row['LifeTime_s'],
            # 'power': row['MeanPower']
        }
        
    feature_matrix_y = []
    task_names_y = []

    for task, info in task_runtime_power.items():
        # if container not in cleaned_container_temporal_signatures:
        
        if task not in filtered_tasks_temporal_signatures:
            continue
        if pd.notna(info['runtime']):
            feature_matrix_y.append([info['runtime']])
            task_names_y.append(task)
            
    # Transform feature matrix K_y into numpy array
    feature_matrix_y = np.array(feature_matrix_y)
    print(f"Feature matrix shape: {feature_matrix_y.shape}")
    df = pd.DataFrame(feature_matrix_y, columns=['runtime'])

    return feature_matrix_y, task_names_y

# 17
# Build feature output matrix for KCCA model.
def buildFeatureMatriceOutputRF(fin_df, filtered_tasks_temporal_signatures):
    """
    Build the feature matrices for the finished containers.
    Returns the feature matrix and the container names only for containers with available power values.
    """
    task_runtime_power = {}

    for idx, row in fin_df.iterrows():
        task_runtime_power[row['Nextflow']] = {
            # 'runtime': row['LifeTime_s'],
            'power': row['MeanPower']
        }
        
    feature_matrix_y = []
    task_names_y = []

    for task, info in task_runtime_power.items():
        # if container not in cleaned_container_temporal_signatures:
        
        if task not in filtered_tasks_temporal_signatures:
            continue
        if pd.notna(info['power']):
            feature_matrix_y.append([info['power']])
            task_names_y.append(task)
            
    # Transform feature matrix K_y into numpy array
    feature_matrix_y = np.array(feature_matrix_y)
    print(f"Feature matrix shape: {feature_matrix_y.shape}")
    df = pd.DataFrame(feature_matrix_y, columns=['power'])

    return feature_matrix_y, task_names_y

# 18
def make_same_dimension(feature_matrix_x_patterns, task_names_x, task_names_y):
    """
    Ensure that the feature matrix X and task names X only include tasks that are common with task names Y.
    """
    # Find the indices of common tasks
    common_tasks = set(task_names_x).intersection(set(task_names_y))
    indices_to_keep = [i for i, task in enumerate(task_names_x) if task in common_tasks]

    # Filter the feature matrix and task names
    feature_matrix_x_patterns = feature_matrix_x_patterns[indices_to_keep]
    task_names_x = [task for task in task_names_x if task in common_tasks]

    print(f"Filtered feature matrix shape: {feature_matrix_x_patterns.shape}")

    return feature_matrix_x_patterns, task_names_x

# 19
# Power unscaled
def splitFeatureMatrices(feature_matrix_x, feature_matrix_y_power, task_names_x, task_names_y):
    """
    Split the feature matrices into training and testing sets.
    """
    X_train, X_test, y_train_runtime, y_test_runtime, train_task_names_x, test_task_names_x, train_task_names_y, test_task_names_y = train_test_split(
        feature_matrix_x, feature_matrix_y_power, task_names_x, task_names_y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, y_train_runtime, train_task_names_x, train_task_names_y

# 20
# Runtime unscaled
def splitFeatureMatrices(feature_matrix_x, feature_matrix_y_runtime, task_names_x, task_names_y):
    """
    Split the feature matrices into training and testing sets.
    """
    X_train, X_test, y_train_runtime, y_test_runtime, train_task_names_x, test_task_names_x, train_task_names_y, test_task_names_y = train_test_split(
        feature_matrix_x, feature_matrix_y_runtime, task_names_x, task_names_y, test_size=0.2, random_state=42
    )
    return X_train, y_train_runtime, train_task_names_x, train_task_names_y

# 21
# Random Forest Regressor to predict the power of tasks, if co-located
def trainPowerWithRandomForest(X, y):
    """
    Train a Random Forest regressor to predict power consumption based on the feature matrix.
    """
    regr = RandomForestRegressor(n_estimators=400,max_depth=2, max_features='log2',random_state=42)

    regr.fit(X, y.ravel()) # Flatten y-vector to 1D

    return regr

# 22
# Random Forest Regressor to predict the runtime of colocatable tasks
def trainRuntimeWithRandomForest(X, y):

    regr = RandomForestRegressor(max_depth=2, random_state=0)

    regr.fit(X, y.ravel())
    return regr


def main():
    """
    Main function to initialize the offline pipeline.
    """
    print("Offline Pipeline executing...")

    # Function calls
    # 9
    scoped_results = 'scoped_results'
    task_pattern_temporal_signatures = build_container_temporal_signatures_scoped_sources(scoped_results, FIN_CONTAINERS)
    logging.info("Built container temporal signatures for scoped data sources.")

    # 10
    tasks_with_all_pairs, all_pairs, all_feature_names, filtered_tasks_temporal_signatures, all_metrics = cleanFeatureVectors(task_pattern_temporal_signatures)
    logging.info("Cleaned feature vectors and filtered containers.")

    # 11
    # With pattern temporal signatures
    feature_matrix_x, full_feature_names, task_names_x = buildFeatureMatriceInput(
    tasks_with_all_pairs, filtered_tasks_temporal_signatures)

    # 12
    fin_df = addPowerToFinContainers(FIN_CONTAINERS, tasks_with_all_pairs, POWER_STATS)

    container_to_nextflow = containerToNfcore(FIN_CONTAINERS, tasks_with_all_pairs, POWER_STATS)
    # logging.info("Mapped containers to nf-core jobs: ", pprint.pprint(container_to_nextflow))
    logging.info("Mapped containers to nf-core jobs.")
    logging.info(f"Keys are {list(container_to_nextflow.keys())[:5]} ...")

    # 13
    finished_containers_dfs_with_power = addPowerToFinContainers(FIN_CONTAINERS, tasks_with_all_pairs, POWER_STATS)
    filtered_fin_df = finished_containers_dfs_with_power[
    finished_containers_dfs_with_power['Nextflow'].isin(tasks_with_all_pairs)].copy()
    feature_matrix_y, task_names_y = buildFeatureMatriceOutput(filtered_fin_df, filtered_tasks_temporal_signatures)

    # 14 
    feature_matrix_x, task_names_x = make_same_dimension(feature_matrix_x, task_names_x, task_names_y)

    # 15
    # For testing
    # feature_matrix_x = [[0.1]*10]*50  # Placeholder feature matrix X
    # feature_matrix_y = [[0.1]*10]*50  # Placeholder feature matrix Y
    reg_model, kcca = fitKCCALinReg(feature_matrix_x, feature_matrix_y)
    logging.info("Fitted KCCA model and linear Regression to the feature matrices.")

    # 16
    finished_containers_dfs_with_power = addPowerToFinContainers(FIN_CONTAINERS, tasks_with_all_pairs, POWER_STATS)
    filtered_fin_df = finished_containers_dfs_with_power[
        finished_containers_dfs_with_power['Nextflow'].isin(tasks_with_all_pairs)
    ].copy()
    feature_matrix_y_runtime, task_names_y  = buildFeatureMatriceOutputRF(filtered_fin_df, filtered_tasks_temporal_signatures)
    logging.info("Built feature matrix for runtime prediction using Random Forest Regressor.")
    
    # 17
    finished_containers_dfs_with_power = addPowerToFinContainers(FIN_CONTAINERS, tasks_with_all_pairs, POWER_STATS)
    filtered_fin_df = finished_containers_dfs_with_power[
        finished_containers_dfs_with_power['Nextflow'].isin(tasks_with_all_pairs)
    ].copy()
    feature_matrix_y_power, task_names_y = buildFeatureMatriceOutputRF(filtered_fin_df, filtered_tasks_temporal_signatures)
    logging.info("Built feature matrix for power prediction using Random Forest Regressor.")

    # 18
    feature_matrix_y_power, task_names_y = make_same_dimension(feature_matrix_y_power, task_names_x, task_names_y)
    feature_matrix_y_runtime, task_names_y = make_same_dimension(feature_matrix_y_runtime, task_names_x, task_names_y)
    logging.info("Ensured feature matrices have the same dimensions.")

    # 19
    X_train, y_train_power, train_task_names_x, train_task_names_y = splitFeatureMatrices(feature_matrix_x, feature_matrix_y_power, task_names_x, task_names_y)
    logging.info("Split feature matrices for power labels into training and testing sets.")

    # 20
    X_train, y_train_runtime, train_task_names_x, train_task_names_y = splitFeatureMatrices(feature_matrix_x, feature_matrix_y_runtime, task_names_x, task_names_y)
    logging.info("Split feature matrices for runtime labels into training and testing sets.")

    # 21
    trainedPowerPredictor = trainPowerWithRandomForest(X_train, y_train_power.ravel())
    logging.info("Trained Random Forest Regressor for power prediction.")
    
    # 22
    trainedRuntimePredictor = trainRuntimeWithRandomForest(X_train, y_train_runtime.ravel())
    logging.info("Trained Random Forest Regressor for runtime prediction.")

    # Make some variables accessible globally if needed
    return filtered_tasks_temporal_signatures, scoped_results, container_to_nextflow, reg_model, kcca, trainedPowerPredictor, trainedRuntimePredictor

if __name__ == "__main__":
    main()
    