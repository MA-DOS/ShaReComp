# imports
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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Specific libraries for machine learning
# Feature extraction and preprocessing
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Clustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering
# Dimensionality reduction and embedding
from mvlearn.embed import KMCCA
# Regression based learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

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
# 1
def readInResultsConf(config_file):
    """
    Read in the results configuration file and return a dictionary.
    """
    monitoring_config = config_file
    with open(monitoring_config, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    filtered_sources = []
    seen = set()
    for target in data['monitoring_targets'].values():
        ds = target.get('data_sources')
        if ds:
            if isinstance(ds, dict):
                ds = [ds]
            for entry in ds:
                filtered = {k: entry[k] for k in ('identifier', 'source') if k in entry}
                if (
                    'source' in filtered and
                    filtered['source'] == 'slurm-job-exporter'
                ):
                    continue
                if 'source' in filtered and 'identifier' in filtered:
                    key = (filtered['source'], filtered['identifier'])
                    if key not in seen:
                        filtered_sources.append(filtered)
                        seen.add(key)
    pprint.pprint(filtered_sources)
    return filtered_sources

# 2
def resultsScope(results_dir, meta_data, data_source, power_metering):
    """
   Creates a copy of the results directory and returns the cleaned file tree depending on the users scope definition.
   Meta data, data source and power metering are mandatory scope definitions.
    """
    # scoped_results_dir = shutil.copytree(results_dir, "/usr/local/bin/scoped_results", dirs_exist_ok=True)
    scoped_results_dir = shutil.copytree(results_dir, "./scoped_results", dirs_exist_ok=True)
    if data_source == 'all':
        print("Data source is set to 'all', no filtering will be applied.")
        return scoped_results_dir
    for metric in os.listdir(scoped_results_dir):
        metric_path = os.path.join(scoped_results_dir, metric)
        if not os.path.isdir(metric_path):
           continue 
    # Decide which subdir to keep for this metric
        if metric == "task_metadata":
            keep = [meta_data]
        elif metric == "task_energy_data":
            keep = [power_metering]
        else:
            keep = [data_source]
    # Walk from base dir and rm all dirs that do not match the scope and the power dirs. 
        for subdir in os.listdir(metric_path):
            subdir_path = os.path.join(metric_path, subdir)
            if os.path.isdir(subdir_path) and subdir not in keep:
                shutil.rmtree(subdir_path, ignore_errors=True)
    print("Successfully scoped results directory:", scoped_results_dir)
    return scoped_results_dir
    #         subdir_name = os.path.basename(subdir_path)
    #         # print("Sub directory name:", subdir_name)
    #         if os.path.isdir(subdir_path) and subdir_name not in [meta_data, data_source, power_metering]:
    #             shutil.rmtree(subdir_path, ignore_errors=True)
    # print("Successfully scoped results directory:", scoped_results_dir) 
    # return scoped_results_dir

# 3
def split_task_timeseries_by_datasource(results_dir, datasource_identifier_map, nextflow_pattern=r"nxf-[A-Za-z0-9]{23}"):
    """
    For each data source in datasource_identifier_map, traverse the results_dir,
    and for each metric, split the time series CSVs into per-task files using the correct identifier column.
    """
    for datasource, identifier in datasource_identifier_map.items():
        for root, dirs, files in os.walk(results_dir):
            if os.path.basename(root) == datasource:
                for metric in os.listdir(root):
                    metric_path = os.path.join(root, metric)
                    if os.path.isdir(metric_path):
                        containers_dir = os.path.join(metric_path, "containers")
                        os.makedirs(containers_dir, exist_ok=True)
                        for file in os.listdir(metric_path):
                            if file.endswith(".csv"):
                                file_path = os.path.join(metric_path, file)
                                df = pd.read_csv(file_path)
                                if identifier not in df.columns:
                                    print(f"Identifier '{identifier}' not found in {file_path}, skipping.")
                                    continue
                                for task_name in df[identifier].unique():
                                    if pd.isna(task_name):
                                        continue
                                    if re.match(nextflow_pattern, str(task_name)):
                                        task_df = df[df[identifier] == task_name]
                                        out_path = os.path.join(containers_dir, f"{task_name}.csv")
                                        task_df.to_csv(out_path, index=False)
                                        # print(f"Saved data for {task_name} to {out_path}")
    print("Finished splitting time series data by data source.")

# 4
def report_missing_tasks_all_sources(results_dir, datasource_identifier_map, fin_containers_df, container_workdirs, nextflow_pattern=r"nxf-[A-Za-z0-9]{23}"):
    """
    For each data source, report how many tasks are missing compared to the finished containers.
    """
    workdir_containers = set(container_workdirs.keys())
    for datasource, identifier in datasource_identifier_map.items():
        found_containers = set()
        for root, dirs, files in os.walk(results_dir):
            if os.path.basename(root) == datasource:
                for metric in os.listdir(root):
                    metric_path = os.path.join(root, metric)
                    if os.path.isdir(metric_path):
                        for file in os.listdir(metric_path):
                            if file.endswith(".csv"):
                                file_path = os.path.join(metric_path, file)
                                df = pd.read_csv(file_path)
                                if identifier not in df.columns:
                                    continue
                                found_containers.update(
                                    str(name) for name in df[identifier].unique()
                                    if pd.notna(name) and re.match(nextflow_pattern, str(name))
                                )
        missing_in_source = workdir_containers - found_containers
        missing_in_workdirs = found_containers - workdir_containers
        print(f"--- {datasource} ---")
        print("Containers in monitored list but NOT in", datasource + ":", missing_in_source)
        print("Count:", len(missing_in_source))
        print("Containers in", datasource, "but NOT in monitored list:", missing_in_workdirs)
        print("Count:", len(missing_in_workdirs))
        print()
        
# 5
def add_workdir_to_all_task_csvs(results_dir, container_workdirs):
    """
    For every data source and metric, update each per-task CSV in 'containers' subfolders
    with the correct WorkDir from container_workdirs.
    """
    for root, dirs, files in os.walk(results_dir):
        if os.path.basename(root) == "containers":
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    fin_container_df = pd.read_csv(file_path)
                    container_name = os.path.splitext(file)[0]
                    if container_name in container_workdirs:
                        workdir = container_workdirs[container_name]
                        fin_container_df['WorkDir'] = workdir
                        fin_container_df.to_csv(file_path, index=False)
                        # print(f"Updated {file_path} with work directory {workdir}")

# 6
def extract_slurm_job_metadata(slurm_metadata_path, slurm_job_col="job_name"):
    """
    Extracts slurm job metadata from time-series CSVs and writes each job's data to a separate file.
    """
    for file in os.listdir(slurm_metadata_path):
        if file.endswith("slurm_job_id.csv"):
            file_path = os.path.join(slurm_metadata_path, file)
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path)
            for job_name in df[slurm_job_col].unique():
                if pd.isna(job_name):
                    continue
                print(f"Processing job: {job_name}")
                job_df = df[df[slurm_job_col] == job_name]
                out_path = os.path.join(slurm_metadata_path, f"{job_name}.csv")
                job_df.to_csv(out_path, index=False)
                print(f"Saved data for {job_name} to {out_path}")

# 7
def update_finished_containers_with_nfcore_task(slurm_metadata_path, fin_containers, workdir_col='WorkDir', slurm_workdir_col='work_dir', slurm_job_col='job_name'):
    """
    Update the finished containers file with the nf-core task name (Nextflow) by matching work directories
    with slurm job metadata.
    """

    updated = False
    for file in os.listdir(slurm_metadata_path):
        if file.endswith("slurm_job_id.csv"):
            file_path = os.path.join(slurm_metadata_path, file)
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path)
            fin_df = pd.read_csv(fin_containers)
            if workdir_col in fin_df.columns and slurm_workdir_col in df.columns:
                for idx, row in df.iterrows():
                    work_dir = row[slurm_workdir_col]
                    slurm_job = row[slurm_job_col]
                    if pd.isna(work_dir) or pd.isna(slurm_job):
                        print(f"Skipping row {idx} due to missing WorkDir or slurm_job.")
                        continue
                    # Update fin_df where WorkDir matches
                    fin_df.loc[fin_df[workdir_col] == work_dir, 'Nextflow'] = slurm_job
                # Write back the updated fin_df
                fin_df.to_csv(fin_containers, index=False)
                print(f"Updated {fin_containers} with slurm job info.")
                updated = True
            else:
                print("WorkDir or job_name column missing in DataFrames.")
    if not updated:
        print("No updates were made to the finished containers file.")

# 8
def add_nextflow_to_all_task_csvs(results_dir, fin_containers_file, workdir_col='WorkDir', nextflow_col='Nextflow'):
    """
    For every data source and metric, update each per-task CSV in 'containers' subfolders
    with the correct Nextflow task value from the finished containers file.
    """
    fin_df = pd.read_csv(fin_containers_file)
    # Ensure WorkDir is string and stripped in fin_df
    fin_df[workdir_col] = fin_df[workdir_col].astype(str).str.strip()
    for root, dirs, files in os.walk(results_dir):
        if os.path.basename(root) == "containers":
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    container_df = pd.read_csv(file_path)
                    if workdir_col in container_df.columns:
                        # Ensure WorkDir is string and stripped in container_df
                        container_df[workdir_col] = container_df[workdir_col].astype(str).str.strip()
                        workdir = container_df[workdir_col].iloc[0]
                        match = fin_df[fin_df[workdir_col] == workdir]
                        if not match.empty and nextflow_col in match.columns:
                            nextflow_value = match[nextflow_col].values[0]
                            container_df[nextflow_col] = nextflow_value
                            container_df.to_csv(file_path, index=False)
                            print(f"Updated {file_path} with Nextflow value {nextflow_value}")
                        else:
                            print(f"No matching Nextflow value found for WorkDir {workdir} in {file_path}") 

# 9
def build_container_temporal_signatures_scoped_sources(results_dir, fin_containers_file):
    """
    Build feature vectors for the scoped data sources and metrics by scanning every containers directory
    under every metric for every data source. Returns a dictionary of container temporal signatures.
    As the power consumption data of the workflow tasks will be used as labels to train models, it will be excluded from the temporal signatures.
    Each container will have a 'temporal_signatures' dict with keys like 'source/metric' for every metric from the scoped data source(s).
    """
    df = pd.read_csv(fin_containers_file)
    container_temporal_signatures = {}
    for idx, row in df.iterrows():
        container_temporal_signatures[row['Name']] = {
            'temporal_signatures': {
            }
        }

    # Feature vectors
    for root, dirs, files in os.walk(results_dir):
        if "task_energy_data" in root.split(os.sep):
            continue
        if "task_" in os.path.basename(root):
            workload_name = os.path.basename(root)
            print("Current workload:", workload_name)
        if os.path.basename(root) == "containers":
            metric_name = os.path.basename(os.path.dirname(root))
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    ts_container_df = pd.read_csv(file_path)
                    ts_container_df['timestamp'] = pd.to_datetime(ts_container_df['timestamp'], unit='ns')
                    ts_container_df.set_index('timestamp', inplace=True)
                    value_cols = [col for col in ts_container_df.columns if col.startswith('Value')]
                    if not value_cols:
                        # print(f"Skipping {file_path} as it does not contain 'value' column.")
                        continue
                    resource_series = ts_container_df[value_cols[0]]  

                    # Feature extraction
                    # peak_value = resource_series.max()
                    # lowest_value = resource_series.min()
                    # mean_value = resource_series.mean()
                    # median_value = resource_series.median()
                    # variance = resource_series.var()
                    # mean_val = resource_series.mean()
                    # if mean_val == 0:
                    #     relative_variance = 0.0  
                    # else:
                    #     relative_variance = (resource_series.var() - mean_val**2) / (mean_val**2)
                    # std_dev = resource_series.std()
                    pattern_vector = resource_series.iloc[np.round(np.linspace(0, len(resource_series) - 1, 10)).astype(int)].to_numpy()

                    # The server spec can come from the host benchmark in nextflow
                    server_spec = {
                        'GHz x Cores': "",
                        'GFlops': "",
                        'RAM': "",
                        'IOPS': "",
                        'Max Network Throughput': "",
                    }

                    feature_vector = { 
                        'pattern' : pattern_vector
                    }

                    container_name = os.path.splitext(file)[0]
                    if container_name in container_temporal_signatures:
                        if feature_vector is not None and feature_vector != {}:
                            # Validation step to account for missing feature values
                            expected_keys = ['pattern']
                            missing_values = [key for key in expected_keys if key not in feature_vector or feature_vector[key] is None]
                            if missing_values:
                                print(f"Warning: Missing values in feature vector for {container_name} in {metric_name}: {missing_values}")
                            if 'pattern_vector' in feature_vector:
                                if not isinstance(feature_vector['pattern_vector'],np.ndarray):
                                    print(f"WARNING: {container_name} {metric_name} pattern_vector shape: {feature_vector['pattern_vector'].shape}")
                            if workload_name not in container_temporal_signatures[container_name]['temporal_signatures']:
                                container_temporal_signatures[container_name]['temporal_signatures'][workload_name] = {} 
                            container_temporal_signatures[container_name]['temporal_signatures'][workload_name][metric_name] = feature_vector
    pprint.pprint(container_temporal_signatures)
    return container_temporal_signatures


# 10
def cleanFeatureVectors(container_temporal_signatures):
    """
    Clean the feature vectors by removing containers that have no temporal signatures.
    This function modifies the input dictionary in place.
    Works with nested structure: {'container': {'temporal_signatures': {'workload': {'metric': {...}}}}}
    """
    cleaned_container_temporal_signatures = container_temporal_signatures.copy()
    none_counter = 0
    to_delete = []
    for name, info in cleaned_container_temporal_signatures.items():
        if not info['temporal_signatures']:
            none_counter += 1
            to_delete.append(name)
    print(f"Total containers with no signature for any metric: {none_counter}")

    for name in to_delete:
        del cleaned_container_temporal_signatures[name]

    print(f"Remaining containers after cleaning: {len(cleaned_container_temporal_signatures)}")

    # Collect all (workload, metric) pairs present in the data
    all_workloads = set()
    all_metrics = set()
    all_pairs = set()
    for info in cleaned_container_temporal_signatures.values():
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
    for info in cleaned_container_temporal_signatures.values():
        for workload_metrics in info['temporal_signatures'].values():
            for metric in workload_metrics.values():
                all_feature_names.update(metric.keys())
    all_feature_names = sorted(all_feature_names)

    containers_with_all_pairs = []
    for container, info in cleaned_container_temporal_signatures.items():
        container_pairs = set()
        for workload, metrics in info['temporal_signatures'].items():
            for metric in metrics.keys():
                container_pairs.add((workload, metric))
        if container_pairs == set(all_pairs):
            containers_with_all_pairs.append(container)
    print(f"Keeping {len(containers_with_all_pairs)} containers with all workload/metric pairs.")

    # Filtered dict: only containers in containers_with_all_pairs
    filtered_containers_temporal_signatures = {
        k: v for k, v in cleaned_container_temporal_signatures.items()
        if k in containers_with_all_pairs
    }

    return (
        cleaned_container_temporal_signatures,
        containers_with_all_pairs,
        all_pairs,
        all_feature_names,
        filtered_containers_temporal_signatures,
        all_metrics
    )
    
# 11
def build_container_temporal_signatures_scoped_sources(results_dir, fin_containers_file):
    """
    Build feature vectors for the scoped data sources and metrics by scanning every containers directory
    under every metric for every data source. Returns a dictionary of container temporal signatures.
    As the power consumption data of the workflow tasks will be used as labels to train models, it will be excluded from the temporal signatures.
    Each container will have a 'temporal_signatures' dict with keys like 'source/metric' for every metric from the scoped data source(s).
    """
    df = pd.read_csv(fin_containers_file)
    container_temporal_signatures = {}
    for idx, row in df.iterrows():
        container_temporal_signatures[row['Name']] = {
            'temporal_signatures': {
            }
        }

    # Feature vectors
    for root, dirs, files in os.walk(results_dir):
        if "task_energy_data" in root.split(os.sep):
            continue
        if "task_" in os.path.basename(root):
            workload_name = os.path.basename(root)
            print("Current workload:", workload_name)
        if os.path.basename(root) == "containers":
            metric_name = os.path.basename(os.path.dirname(root))
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    ts_container_df = pd.read_csv(file_path)
                    ts_container_df['timestamp'] = pd.to_datetime(ts_container_df['timestamp'], unit='ns')
                    ts_container_df.set_index('timestamp', inplace=True)
                    value_cols = [col for col in ts_container_df.columns if col.startswith('Value')]
                    if not value_cols:
                        # print(f"Skipping {file_path} as it does not contain 'value' column.")
                        continue
                    resource_series = ts_container_df[value_cols[0]]  

                    # Feature extraction
                    peak_value = resource_series.max()
                    lowest_value = resource_series.min()
                    mean_value = resource_series.mean()
                    median_value = resource_series.median()
                    variance = resource_series.var()
                    mean_val = resource_series.mean()
                    if mean_val == 0:
                        relative_variance = 0.0  
                    else:
                        relative_variance = (resource_series.var() - mean_val**2) / (mean_val**2)
                    std_dev = resource_series.std()
                    pattern_vector = resource_series.iloc[np.round(np.linspace(0, len(resource_series) - 1, 10)).astype(int)].to_numpy()

                    # The server spec can come from the host benchmark in nextflow
                    server_spec = {
                        'GHz x Cores': "",
                        'GFlops': "",
                        'RAM': "",
                        'IOPS': "",
                        'Max Network Throughput': "",
                    }

                    # feature_vector = { 
                    #     'peak_value': peak_value, 'lowest_value': lowest_value, 'mean': mean_value, 
                    #     'variance': variance
                    # }

                    # TODO: Maybe add median and the pattern vector to the feature vector.
                    # I think variance does not make sense
                    feature_vector = { 
                        'peak_value': peak_value, 'lowest_value': lowest_value, 'mean': mean_value, 
                    }

                    container_name = os.path.splitext(file)[0]
                    if container_name in container_temporal_signatures:
                        if feature_vector is not None and feature_vector != {}:
                            # Validation step to account for missing feature values
                            expected_keys = ['peak_value', 'lowest_value', 'mean']
                            missing_values = [key for key in expected_keys if key not in feature_vector or feature_vector[key] is None]
                            if missing_values:
                                print(f"Warning: Missing values in feature vector for {container_name} in {metric_name}: {missing_values}")
                            if 'pattern_vector' in feature_vector:
                                if not isinstance(feature_vector['pattern_vector'],np.ndarray):
                                    print(f"WARNING: {container_name} {metric_name} pattern_vector shape: {feature_vector['pattern_vector'].shape}")
                            if workload_name not in container_temporal_signatures[container_name]['temporal_signatures']:
                                container_temporal_signatures[container_name]['temporal_signatures'][workload_name] = {} 
                            container_temporal_signatures[container_name]['temporal_signatures'][workload_name][metric_name] = feature_vector
    pprint.pprint(container_temporal_signatures)
    return container_temporal_signatures

# 12
def buildFeatureMatriceInput(containers_with_all_metrics, cleaned_container_temporal_signatures):
    """
    Build the feature matrices for the containers with all metrics and all workloads.
    Returns the feature matrix and the container names.
    """
    # Collect all (workload, metric, feature) triplets present in the data
    all_triplets = set()
    for info in cleaned_container_temporal_signatures.values():
        for workload, metrics in info['temporal_signatures'].items():
            for metric, feats in metrics.items():
                for feat in feats.keys():
                    all_triplets.add((workload, metric, feat))
    all_triplets = sorted(all_triplets)

    # Build full feature names
    full_feature_names = [f"{w}_{m}_{f}" for (w, m, f) in all_triplets]

    feature_matrix_x = []
    container_names_x = []
    for container in containers_with_all_metrics:
        info = cleaned_container_temporal_signatures[container]
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
        container_names_x.append(container)

    feature_matrix_x = np.array(feature_matrix_x)
    print(f"Feature matrix shape: {feature_matrix_x.shape}")
    df = pd.DataFrame(feature_matrix_x, columns=full_feature_names)
    print(df)
    return feature_matrix_x, full_feature_names, container_names_x

# 13
def addPowerToFinContainers(fin_containers, containers_with_all_metrics, power_stats):
    """
    Add power values to the finished containers file.
    """
    fin_df = pd.read_csv(fin_containers)
    power_stat_files = set(f[:-4] for f in os.listdir(power_stats) if f.endswith('.csv'))
    # print(power_stat_files)

    for container in containers_with_all_metrics:
        # print(container)
        if container in power_stat_files:
            power_df = pd.read_csv(os.path.join(power_stats, f"{container}.csv"))
            # print(power_df.head())
            mean_power = power_df['Value (microjoules)'].mean() if 'Value (microjoules)' in power_df.columns else None
            fin_df.loc[fin_df['Name'] == container, 'MeanPower'] = mean_power
    fin_df.to_csv(fin_containers, index=False)
    return fin_df

# 14
# Build feature output matrix for KCCA model.
def buildFeatureMatriceOutput(fin_df):
    """
    Build the feature matrices for the finished containers.
    Returns the feature matrix and the container names only for containers with available power values.
    """
    container_runtime_power = {}

    fin_df['LifeTime_s'] = (
        fin_df['LifeTime']
        .str.extract(r'([0-9.]+)(ms|s)', expand=True)
        .assign(
            value=lambda x: x[0].astype(float),
            seconds=lambda x: np.where(x[1] == 'ms', x['value'] / 1000, x['value'])
        )['seconds']
    )

    for idx, row in fin_df.iterrows():
        container_runtime_power[row['Name']] = {
            'runtime': row['LifeTime_s'],
            'power': row['MeanPower']
        }
        
    feature_matrix_y = []
    container_names_y = []

    for container, info in container_runtime_power.items():
        if container not in cleaned_container_temporal_signatures:
            continue
        if pd.notna(info['runtime']) and pd.notna(info['power']):
            feature_matrix_y.append([info['runtime'], info['power']])
            container_names_y.append(container)
            
    # Transform feature matrix K_y into numpy array
    feature_matrix_y = np.array(feature_matrix_y)
    print(f"Feature matrix shape: {feature_matrix_y.shape}")
    df = pd.DataFrame(feature_matrix_y, columns=['runtime', 'power'])
    print(df)

    return feature_matrix_y, container_names_y

# 15
def scaleFeatureMatrices(feature_matrix_x, feature_matrix_y):
    """
    Scale the feature matrices using StandardScaler.
    Returns the scaled feature matrices.
    """
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaled_x = scaler_x.fit_transform(feature_matrix_x)
    scaled_y = scaler_y.fit_transform(feature_matrix_y)

    print(f"Scaled feature matrix X shape: {scaled_x.shape}")
    print(f"Scaled feature matrix Y shape: {scaled_y.shape}")
    
    return scaled_x, scaled_y, scaler_x, scaler_y

# 16
def splitFeatureMatrices(feature_matrix_x, feature_matrix_y, container_names_x, container_names_y):
    """
    Split the feature matrices into training and testing sets.
    """
    X_train, X_test, y_train, y_test, train_container_names_x, test_container_names_x, train_container_names_y, test_container_names_y = train_test_split(
        feature_matrix_x, feature_matrix_y, container_names_x, container_names_y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, train_container_names_x, test_container_names_x, train_container_names_y, test_container_names_y









if __name__ == "__main__":
    # Function calls
    # 1
    filtered_sources = readInResultsConf("/usr/local/bin/results/config.yml")

    # 2
    scoped_results = resultsScope(RESULTS_DIR, META_DATA, DATA_SOURCE, POWER_METERING) 

    # 3
    datasource_identifier_map = {d['source']: d['identifier'] for d in filtered_sources}
    split_task_timeseries_by_datasource(scoped_results, datasource_identifier_map)

    # 4
    datasource_identifier_map = {d['source']: d['identifier'] for d in filtered_sources}
    fin_containers = "/usr/local/bin/results/died_nextflow_containers.csv"
    fin_containers_df = pd.read_csv(fin_containers)
    container_workdirs = {row['Name']: row['WorkDir'] for idx, row in fin_containers_df.iterrows()}
    report_missing_tasks_all_sources(scoped_results, datasource_identifier_map, fin_containers_df, container_workdirs)
    
    # 5
    add_workdir_to_all_task_csvs(scoped_results, container_workdirs)
    
    # 6
    extract_slurm_job_metadata("/usr/local/bin/results/task_metadata/slurm-job-exporter/slurm_job_id")        

    # 7
    slurm_metadata_path = os.path.join(scoped_results, "task_metadata", "slurm-job-exporter", "slurm_job_id")
    update_finished_containers_with_nfcore_task(slurm_metadata_path, FIN_CONTAINERS)

    # 8
    add_nextflow_to_all_task_csvs(scoped_results, FIN_CONTAINERS)

    # 9
    pattern_temporal_signature = build_container_temporal_signatures_scoped_sources(scoped_results, FIN_CONTAINERS)

    # 10
    container_temporal_signatures = build_container_temporal_signatures_scoped_sources(scoped_results, FIN_CONTAINERS)
    
    # 11
    cleaned_container_temporal_signatures, containers_with_all_pairs, all_pairs, all_feature_names, filtered_containers_temporal_signatures, all_metrics = cleanFeatureVectors(container_temporal_signatures)
    cleaned_pattern_temporal_signatures, containers_with_all_pairs, all_pairs, all_feature_names, filtered_containers_temporal_signatures, all_metrics = cleanFeatureVectors(container_temporal_signatures)

    # 12
    # With complete temporal signatures
    feature_matrix_x, full_feature_names, container_names_x = buildFeatureMatriceInput(
        containers_with_all_pairs, cleaned_container_temporal_signatures
    )        
    pprint.pprint(full_feature_names)
    print(container_names_x)

    # With pattern temporal signatures
    feature_matrix_x_patterns, full_feature_names, container_names_x = buildFeatureMatriceInput(
        containers_with_all_pairs, cleaned_pattern_temporal_signatures
    )
    pprint.pprint(full_feature_names)
    print(container_names_x)
    
    # 13
    fin_df = addPowerToFinContainers(FIN_CONTAINERS, containers_with_all_pairs, POWER_STATS)
    
    # 14
    finished_containers_dfs_with_power = addPowerToFinContainers(FIN_CONTAINERS, containers_with_all_pairs, POWER_STATS)
    filtered_fin_df = finished_containers_dfs_with_power[finished_containers_dfs_with_power['Name'].isin(containers_with_all_pairs)].copy()
    feature_matrix_y, container_names_y = buildFeatureMatriceOutput(filtered_fin_df)

    # 15
    # Scale with full temporal signatures
    scaled_feature_matrix_x, scaled_feature_matrix_y, scaler_x, scaler_y = scaleFeatureMatrices(feature_matrix_x, feature_matrix_y)

    # Scale with pattern temporal signatures
    scaled_feature_matrix_x_pattern, scaled_feature_matrix_y_pattern, scaler_x, scaler_y = scaleFeatureMatrices(feature_matrix_x_patterns, feature_matrix_y)
 
    # 16
    # Train Test Split with full temporal signatures
    X_train, X_test, y_train, y_test, train_container_names_x, test_container_names_x, train_container_names_y, test_container_names_y = splitFeatureMatrices(scaled_feature_matrix_x, scaled_feature_matrix_y, container_names_x, container_names_y)

    # Train Test Split with pattern temporal signatures
    X_train_pattern, X_test_pattern, y_train_pattern, y_test_pattern, train_container_names_x, test_container_names_x, train_container_names_y, test_container_names_y = splitFeatureMatrices(scaled_feature_matrix_x_pattern, scaled_feature_matrix_y, container_names_x, container_names_y)

    x_train_df = pd.DataFrame(X_train, columns=full_feature_names)
    x_test_df = pd.DataFrame(X_test, columns=full_feature_names)
    y_train_df = pd.DataFrame(y_train, columns=['runtime', 'power'])
    y_test_df = pd.DataFrame(y_test, columns=['runtime', 'power'])
    

# end main