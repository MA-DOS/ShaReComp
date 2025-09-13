# Basic libraries
import shutil 
import os
import docker
import logging
import random
import sys
import time
from collections import defaultdict
import pprint
from datetime import datetime
import numpy as np
import pandas as pd
# Clustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from cca_zoo.nonparametric import KCCA

logging.basicConfig(level=logging.DEBUG,
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler(sys.stdout)],)
logger = logging.getLogger(__name__)

def truncatePeakTimeSeries(df_i, df_j):
    """
    Truncate the peak time series to the length of the shorter series.
    """
    if len(df_i) == len(df_j):
        print("Both series are of equal length:", len(df_i))
        return df_i, df_j
    min_length = min(len(df_i), len(df_j))
    df_i = df_i.iloc[:min_length]
    df_j = df_j.iloc[:min_length]
    # print("Truncated series to length:", min_length)
    return df_i, df_j

# Helper to get the according peak time series for the current nextflow task.
def getPeakTimeSeriesForTask(task_name, scoped_results, workload_type_map,type=None):
    """
    Get the peak time series for a given task name.
    """
    
    inverted_workload_type = next((k for k, v in workload_type_map.items() if v == type), None)

    current_workload_dir = os.path.join(scoped_results, inverted_workload_type) if inverted_workload_type else scoped_results

    for root, dirs, files in os.walk(current_workload_dir):
        if os.path.basename(root) == "containers":
            peak_file = os.path.join(root, f"PEAK_Series_{task_name}.csv")
            if os.path.exists(peak_file):
                if type is not None:
                    print(f"Found peak time series file for {task_name}")
                    # logger.info(f"Found peak time series file for {task_name} with workload type {type}")
                return pd.read_csv(peak_file)
    # print(f"Peak time series file not found for task: {task_name}")
    logger.info(f"Peak time series file not found for task: {task_name}")
    return None

# To get affinity score for a pair:
def get_affinity_score(type1, type2, aff_df):
    # Try both (type1, type2) and (type2, type1) for symmetry
    row = aff_df[
        ((aff_df['workload_1'] == type1) & (aff_df['workload_2'] == type2)) |
        ((aff_df['workload_1'] == type2) & (aff_df['workload_2'] == type1))
    ]
    if not row.empty:
        # logger.info(f"Affinity score for ({type1}, {type2}): {row['affinity_score'].values[0]}")
        return row['affinity_score'].values[0]
    else:
        return None

def computeTaskSignatureDistances(scoped_results, filtered_tasks_temporal_signatures, container_to_nextflow, workload_type_map):
    """
    Compute the distances between task signatures in the feature space.
    Returns a distance matrix based on the custom distance function.
    
    Args:
        scoped_results: Result dictionary holding the peak time series for each task's metric.
    Returns:
        distance_matrix: Numpy array of distances between task signatures.
    """
    
    # Get the affinity scores of the workload experiments
    aff_df = pd.read_csv("affinity_score_matrix.csv")
    if aff_df is None or aff_df.empty:
        raise ValueError("Affinity score matrix could not be loaded or is empty.")
    # logger.info("Loaded affinity score matrix.")
    
    # Use the keys of cleaned_container_temporal_signatures as task identifiers
    nextflow_jobs = list(filtered_tasks_temporal_signatures.keys())
    logger.info(f"Nextflow jobs to process: {nextflow_jobs}")
    
    filtered_jobs = []
    for job in nextflow_jobs:
        # print(f"Processing job: {job}")
        logger.info(f"Processing job: {job}")
        # logger.info(f"Getting peak time series for job: {job}")
        peak_df = getPeakTimeSeriesForTask(container_to_nextflow[job], scoped_results, workload_type_map)
        if peak_df is not None and not peak_df['peak_value'].nunique() == 1:
            filtered_jobs.append(job)

    distance_matrix = np.full((len(filtered_jobs), len(filtered_jobs)), np.nan)
    
    # Catch the calculated distances for the job pair i,j for distribution mapping
    distances = []

    for i in range(len(filtered_jobs)):
        for j in range(i + 1, len(filtered_jobs)):
            job_i = filtered_jobs[i]
            job_j = filtered_jobs[j]
            workloads_i = list(filtered_tasks_temporal_signatures[job_i]['temporal_signatures'].keys())
            workloads_j = list(filtered_tasks_temporal_signatures[job_j]['temporal_signatures'].keys())

            # Reset the temporary terms for each job pair
            distance_i_j = 0.0
            # print("Reset distance for next job pair:", job_i, job_j)
            
            # Keep track of processed affinity pairs per task
            processed_pairs = set()
            for wi in workloads_i:
                for wj in workloads_j:
                    type_i = workload_type_map.get(wi, wi)
                    type_j = workload_type_map.get(wj, wj)

                    pair = frozenset([type_i, type_j])
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)

                    # -------------------------------------------------------------------------------------------------
                    # TERM 1 of the distance equation for each job i, j: Get the affinity score for the pair of workload types
                    # -------------------------------------------------------------------------------------------------
                    affinity_score = get_affinity_score(type_i, type_j, aff_df)
                    # print(f"Processing jobs {job_i} and {job_j} with workload type {type_i} vs {type_j}: affinity_score={affinity_score}")

                    # -------------------------------------------------------------------------------------------------
                    # Term 2 of the distance equation for each job i, j: Get the peak time series of the workload type 1
                    # -------------------------------------------------------------------------------------------------
                    # If one time series is constant, set the distance to 0.
                    # print("Computing correlation for workload types in TERM 2:", type_i, type_i)

                    peak_df_i = getPeakTimeSeriesForTask(container_to_nextflow[job_i], scoped_results, workload_type_map, type_i)
                    peak_df_j = getPeakTimeSeriesForTask(container_to_nextflow[job_j], scoped_results,workload_type_map, type_i)

                    # Truncate the peak time series in place to the same lenght
                    trun_peak_df_i, trun_peak_df_j = truncatePeakTimeSeries(peak_df_i, peak_df_j)
                    
                    # Compute the correlation for the peak time series of the same workload type
                    try:
                        corr_i_j_R1 = pearsonr(trun_peak_df_i['peak_value'], trun_peak_df_j['peak_value'])[0] 
                        if corr_i_j_R1 is None or np.isnan(corr_i_j_R1):
                            corr_i_j_R1 = 0.0
                            # print(f"Setting correlation to 0 for {job_i} vs {job_j} with workload type {type_i} due to NaN value.")
                        # print(f"Correlation for {job_i} vs {job_j} with workload type {type_i}: {corr_i_j_R1}")
                    except ValueError as e:
                        # print(f"Error computing correlation for {job_i} vs {job_j} with workload type {type_i}{type_i}: {e}")
                        corr_i_j_R1 = 0.0
                        # print(f"Setting correlation to 0 for {job_i} vs {job_j} with workload type {type_i}{type_i} due to one of two series being constant.")
                    
                    
                    # -------------------------------------------------------------------------------------------------
                    # TERM 3 of the distance equation for each job i, j: Get the correlation of the peak time series of the identical workload types 2
                    # -------------------------------------------------------------------------------------------------
                    # If one time series is constant, set the distance to 0.
                    # print("Computing correlation for workload types in TERM 3:", type_j, type_j)
                    
                    peak_df_i = getPeakTimeSeriesForTask(container_to_nextflow[job_i], scoped_results, workload_type_map,type_j)
                    peak_df_j = getPeakTimeSeriesForTask(container_to_nextflow[job_j], scoped_results, workload_type_map,type_j)
                    
                    # Truncate the peak time series in place to the same lenght
                    trun_peak_df_i, trun_peak_df_j = truncatePeakTimeSeries(peak_df_i, peak_df_j)

                    # Compute the correlation for the peak time series of the same workload type
                    try:
                        corr_i_j_R2 = pearsonr(trun_peak_df_i['peak_value'], trun_peak_df_j['peak_value'])[0] 
                        if corr_i_j_R2 is None or np.isnan(corr_i_j_R2):
                            corr_i_j_R2 = 0.0
                            # print(f"Setting correlation to 0 for {job_i} vs {job_j} with workload type {type_i} due to NaN value.")
                        # print(f"Correlation for {job_i} vs {job_j} with workload type {type_j} {type_j}: {corr_i_j_R2}")
                    except ValueError as e:
                        # print(f"Error computing correlation for {job_i} vs {job_j} with workload type {type_i}: {e}")
                        corr_i_j_R2 = 0.0
                        # print(f"Setting correlation to 0 for {job_i} vs {job_j} with workload type {type_j} {type_j} due to one of two series being constant.")

                    # -------------------------------------------------------------------------------------------------
                    # Sum over jobs i,j per metric pair
                    # -------------------------------------------------------------------------------------------------
                    distance_i_j += affinity_score * corr_i_j_R1 * corr_i_j_R2
                
            # -------------------------------------------------------------------------------------------------
            # Write distance matrix entry for the job pair i,j
            # -------------------------------------------------------------------------------------------------
            # print(f"Distance for job pair ({job_i}, {job_j}): {distance_i_j}")
            
            # Write the distances into list for distribution mapping
            distances.append(distance_i_j)
            
            distance_matrix[i, j] = distance_i_j
            # I think only one triangle of the matrix is enough. May increase performance.
            distance_matrix[j, i] = distance_i_j

    print("Distance matrix computed.")
    # Fill the diagonal with zeros (distance to self is zero)
    np.fill_diagonal(distance_matrix, 0.0)
    # print("Distance matrix:\n", distance_matrix)

    distance_df = pd.DataFrame(distance_matrix, index=filtered_jobs, columns=filtered_jobs)
                    
    return distance_matrix, distance_df

def computeMergeThreshold(distance_matrix):

    # n_quantiles is set to the training set size rather than the default value
    # to avoid a warning being raised by this example
    qt = QuantileTransformer(
        n_quantiles=len(distance_matrix), output_distribution="normal" 
    )

    # transformed_distances = qt.fit_transform(np.array(distances)).reshape(-1, 1)
    transformed_distances = qt.fit_transform(distance_matrix)
    # print(transformed_distances)

    # Determine threshold
    # 1. Get the lower triangle of the distance matrix without the diagonal
    tril_values = transformed_distances[np.tril_indices_from(transformed_distances, k=-1)]
    tril_values_raw = distance_matrix[np.tril_indices_from(distance_matrix, k=-1)]

    # 2. Compute the n-th percentile
    threshold_transformed = np.percentile(tril_values, 15)
    # I only use the raw thesholds for now.
    threshold_raw = np.percentile(tril_values_raw, 40)

    print("Raw Threshold for current distance matrix:", threshold_raw)

    return threshold_raw

# Run agglomerative clustering algorithm on the distance matrix
def runAgglomerativeClustering(distance_matrix, threshold):
    """
    Run agglomerative clustering on the distance matrix.
    Returns the cluster labels for each task.
    
    Args:
        distance_matrix: Numpy array of distances between task signatures.
    Returns:
        cluster_labels: Numpy array of cluster labels for each task.
    """
    clustering = AgglomerativeClustering(n_clusters = None, metric='precomputed', linkage='average', compute_full_tree=True, compute_distances=False, distance_threshold=threshold).fit(distance_matrix)
    cluster_labels = clustering.labels_
    # print(f"Number of clusters found: {len(set(cluster_labels))}")
    # print(f"Cluster labels: {cluster_labels}")
    job_to_cluster = dict(zip(distance_matrix.index, cluster_labels))
    return job_to_cluster

def clusterToJobs(job_to_cluster):
    """
    Convert job to cluster mapping to cluster to jobs mapping.
    Returns a dictionary where keys are clusters and values are lists of jobs in those clusters.
    """
    
    cluster_to_jobs = defaultdict(list)
    for k, v in job_to_cluster.items():
        cluster_to_jobs[v].append(k)

    # Collect keys to delete
    # keys_to_delete = [k for k, v in cluster_to_jobs.items() if len(v) == 1]
    # for k in keys_to_delete:
    #     del cluster_to_jobs[k]

    return cluster_to_jobs

def flatten_signature_dict(signature_dict):
    # signature_dict: the nested dict for one job
    flat = {}
    for workload, metrics in signature_dict.items():
        for metric, features in metrics.items():
            for feature, value in features.items():
                flat_key = f"{workload}/{metric}/{feature}"
                flat[flat_key] = value
    return flat

def updateTaskSignatureToColoc(cluster_to_jobs, shortened_filtered_tasks_temporal_signatures):
    
    logging.info("Updating task signatures to coloc signatures...")
    logging.info("Clusters to jobs mapping: %s", cluster_to_jobs)
    logging.info("Shortened filtered tasks temporal signatures keys: %s", shortened_filtered_tasks_temporal_signatures.keys())
    coloc_signatures = {}
        
    for k, v in cluster_to_jobs.items():
        
        if len(v) == 1:
            # print("Single job in cluster, skipping:", v)
            continue

        # Initialize the coloc task signature
        coloc_dataframes = []
        
        for job in v:
            # print(job) 
            vector = shortened_filtered_tasks_temporal_signatures[job]['temporal_signatures']
            flattened_vector = flatten_signature_dict(vector)
            df = pd.DataFrame([flattened_vector])
            coloc_dataframes.append(df)
            
        # Merge the dataframes for the coloc task and write back to updated dict
        concatenated_df = pd.concat(coloc_dataframes, ignore_index=True)
        summed_df = concatenated_df.sum()

        # Convert the summed DataFrame to a dictionary
        coloc_signatures[k] = summed_df.to_dict()

    return coloc_signatures         
            
def buildColocFeatureMatrix(coloc_signatures):
    """
    Build feature matrices for clusters from the coloc_signatures dictionary.
    Each cluster will have a feature vector of shape (1, 90).
    """

    coloc_feature_matrix = {}
    
    for cluster_id, metrics in coloc_signatures.items():
        # Flatten all metric arrays into a single feature vector
        feature_vector = []
        for metric, values in metrics.items():
            if isinstance(values, np.ndarray):
                feature_vector.extend(values.tolist())
            else:
                raise ValueError(f"Expected numpy array for metric '{metric}', but got {type(values)}")
        
        coloc_feature_matrix[cluster_id] = np.array(feature_vector).reshape(1, -1)
    
    return coloc_feature_matrix

def predictKCCALinReg(X_unseen, reg, kcca):
    logger.info("Predicting with KCCA Linear Regression model...")
    try:
        X_unseen_scaled = StandardScaler().fit_transform(X_unseen)
        logger.info("Scaled unseen data for prediction.")
        X_unseen_scaled_latent = kcca.transform((X_unseen_scaled, None))[0]
        Y_pred = reg.predict(X_unseen_scaled_latent)
        return Y_pred
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise None
