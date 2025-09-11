from fastapi import FastAPI as api, HTTPException
from pydantic import BaseModel 
from typing import Dict,List
from collections import defaultdict
from datetime import datetime
import random
import random
import time

# Import my offline processing and training procedure
import offline_init

# Import ShaReComp helpers
from utils import helpers

# Start up the server
app = api()

@app.on_event("startup")
async def startup_event():
# Run the offline stuff
    print("Running offline initialization...")
    global filtered_tasks_temporal_signatures,scoped_results,containerToNfCore, reg_model, trainedPowerPredictor, trainedRuntimePredictor
    filtered_tasks_temporal_signatures, scoped_results, containerToNfCore, reg_model, trainedPowerPredictor, trainedRuntimePredictor = offline_init.main()
    print("Offline initialization completed.")

# Global variable to store clusters
# TODO: Might need to swithc to fastapi state
# Overwrite the clusters for each request
clusters_store = {}
# Overwrite the filteres signatuers for each request
filtered_signatures = {}

# --- Endpoint 1: Clusterize nf-core jobs ---

# Request
class ClusterizeJobsRequest(BaseModel):
    job_names: List[str] # Holds a list of nf-core jobs to be clusterd

# Model
class Cluster(BaseModel):
    cluster_id: int
    jobs: List[str]

# Response
class ClusterizeJobsResponse(BaseModel):
    run_id: str
    clusters: Dict[int, List[str]]

# Define first POST endpoint
@app.post("/clusterize_jobs", response_model=ClusterizeJobsResponse)
def clusterize_jobs(request: ClusterizeJobsRequest):
    """
    Endpoint to clusterize nf-core jobs based on historical data.
    """ 

    # Validate the input
    if not request.job_names:
        raise HTTPException(status_code=400, detail="Unknown or empty job names.")

    try:
        # Get a copy of filtered_tasks_temporal_signatures
        print("Filtering the offline-built signatures...")
        filtered_signatures = filtered_tasks_temporal_signatures.copy()
        
        # Filter according to the jobs in the request
        filtered_signatures = {nf_core_job: signature for nf_core_job, signature in filtered_signatures.items() if nf_core_job in request.job_names}
        print("Filtered signatures:", filtered_signatures.keys())

        # Pass the scoped_results folder, the filtered signatures and the mapping of containers to nf-core jobs to the Distance Computation
        workload_type_map = {
        "task_memory_data": "mem",
        "task_cpu_data": "cpu",
        "task_disk_data": "fileio"
        }
            
        distance_matrix,distance_df = helpers.computeTaskSignatureDistances(scoped_results, filtered_signatures, containerToNfCore, workload_type_map)
        print("Computed distance matrix.")
        print(distance_matrix)

        # Compute the merge threshold
        threshold_raw = helpers.computeMergeThreshold(distance_matrix)
        
        # Run the clustering algorithm
        job_to_cluster = helpers.runAgglomerativeClustering(distance_df, threshold_raw)
        
        # Output the clusters
        cluster_to_jobs = helpers.clusterToJobs(job_to_cluster)

        # Convert to type that pydantic takes
        clusters = dict(cluster_to_jobs)

        # Generate a unique run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d')}_{random.randint(1000, 9999):04x}"

        # Return the response
        clusters_store["clusters"] = clusters
        return ClusterizeJobsResponse(run_id=run_id, clusters=clusters)

    except Exception as e:
        # Handle clustering failure
        raise HTTPException(status_code=500, detail="Clustering failed") from e

# --- Endpoint 2: Predict runtime and power consumption for job-clusters ---

# Model
class PredictionModel(BaseModel):
    model_type: List[str]
    # Both models predict both runtime and power consumption, rest is handled in simulator environment
    # categories: List[str]

# Request
class PredictRequest(BaseModel):
    cluster_ids: List[str] 
    prediction_models: List[PredictionModel]  # List of models to use for prediction

# Response
class PredictionResponse(BaseModel):
    run_id: str
    predictions: Dict[int, Dict[str, Dict[str, float]]]

# Define second POST endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    """
    Endpoint to predict runtime and power consumption for job-clusters.
    """ 

    # Validate the input
    if not request.cluster_ids or not request.prediction_models:
        raise HTTPException(status_code=400, detail="Unknown or empty cluster IDs or prediction models.")

    try:
        # TODO: Implement
        # Placeholder for prediction logic
        predictions = {}
        for cluster_id in request.cluster_ids:
            cluster_predictions = {}
            for model in request.prediction_models:
                if not model.model_type:
                    raise HTTPException(status_code=400, detail="Model type cannot be empty.")
                
                model_type = model.model_type[0]  # Safely access the first model type
                if model_type == "kcca":
                    # Get the current clustering as the simulator always 1) asks for the clusters and 2) asks for predictions for each cluster
                    # TODO: Might need logic to recompute clusters and be able to properly update their task signatures
                    clusters = clusters_store.get("clusters", {})
                        
                    # Update the clusters signatures
                    coloc_signatures = helpers.updateTaskSignatureToColoc(clusters, filtered_signatures)

                    # Transform the signatures into a model acceptable format
                    coloc_feature_matrix = helpers.buildColocFeatureMatrix(coloc_signatures)
                    
                    # Call the KCCA model to predict runtime and power consumption
                    Y_pred = helpers.predictKCCALinReg(coloc_feature_matrix[cluster_id], reg_model)
                    
                    cluster_predictions["kcca"] = {
                        
                        
                        # "runtime":12,
                        # "power": 24,
                    }
                elif model_type == "rfr":
                    cluster_predictions["rfr"] = {
                        "runtime": round(random.uniform(0.6, 0.8), 3),
                        "power": round(random.uniform(118, 122), 1)
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
            predictions[int(cluster_id)] = cluster_predictions 

        # Generate a unique run ID
        run_id = f"pred_{datetime.now().strftime('%Y%m%d')}_{random.randint(1000, 9999):04x}"

        # Return the response
        return PredictionResponse(run_id=run_id, predictions=predictions)

    except Exception as e:
        # Handle prediction failure
        raise HTTPException(status_code=500, detail="Prediction failed") from e