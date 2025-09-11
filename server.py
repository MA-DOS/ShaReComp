from fastapi import FastAPI as api, HTTPException
from pydantic import BaseModel 
from typing import Dict,List
from datetime import datetime
import random
import random
import time

# Import my offline processing and training procedure
import offline_init


# Start up the server
app = api()

@app.on_event("startup")
async def startup_event():
# Run the offline stuff
    print("Running offline initialization...")
    offline_init.main()  
    print("Offline initialization completed.")

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
        # TODO: Implement
        # Placeholder for clustering logic
        clusters = {
            1: ["nf-core/rnaseq", "nf-core/methylseq"],
            2: ["nf-core/atacseq", "nf-core/cutadapt"],
            3: ["nf-core/sarek"]
        }

        ### Clustering logic from ShaReComp ###

        

        # Generate a unique run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d')}_{random.randint(1000, 9999):04x}"

        # Return the response
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
                    cluster_predictions["kcca"] = {
                        "runtime": round(random.uniform(0.5, 0.7), 3),
                        "power": round(random.uniform(115, 120), 1)
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