from fastapi import FastAPI
import uvicorn
import os
from main import run_pipeline

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the BL + News Analysis API."}

@app.get("/run_pipeline")
def run_pipeline_endpoint(n_days: int = 10):
    """
    Run the pipeline for the last `n_days`. Returns a JSON with summarized data.
    """
    df_plot, _ = run_pipeline(n_days=n_days)
    if df_plot is None:
        return {"status": "No data or not enough days to run pipeline."}

    # Convert the final DataFrame to a list of records
    records = df_plot.to_dict(orient="records")
    return {
        "status": "success",
        "n_days": n_days,
        "data": records
    }

if __name__ == "__main__":
    # This launches the FastAPI server on localhost:8000 by default
    uvicorn.run(app, host="0.0.0.0", port=8000)
