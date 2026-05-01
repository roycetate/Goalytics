from fastapi import FastAPI
import json
import os

app = FastAPI(title="Goalytics API")

DATA_FILE = "../data/live_stats.json"


@app.get("/")
def home():
    return {"message": "Goalytics Backend Running ⚽"}


@app.get("/stats")
def get_stats():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    else:
        return {"message": "No data available yet"}