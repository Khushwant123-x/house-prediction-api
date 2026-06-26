import io
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import requests

# Google Drive direct download links
model_url = "https://drive.google.com/uc?export=download&id=1eYml3WNP_-zm8HBaNXZ7pNgMLdUcbyaH"

feature_url = "https://drive.google.com/uc?export=download&id=1Z-5vC7M3quoftvLmzceSqwJuQzseX-4E"

def download_file(url, filename):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

download_file(model_url, "house_model.joblib")
download_file(feature_url, "house_features.joblib")

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

# input schema
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of neighbourhood")
    HouseAge: float = Field(gt=0, description="Average age of house in the block")
    AveRooms: float = Field(gt=0, description="Average number of rooms in the house")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms in the house")
    Population: float = Field(gt=0, description="Total Population")
    AveOccup: float = Field(gt=0, description="Average occupancy per household") 
    Latitude: float = Field(ge=32, le=42, description="Latitude")
    Longitude: float = Field(ge=-125, le=114, description="Longitude")

@app.get("/")
def home():
    return {
        "message": "California house prediction API",
        "status": "running",
        "endpoint": "Send POST request to /prediction"
    }

@app.get("/health")   
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"
    }

@app.post("/prediction")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousands",
            "fidence_range": f"${price_usd-39000:,.0f} to ${price_usd+39000:,.0f}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )
    
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents)) 

    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file: {missing_cols}"
        )
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="Your file is empty"
        )

    try:
        prediction = model.predict(df[required_columns])

        df["predicted_price_usd"] = prediction * 100000
        df["predicted_price_usd"] = df["predicted_price_usd"].apply(lambda x: f"${x:,.0f}")

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
