from fastapi import FastAPI, UploadFile, File
import shutil
import os

from data_loader import load_data
from preprocessing import prepare_data
from model import train_model, make_forecast

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_path = "temp.csv"

    # Guardar archivo temporal
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Procesar
    df = load_data(temp_path)
    df_prophet = prepare_data(df)
    model = train_model(df_prophet)
    forecast = make_forecast(model, periods=30)

    pred_30 = int(forecast.tail(30)['yhat'].mean().round())

    # (opcional) borrar archivo temporal
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {"prediccion_30_dias": pred_30}
