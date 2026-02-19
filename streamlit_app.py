import streamlit as st
import pandas as pd

from data_loader import load_data
from preprocessing import prepare_data
from model import train_model, make_forecast

st.title("Forecast de Demanda para PYMEs")

uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    df_prophet = prepare_data(df)

    model = train_model(df_prophet)

    forecast = make_forecast(model, periods=30)

    pred_30 = int(forecast.tail(30)['yhat'].mean().round())

    st.success(f"Demanda promedio proyectada próximos 30 días: {pred_30} unidades/día")

    st.line_chart(
        forecast[['ds','yhat']].set_index('ds').tail(60)
    )
