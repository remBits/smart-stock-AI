from prophet import Prophet

def train_model(df_prophet):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df_prophet)
    return model

def make_forecast(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    return forecast
