import matplotlib.pyplot as plt

def generate_report(df_prophet, forecast, stock_actual=100):

    pred_30 = int(forecast.tail(30)['yhat'].mean().round())
    pred_7 = int(forecast.tail(7)['yhat'].mean().round())
    dias_stock = round(stock_actual / pred_30, 1)

    print(f"Demanda próxima semana: {pred_7}")
    print(f"Demanda próximos 30 días: {pred_30}")
    print(f"Stock dura aprox: {dias_stock} días")

    historico_30 = df_prophet.tail(30)
    forecast_30 = forecast.tail(30)

    plt.figure(figsize=(10,5))
    plt.plot(historico_30['ds'], historico_30['y'])
    plt.plot(forecast_30['ds'], forecast_30['yhat'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
