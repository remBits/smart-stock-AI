from sklearn.metrics import mean_absolute_error

def evaluate(df_prophet, forecast):
    df_eval = forecast.merge(
        df_prophet[['ds','y']],
        on='ds',
        how='inner'
    )

    mae_prophet = mean_absolute_error(df_eval['y'], df_eval['yhat'])

    df_eval['naive'] = df_eval['y'].shift(1)
    mae_naive = mean_absolute_error(
        df_eval['y'][1:],
        df_eval['naive'][1:]
    )

    return mae_prophet, mae_naive
