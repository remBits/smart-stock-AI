def prepare_data(df):
    df_daily = (
        df
        .groupby('Date')['Units Sold']
        .sum()
        .reset_index()
    )

    df_daily = df_daily.rename(columns={
        'Date': 'ds',
        'Units Sold': 'y'
    })

    return df_daily.sort_values('ds')
