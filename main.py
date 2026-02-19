from data_loader import load_data, filter_pyme
from preprocessing import prepare_data
from model import train_model, make_forecast
from evaluation import evaluate
from reporting import generate_report


def main():

    df = load_data("retail_store_inventory.csv")

    df_pyme = filter_pyme(df, "S001", "P0001")

    df_prophet = prepare_data(df_pyme)

    model = train_model(df_prophet)

    forecast_df = make_forecast(model, periods=30)

    mae_prophet, mae_naive = evaluate(df_prophet, forecast_df)

    print("MAE Prophet:", mae_prophet)
    print("MAE Naive:", mae_naive)

    generate_report(df_prophet, forecast_df, stock_actual=100)


if __name__ == "__main__":
    main()
