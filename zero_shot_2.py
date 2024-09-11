#%%
# Class to Create and Forecast time series using Amazon Chronos

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import json
import torch
import random
import transformers
from statsmodels.tsa.seasonal import seasonal_decompose
from chronos import ChronosPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.common import ListDataset
from tqdm.auto import tqdm


class ChronosForecaster:

    def __init__(self) -> None:
        transformers.set_seed(42)
        self.limit_pred_len = True
        pass

    @staticmethod
    def is_gpu():
        return torch.cuda.is_available()
    
    def get_series_decomposition(self, series, model='additive', period=12):
        result = seasonal_decompose(series, model=model, period=period)
        result.plot()
        return result

    def calculate_metrics(self, predicted, ground_truth, metrics_df):
        mae = mean_absolute_error(ground_truth, predicted)
        mse = mean_squared_error(ground_truth, predicted)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(ground_truth, predicted)
        r_squared = r2_score(ground_truth, predicted)

        return {
            'MAE': round(mae,2),
            'MSE': round(mse,2),
            'RMSE': round(rmse,2),
            'MAPE': round(mape,2),
            'R-squared': round(r_squared,2),
            'MASE': metrics_df.iloc[0, 0],
            'WQL': metrics_df.iloc[0, 1]
        }
    
    # Split by percentage
    def split_train_test(self, df, test_size=0.2, limit_pred_length=True):
        self.df = df
        _, _, self.y_train, self.y_test = train_test_split(df, df, test_size=test_size, random_state=42, shuffle=False)
        clipped = False
        self.limit_pred_len = limit_pred_length
        if len(self.y_test) > 64 and self.limit_pred_len:
            print(f"[!] Test array size is {len(self.y_test)} which is > 64!")
            print("Clipping test data size to first 64 samples for acceptable accuracy")
            self.y_test = self.y_test[:64]
            self.y_train = self.y_train[:-len(self.y_test)]
            clipped = True
        else:
            print('[!] WARNING: limit_pred_length is disabled! Far future prediction will have low accuracy!')
        return self.y_train, self.y_test, clipped
    
    # Split by number of samples to be predicted
    def split_train_test_by_sample_size(self, df, target_column, test_sample_size=24):
        self.df = df
        self.y_test:pd.DataFrame = df.iloc[-test_sample_size:][target_column]
        self.y_train:pd.DataFrame = df.iloc[:len(df)-test_sample_size][target_column]

        test_timestamp=df.iloc[-test_sample_size:]['timestamp']
        test_timestamp = test_timestamp.sort_values()

        # Prepare ListDataset for GluonTS
        freq = "D"  # Daily frequency
        test_data = ListDataset(
            [{"start": test_timestamp.iloc[0], "target": self.y_test.values}],  # Assuming 'close' is the target column
            freq=freq
        )

        _, test_data = split(test_data, offset=-len(self.y_test))
        test_data = test_data.generate_instances(len(self.y_test))

        self.limit_pred_len = False
        
        return self.y_train, self.y_test, test_data

    def predict(self, test_data, model_name="amazon/chronos-t5-small", num_samples=20, temp=1, top_k=50, top_p=1, debug=True):
        device = "cuda"
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        try:
            context = torch.tensor(self.y_train)
        except ValueError:
            context = torch.tensor(self.y_train.astype(float).values)
        prediction_length = len(self.y_test)
        if debug:
            print(f"Using {device}\nContext length = {len(self.y_train)}\nForecast length = {prediction_length}\nSample size = {num_samples}")
        self.forecast = self.pipeline.predict(
            context,
            prediction_length,
            num_samples=num_samples,
            temperature=temp,
            top_k=int(top_k),
            top_p=top_p,
            limit_prediction_length=self.limit_pred_len
        )

        # Generate forecast samples ==================
        forecast_samples = []
        for batch in tqdm(batcher(test_data.input, batch_size=32)):
            context = [torch.tensor(entry["target"]) for entry in batch]
            forecast_samples.append(
                self.pipeline.predict(
                    context,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                ).numpy()
            )
        forecast_samples = np.concatenate(forecast_samples)

        # Convert forecast samples into gluonts SampleForecast objects
        sample_forecasts = []
        for item, ts in zip(forecast_samples, test_data.input):
            forecast_start_date = ts["start"] + len(ts["target"])
            sample_forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        # Evaluate
        metrics_df = evaluate_forecasts(
            sample_forecasts,
            test_data=test_data,
            metrics=[
                MASE(),
                MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
            ],
        )

        # self.forecast_index = range(len(self.y_train), len(self.y_train) + prediction_length)
        self.forecast_index = range(self.y_test.index[0], self.y_test.index[-1] + 1)
        self.low, self.median, self.high = np.quantile(self.forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        # self.median = self.forecast[0].numpy().mean(axis=0)           # Mean instead of median
        self.metrics = self.calculate_metrics(self.median, self.y_test, metrics_df)
        self.params = json.dumps({
                "model": model_name,
                "top_p": top_p,
                "top_k": top_k,
                "tempearature": temp,
                "num_samples": num_samples,
                "context_length": len(self.y_train),
                "prediction_length": prediction_length
            },
            indent=2)
        
        

        return (self.forecast, self.metrics, self.median)
    
    def plot_forecast(self, series:pd.Series):
        plt.figure(figsize=(8, 4))
        plt.plot(series, color="royalblue", label="Historical Data")
        plt.plot(self.y_test, color="green", label="Ground Truth")
        plt.plot(self.forecast_index, self.median, color="tomato", label="Median Forecast")
        plt.fill_between(self.forecast_index, self.low, self.high, color="tomato", alpha=0.3, label="80% Prediction Interval")
        plt.title('CHRONOS Forecasting')
        plt.figtext(1, 0.55, "Metrics:\n" + json.dumps(self.metrics, indent=2))
        plt.figtext(1, 0.1, "Params:\n" + self.params)
        plt.legend()
        plt.grid()
        return plt

    # Run tunning experiments for model param optimisation
    def tune_model(self, num_iterations, series, metric_name='MSE'):
        list_metrics = []

        for i in range(num_iterations):
            model_name = random.choice(["amazon/chronos-t5-small"])
            top_p = random.randint(10, 100)
            top_k = round(random.randrange(1,10,1)/10,1)
            temp = round(random.randrange(1,10,1)/10,1)
            num_samples = random.randint(9,51)

            print(f'Running iteration {i} -> top_p = {top_p}, top_k = {top_k}, temp = {temp}, num_samples = {num_samples}')
            
            _, result, _ = self.predict(
                model_name=model_name,
                num_samples=num_samples,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                debug=False
            )

            plot = self.plot_forecast(series)
            plot.savefig(f'results/{series.name}-{i}.png', bbox_inches='tight')
            plot.close()

            result['filename'] = f'results/{series.name}-{i}.png'

            list_metrics.append(result | json.loads(self.params))

        return pd.DataFrame(sorted(list_metrics, key=lambda x: x[metric_name]))
    
df = pd.read_csv('/home/tifzaki/dataset/LQ45-daily/ANTM.csv')

PRED_DURATION = 365 * 1 # A multiplier of 24 as I was working with hourly dataset
FORECAST_LEN = 64 # Trying to keep this less than 64 data points for best chronos performance
PRED_START = len(df) - PRED_DURATION # The index where to split context and test data from your df
CONTEXT_LEN = 4000 # Choose this wisely from above hyperparameter tuning experiment

agg_metrics = []
pred_indices = [i for i in range(len(df), PRED_START, -FORECAST_LEN)][::-1]


og_df = df.copy()
TARGET_COLUMN = 'close'

for idx in pred_indices:
    print(f'Running with idx = {idx}')
    fc = ChronosForecaster()
    
    df = og_df[(idx-CONTEXT_LEN):(idx+FORECAST_LEN)].copy()

    if idx > len(og_df):
        continue

    df.dropna(inplace=True)

    context, test, test_data = fc.split_train_test_by_sample_size(df, target_column=TARGET_COLUMN, test_sample_size=FORECAST_LEN)

    forecast, metrics, median = fc.predict( test_data, model_name="amazon/chronos-t5-small", num_samples=3)

    metrics['idx_context_start'] = idx - CONTEXT_LEN
    metrics['idx_context_end'] = idx
    metrics['idx_forecast_start'] = idx + 1
    metrics['idx_forecast_end'] = idx + FORECAST_LEN

    agg_metrics.append(metrics | json.loads(fc.params))

    fc.plot_forecast(df[TARGET_COLUMN])
# %%
