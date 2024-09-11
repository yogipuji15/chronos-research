#%%
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from gluonts.dataset.common import ListDataset

from chronos import ChronosPipeline

# Load dataset
batch_size = 32
num_samples = 20
df = pd.read_csv('/home/tifzaki/dataset/LQ45-daily/ANTM.csv')

# Ensure the date column is in datetime format and sort by date
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

# Prepare ListDataset for GluonTS
prediction_length = 64
freq = "D"  # Daily frequency
dataset = ListDataset(
    [{"start": df['timestamp'].iloc[0], "target": df['close'].values}],  # Assuming 'close' is the target column
    freq=freq
)

# Load Chronos
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

# Split dataset for evaluation
train_data, test_data = split(dataset, offset=-prediction_length)
test_data = test_data.generate_instances(prediction_length)


# Generate forecast samples
forecast_samples = []
for batch in tqdm(batcher(test_data.input, batch_size=32)):
    context = [torch.tensor(entry["target"]) for entry in batch]
    forecast_samples.append(
        pipeline.predict(
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
forecast_index = range(test_data.index[0], test_data.index[-1] + 1)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
# median = forecast[0].numpy().mean(axis=0)           # Mean instead of median
metrics = calculate_metrics(median, y_test, metrics_df)
params = json.dumps({
        "model": model_name,
        "top_p": top_p,
        "top_k": top_k,
        "tempearature": temp,
        "num_samples": num_samples,
        "context_length": len(y_train),
        "prediction_length": prediction_length
    },
    indent=2)


metrics_df

print(metrics_df.iloc[0, 1])
metrics_df