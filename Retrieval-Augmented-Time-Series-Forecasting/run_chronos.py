import logging
from pathlib import Path
from typing import Iterable, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from time_series_utils import augment_time_series, denormalize_predictions, normalize_context
from chronos_local import ChronosPipeline
from datasets import Dataset

app = typer.Typer(pretty_exceptions_enable=False)


# Taken from pandas._libs.tslibs.dtypes.OFFSET_TO_PERIOD_FREQSTR
offset_alias_to_period_alias = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q",
    "BQE-JAN": "Q",
    "BQE-FEB": "Q",
    "BQE-MAR": "Q",
    "BQE-APR": "Q",
    "BQE-MAY": "Q",
    "BQE-JUN": "Q",
    "BQE-JUL": "Q",
    "BQE-AUG": "Q",
    "BQE-SEP": "Q",
    "BQE-OCT": "Q",
    "BQE-NOV": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "Y": "Y",
    "BYE": "Y",
    "BYE-DEC": "Y",
    "BYE-JAN": "Y",
    "BYE-FEB": "Y",
    "BYE-MAR": "Y",
    "BYE-APR": "Y",
    "BYE-MAY": "Y",
    "BYE-JUN": "Y",
    "BYE-JUL": "Y",
    "BYE-AUG": "Y",
    "BYE-SEP": "Y",
    "BYE-OCT": "Y",
    "BYE-NOV": "Y",
    "YS": "Y",
    "BYS": "Y",
    "QS-JAN": "Q",
    "QS-FEB": "Q",
    "QS-MAR": "Q",
    "QS-APR": "Q",
    "QS-MAY": "Q",
    "QS-JUN": "Q",
    "QS-JUL": "Q",
    "QS-AUG": "Q",
    "QS-SEP": "Q",
    "QS-OCT": "Q",
    "QS-NOV": "Q",
    "QS-DEC": "Q",
    "BQS-JAN": "Q",
    "BQS-FEB": "Q",
    "BQS-MAR": "Q",
    "BQS-APR": "Q",
    "BQS-MAY": "Q",
    "BQS-JUN": "Q",
    "BQS-JUL": "Q",
    "BQS-AUG": "Q",
    "BQS-SEP": "Q",
    "BQS-OCT": "Q",
    "BQS-NOV": "Q",
    "BQS-DEC": "Q",
    "YS-JAN": "Y",
    "YS-FEB": "Y",
    "YS-MAR": "Y",
    "YS-APR": "Y",
    "YS-MAY": "Y",
    "YS-JUN": "Y",
    "YS-JUL": "Y",
    "YS-AUG": "Y",
    "YS-SEP": "Y",
    "YS-OCT": "Y",
    "YS-NOV": "Y",
    "YS-DEC": "Y",
    "BYS-JAN": "Y",
    "BYS-FEB": "Y",
    "BYS-MAR": "Y",
    "BYS-APR": "Y",
    "BYS-MAY": "Y",
    "BYS-JUN": "Y",
    "BYS-JUL": "Y",
    "BYS-AUG": "Y",
    "BYS-SEP": "Y",
    "BYS-OCT": "Y",
    "BYS-NOV": "Y",
    "BYS-DEC": "Y",
    "Y-JAN": "Y-JAN",
    "Y-FEB": "Y-FEB",
    "Y-MAR": "Y-MAR",
    "Y-APR": "Y-APR",
    "Y-MAY": "Y-MAY",
    "Y-JUN": "Y-JUN",
    "Y-JUL": "Y-JUL",
    "Y-AUG": "Y-AUG",
    "Y-SEP": "Y-SEP",
    "Y-OCT": "Y-OCT",
    "Y-NOV": "Y-NOV",
    "Y-DEC": "Y-DEC",
    "Q-JAN": "Q-JAN",
    "Q-FEB": "Q-FEB",
    "Q-MAR": "Q-MAR",
    "Q-APR": "Q-APR",
    "Q-MAY": "Q-MAY",
    "Q-JUN": "Q-JUN",
    "Q-JUL": "Q-JUL",
    "Q-AUG": "Q-AUG",
    "Q-SEP": "Q-SEP",
    "Q-OCT": "Q-OCT",
    "Q-NOV": "Q-NOV",
    "Q-DEC": "Q-DEC",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "W-SUN": "W-SUN",
}


def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    print(hf_dataset.features["timestamp"])
    print(series_fields)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    series_fields.remove("timestamp")
    # dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_freq = pd.infer_freq(hf_dataset[0]["timestamp"])
    dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)
    dataset_freq = "D"

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamp"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    # assert len(gts_dataset) == dataset_length

    return gts_dataset

def get_test_sequences(test_input, test_len):
    test_instance_list = []
    for instance in test_input:
        test_instance = instance["target"][-test_len:]
        test_values = test_instance.astype(float)  
        test_values_tensor = torch.tensor(test_values)
        test_instance_list.append(test_values_tensor)

    return test_instance_list


# def load_and_split_dataset(backtest_config: dict):
#     hf_repo = backtest_config["hf_repo"]
#     dataset_name = backtest_config["name"]
#     offset = backtest_config["offset"]
#     prediction_length = backtest_config["prediction_length"]
#     num_rolls = backtest_config["num_rolls"]
#     max_history = backtest_config["max_history"]
#     distance = backtest_config["distance"]

#     # This is needed because the datasets in autogluon/chronos_datasets_extra cannot
#     # be distribued due to license restrictions and must be generated on the fly
#     trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False

#     ds = datasets.load_dataset(
#         hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
#     )
#     ds.set_format("numpy")

#     gts_dataset = to_gluonts_univariate(ds)
#     train_df, remaining_df = train_test_split(gts_dataset, train_size=0.7, random_state=42)
#     valid_df, test_df = train_test_split(remaining_df, test_size=0.2 / (0.1 + 0.2), random_state=42)

#     # Split dataset for evaluation
#     _, test_template = split(test_df, offset=offset)   
#     test_data = test_template.generate_instances(prediction_length, windows=num_rolls, distance=distance, max_history=max_history)
    
#     return test_data, train_df


# def load_and_split_dataset(backtest_config: dict):
#     hf_repo = backtest_config["hf_repo"]
#     dataset_name = backtest_config["name"]
#     offset = backtest_config["offset"]
#     prediction_length = backtest_config["prediction_length"]
#     num_rolls = backtest_config["num_rolls"]
#     max_history = backtest_config["max_history"]
#     distance = backtest_config["distance"]

#     # Jika dataset berada di direktori lokal, gunakan path lokal
#     if hf_repo.startswith("/"):  # Jika hf_repo adalah path lokal
#         file_path = f"{hf_repo}/{dataset_name}.csv"

#         df = pd.read_csv(file_path)
#         df = pd.DataFrame(df)
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         df = df.sort_values(by='timestamp')
#         data = Dataset.from_pandas(df).train_test_split(test_size=0.2)

#         train_ds=Dataset.from_pandas(pd.DataFrame(data['train']).sort_values(by='timestamp'))
#         test_ds=Dataset.from_pandas(pd.DataFrame(data['test']).sort_values(by='timestamp'))

#         combined_data = {}

#         # Iterate through the columns of train and test, and combine them
#         for column in data['train'].column_names:
#             combined_data[column] = [train_ds[column], test_ds[column]]

#         # Create the combined dataset
#         combined_dataset = Dataset.from_dict(combined_data)
#         combined_dataset
#         combined_dataset.set_format("numpy")  # Pastikan format numpy
#         # Konversi dataset ke format GluonTS
#         gts_dataset = to_gluonts_univariate(combined_dataset)

#     else:
#         # Jika repo adalah nama dataset dari Hugging Face Hub
#         trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False
#         ds = datasets.load_dataset(hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code)
#         ds.set_format("numpy")
#         # Konversi dataset ke format GluonTS
#         gts_dataset = to_gluonts_univariate(ds)
    
#     print(gts_dataset)
#     print("================================================")
    
#     train_df, remaining_df = train_test_split(gts_dataset, train_size=0.7, random_state=42)
#     valid_df, test_df = train_test_split(remaining_df, test_size=0.2 / (0.1 + 0.2), random_state=42)

#     # Split dataset untuk evaluasi
#     _, test_template = split(test_df, offset=offset)
#     test_data = test_template.generate_instances(prediction_length, windows=num_rolls, distance=distance, max_history=max_history)
    
#     return test_data, train_df


def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]
    max_history = backtest_config["max_history"]
    distance = backtest_config["distance"]

    file_path = f"{hf_repo}/{dataset_name}.csv"
    df = pd.read_csv(file_path)

    # Ensure the date column is in datetime format and sort by date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    freq = "D"  # Daily frequency
    dataset = [
        {
            "start": pd.Period(df['timestamp'].iloc[0], freq=freq),
            "target": df["close"].values
        },
        {
            "start": pd.Period(df['timestamp'].iloc[0], freq=freq),
            "target": df["open"].values
        }
    ]

    test = [
        {
            "start": pd.Period(df['timestamp'].iloc[0], freq=freq),
            "target": df["close"].iloc[:3500].values
        },
        {
            "start": pd.Period(df['timestamp'].iloc[0], freq=freq),
            "target": df["open"].iloc[:3500].values
        }
    ]

    data_augment=pd.read_csv("/home/yogi/chronos-research/Retrieval-Augmented-Time-Series-Forecasting/augment.csv")
    augment=[]

    for column in data_augment.columns:
        freq = "D"  # Daily frequency
        augment_data = {
            "start": pd.Period(pd.to_datetime(data_augment[column][0]), freq=freq),
            "target": pd.to_numeric(data_augment[column].iloc[1:].fillna(0).values)
        }
        augment.append(augment_data)

    train_df, test_df = train_test_split(dataset, train_size=0.7, random_state=42, shuffle=False)

    # Split dataset for evaluation
    _, test_template = split(test_df, offset=-prediction_length)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls, distance=distance, max_history=max_history)

    # Split dataset for evaluation
    # train_data, test_data = split(dataset, offset=-prediction_length)
    # test_data = test_data.generate_instances(prediction_length)
    print(test)
    print("=====================================================================")
    return test_data, augment

def generate_sample_forecasts(
    train_df,
    augment: bool,
    top_n: int,
    test_data_input: Iterable,
    pipeline: ChronosPipeline,
    prediction_length: int,
    batch_size: int,
    num_samples: int,
    **predict_kwargs,
):
    # Generate forecast samples
    forecast_samples = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]

        if augment:
            context,mean_std_values = augment_time_series(train_df, pipeline, context, prediction_length, top_n)
        else:
            context,mean_std_values = normalize_context(context)
        
        prediction = pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples,
            limit_prediction_length=False,
            **predict_kwargs,
        )
        prediction = denormalize_predictions(prediction, mean_std_values)
        forecast_samples.append(prediction)
        
    # Convert forecast samples into gluonts SampleForecast objects
    sample_forecasts = []
    forecast_samples = np.concatenate(forecast_samples)

    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts

@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-small",
    device: str = "cuda:1",
    torch_dtype: str = "bfloat16",
    batch_size: int = 30,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    augment: bool = True,
    top_n: int = 1,
):
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    for config in backtest_configs:
  
        pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
        )

        dataset_name = config["name"]
        prediction_length = config["prediction_length"]
        
        logger.info(f"Loading {dataset_name}")
        test_data, train_df = load_and_split_dataset(backtest_config=config)

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        sample_forecasts = generate_sample_forecasts(
            train_df,
            augment, 
            top_n,
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        logger.info(f"Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                sample_forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        result_rows.append(
            {"dataset": dataset_name, "model": chronos_model_id, **metrics[0]}
        )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    results_df.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    app()