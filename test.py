from pathlib import Path
from typing import Tuple

import torch
import pytest
import pandas as pd

from chronos import ChronosConfig, ChronosPipeline, MeanScaleUniformBins

torch_dtype = torch.bfloat16

print("=========================================================================================")
print("test_pipeline_predict")
print("=========================================================================================")
pipeline = ChronosPipeline.from_pretrained(
    "/home/yogi/chronos-research/chronos-forecasting/test/dummy-chronos-model",
    device_map="cpu",
    torch_dtype=torch_dtype,
)
context = 10 * torch.rand(size=(4, 16)) + 10
print("context= ",context)

# input: tensor of shape (batch_size, context_length)

samples = pipeline.predict(context, num_samples=12, prediction_length=3)

print("samples1 (4, 12, 3)= ",samples)