from pathlib import Path
from typing import Tuple

import torch
import pytest
import pandas as pd

from chronos import ChronosConfig, ChronosPipeline, MeanScaleUniformBins
import numpy as np

torch_dtype = torch.bfloat16

# print("=========================================================================================")
# print("test_pipeline_predict")
# print("=========================================================================================")
# pipeline = ChronosPipeline.from_pretrained(
#     "/home/yogi/chronos-research/chronos-forecasting/test/dummy-chronos-model",
#     device_map="cpu",
#     torch_dtype=torch_dtype,
# )
# context = 10 * torch.rand(size=(4, 16)) + 10
# print("context= ",context)

# # input: tensor of shape (batch_size, context_length)

# samples = pipeline.predict(context, num_samples=12, prediction_length=3)

# print("samples1 (4, 12, 3)= ",samples)








# EMBED ====================================================================================

pipeline = ChronosPipeline.from_pretrained(
    "/home/yogi/chronos-research/chronos-forecasting/test/dummy-chronos-model",
    device_map="cpu",
    torch_dtype=torch_dtype,
)
d_model = pipeline.model.model.config.d_model
# context = 10 * torch.rand(size=(4, 16)) + 10
# expected_embed_length = 16 + (1 if pipeline.model.config.use_eos_token else 0)



# Load dataset
df = pd.read_csv('/home/yogi/chronos-research/dataset/LQ45-daily/ANTM.csv')

# Ensure the date column is in datetime format and sort by date
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')
df = df.iloc[1165:1177]

# Extract 'close' column as the target (context)
context_data = df['close'].values
context = torch.tensor(context_data).unsqueeze(0)  # Add batch dimension



print("context= ",context)

# input: tensor of shape (batch_size, context_length)

embedding, scale = pipeline.embed(context)

print("embedding1= ",embedding)

num_tokens = 3  # Misalnya, kita akan menggunakan 5 token
embedding_dim = 3  # Ukuran embedding
embedding_table = np.random.rand(num_tokens, embedding_dim)
print("Test= ",embedding_table)
# # input: batch_size-long list of tensors of shape (context_length,)

# embedding, scale = pipeline.embed(list(context))
# print("embedding2 (batch_size-long list)= ",embedding)

# # input: tensor of shape (context_length,)
# embedding, scale = pipeline.embed(context[0, ...])
# print("embedding3= ",embedding)