#%%
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Tuple

import torch
import pytest
import pandas as pd

from chronos import ChronosConfig, ChronosPipeline, MeanScaleUniformBins


@pytest.mark.parametrize("n_numerical_tokens", [510])
@pytest.mark.parametrize("n_special_tokens", [2])
def test_tokenizer_consistency(n_numerical_tokens: int, n_special_tokens: int):
    # Load dataset
    df = pd.read_csv('/home/yogi/chronos-research/dataset/LQ45-daily/ANTM.csv')

    # Ensure the date column is in datetime format and sort by date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    df = df.iloc[1165:1177]

    print("timestamp= ",df['timestamp'].values)

    # Extract 'close' column as the target (context)
    context_data = df['close'].values
    context_tensor = torch.tensor(context_data).unsqueeze(0)  # Add batch dimension

    n_tokens = n_numerical_tokens + n_special_tokens

    print("n_tokens",n_tokens)

    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs=dict(low_limit=-1.0, high_limit=1.0),
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=512,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    tokenizer = config.create_tokenizer()
    assert isinstance(tokenizer, MeanScaleUniformBins)

    context = context_tensor  # add batch dimension
    # scale = torch.ones((1,))  # fix the scale to one to turn off scaling
    print("context= ",context)

    token_ids, _, scale = tokenizer._input_transform(context)

    print("token_ids= ",token_ids)

    samples = tokenizer.output_transform(
        token_ids.unsqueeze(1),  # add sample dimension
        scale=scale,
    )

    print("samples= ",samples)

    # assert (samples[0, 0, :] == context).all()


@pytest.mark.xfail
@pytest.mark.parametrize("n_numerical_tokens", [5, 10, 27])
@pytest.mark.parametrize("n_special_tokens", [2, 5, 13])
@pytest.mark.parametrize("use_eos_token", [False, True])
def test_tokenizer_fixed_data(
    n_numerical_tokens: int, n_special_tokens: int, use_eos_token: bool
):
    n_tokens = n_numerical_tokens + n_special_tokens
    context_length = 3

    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs=dict(low_limit=-1.0, high_limit=1.0),
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=use_eos_token,
        model_type="seq2seq",
        context_length=512,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    tokenizer = config.create_tokenizer()

    context = torch.tensor(
        [
            [-3.7, 3.7],
            [-42.0, 42.0],
        ]
    )
    batch_size, _ = context.shape

    token_ids, attention_mask, scale = tokenizer.context_input_transform(context)

    assert token_ids.shape == (batch_size, context_length + 1 * use_eos_token)
    assert all(token_ids[:, 0] == torch.tensor([0]).repeat(batch_size))
    assert all(token_ids[:, 1] == torch.tensor([n_special_tokens]).repeat(batch_size))
    assert all(token_ids[:, 2] == torch.tensor([n_tokens - 1]).repeat(batch_size))

    if use_eos_token:
        assert all(token_ids[:, 3] == torch.tensor([1]).repeat(batch_size))

    samples = tokenizer.output_transform(
        torch.arange(n_special_tokens, n_tokens).unsqueeze(0).repeat(batch_size, 1, 1),
        tokenizer_state=scale,
    )

    assert (samples[:, 0, [0, -1]] == context).all()


@pytest.mark.xfail
@pytest.mark.parametrize("use_eos_token", [False, True])
def test_tokenizer_random_data(use_eos_token: bool):
    # Load dataset
    df = pd.read_csv('/home/yogi/chronos-research/dataset/LQ45-daily/ANTM.csv')

    # Ensure the date column is in datetime format and sort by date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    df = df.head(20)

    # Extract 'close' column as the target (context)
    context_data = df['close'].values

    # Prepare context as tensor (with NaNs if necessary)
    context = torch.tensor(context_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    context_length = 8
    n_tokens = 256
    n_special_tokens = 2

    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs=dict(low_limit=-1.0, high_limit=1.0),
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=use_eos_token,
        model_type="seq2seq",
        context_length=context_length,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    tokenizer = config.create_tokenizer()

    # context = torch.tensor(
    #     [
    #         [torch.nan, torch.nan, 1.0, 1.1, torch.nan, 2.0],
    #         [3.0, torch.nan, 3.9, 4.0, 4.1, 4.9],
    #     ]
    # )

    print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    print("context= ",context)

    token_ids, attention_mask, scale = tokenizer.context_input_transform(context)

    print("token_ids= ",token_ids)
    print("attention_mask= ",attention_mask)

    assert token_ids.shape == (
        *context.shape[:-1],
        context_length + 1 * use_eos_token,
    )
    assert attention_mask.shape == (
        *context.shape[:-1],
        context_length + 1 * use_eos_token,
    )
    assert scale.shape == context.shape[:1]

    sample_ids = torch.randint(low=n_special_tokens, high=n_tokens, size=(1, 10, 4))
    sample_ids[0, 0, 0] = n_special_tokens
    sample_ids[-1, -1, -1] = n_tokens - 1

    samples = tokenizer.output_transform(token_ids, scale)
    print("samples= ",samples)

    assert samples.shape == (2, 10, 4)


def validate_tensor(samples: torch.Tensor, shape: Tuple[int, ...]) -> None:
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == shape


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
def test_pipeline_predict(torch_dtype: str):
    print("=========================================================================================")
    print("test_pipeline_predict")
    print("=========================================================================================")
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        device_map="cpu",
        torch_dtype=torch_dtype,
    )
    context = 10 * torch.rand(size=(4, 16)) + 10
    print("context= ",context)

    # input: tensor of shape (batch_size, context_length)

    samples = pipeline.predict(context, num_samples=12, prediction_length=3)
    validate_tensor(samples, (4, 12, 3))
    print("samples1 (4, 12, 3)= ",samples)

    return

    with pytest.raises(ValueError):
        print("test")
        samples = pipeline.predict(context, num_samples=7, prediction_length=65)
        print("samples1 (4, 7, 65)= ",samples)

    samples = pipeline.predict(
        context, num_samples=7, prediction_length=65, limit_prediction_length=False
    )
    print("samples1 (4, 7, 65)= ",samples)
    validate_tensor(samples, (4, 7, 65))

    return

    # input: batch_size-long list of tensors of shape (context_length,)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("context2= ",list(context))
    samples = pipeline.predict(list(context), num_samples=12, prediction_length=3)
    print("samples2 (4, 12, 3)= ",samples)
    validate_tensor(samples, (4, 12, 3))

    with pytest.raises(ValueError):
        print("test2")
        samples = pipeline.predict(list(context), num_samples=7, prediction_length=65)
        print("samples2 (4, 7, 65)= ",samples)

    samples = pipeline.predict(
        list(context),
        num_samples=7,
        prediction_length=65,
        limit_prediction_length=False,
    )
    print("samples2 (4, 7, 65)= ",samples)
    validate_tensor(samples, (4, 7, 65))

    # input: tensor of shape (context_length,)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("context3= ",context[0, ...])
    samples = pipeline.predict(context[0, ...], num_samples=12, prediction_length=3)
    print("samples3 (1, 12, 3)= ",samples)
    validate_tensor(samples, (1, 12, 3))

    with pytest.raises(ValueError):
        print("test3")
        samples = pipeline.predict(context[0, ...], num_samples=7, prediction_length=65)
        print("samples3 (1, 7, 65)= ",samples)

    samples = pipeline.predict(
        context[0, ...],
        num_samples=7,
        prediction_length=65,
        limit_prediction_length=False,
    )
    print("samples3 (1, 7, 65)= ",samples)
    validate_tensor(samples, (1, 7, 65))


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
def test_pipeline_embed(torch_dtype: str):
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        device_map="cpu",
        torch_dtype=torch_dtype,
    )
    d_model = pipeline.model.model.config.d_model
    context = 10 * torch.rand(size=(4, 16)) + 10
    expected_embed_length = 16 + (1 if pipeline.model.config.use_eos_token else 0)

    # input: tensor of shape (batch_size, context_length)

    embedding, scale = pipeline.embed(context)
    validate_tensor(embedding, (4, expected_embed_length, d_model))
    validate_tensor(scale, (4,))

    # input: batch_size-long list of tensors of shape (context_length,)

    embedding, scale = pipeline.embed(list(context))
    validate_tensor(embedding, (4, expected_embed_length, d_model))
    validate_tensor(scale, (4,))

    # input: tensor of shape (context_length,)
    embedding, scale = pipeline.embed(context[0, ...])
    validate_tensor(embedding, (1, expected_embed_length, d_model))
    validate_tensor(scale, (1,))

# %%
