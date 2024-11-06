from transformers import T5Tokenizer, T5Model

# Memuat tokenizer dan model
model = T5Model.from_pretrained("/home/yogi/chronos-research/chronos-forecasting/test/dummy-chronos-model")

# Mengakses embedding layer dari model
embedding_layer = model.get_input_embeddings()

# Mendapatkan bobot embedding
embedding_weights = embedding_layer.weight.data.numpy()

# Cetak ukuran embedding weights
print("Ukuran Tabel Embedding:", embedding_weights)
print("Ukuran Tabel Embedding:", embedding_weights.shape)

# Tampilkan embedding untuk beberapa token pertama
print("Embedding untuk token pertama (token ID 0):", embedding_weights[510])

