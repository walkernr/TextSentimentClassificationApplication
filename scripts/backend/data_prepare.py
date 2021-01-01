import random
import torch
from torchtext import data, datasets


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def preprocess_data(data_path, vector_path, text_path, label_path, seed):
    text = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
    label = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(text_field=text, label_field=label, root=data_path)
    train_data, valid_data = train_data.split(random_state=random.seed(seed))
    max_vocab_size = 25_000
    text.build_vocab(train_data, max_size=max_vocab_size, vectors='glove.6B.100d', vectors_cache=vector_path, unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data)
    torch.save(text, text_path)
    torch.save(label, label_path)
    return train_data, valid_data, test_data


def load_text_vocab(text_path):
    return torch.load(text_path)


def load_label_vocab(label_path):
    return torch.load(label_path)