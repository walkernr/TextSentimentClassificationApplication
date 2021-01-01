import torch
import torch.nn as nn
import torch.nn.functional as F


class text_classifier_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)
        self.fc = nn.Linear(self.embedding_dim, self.output_dim)


    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)
