from torch import nn
import torch

import config


class ReviewAnalyzeModel(nn.Module):

	def __init__(self, vocab_size, padding_idx):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_idx)
		self.lstm = nn.LSTM(config.EMBEDDING_DIM, config.HIDDEN_SIZE, batch_first=True)
		self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1)

	def forward(self, x: torch.Tensor):
		# x.shape (batch_size, seq_len)
		embed = self.embedding(x)
		# embedding.shape (batch_size, seq_len, embedding_dim)
		output, (_, _) = self.lstm(embed)
		# output.shape (batch_size, seq_len, hidden_size)
		# 获取每个样本真实的最后一个token的隐藏状态
		batch_indexes = torch.arange(0, output.shape[0])
		seq_indexes = (x != self.embedding.padding_idx).sum(dim=1)

		# last_hidden.shape (batch_size, hidden_size)
		last_hidden = output[batch_indexes, seq_indexes - 1]
		output = self.linear(last_hidden).squeeze(1)
		# output.shape (batch_size)
		return output
