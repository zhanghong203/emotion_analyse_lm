import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel
import config


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
	total_loss = 0
	model.train()
	for inputs, targets in dataloader:
		inputs = inputs.to(device)  # inputs.shape (batch_size, seq_len)
		targets = targets.to(device)  # targets.shape (batch_size)
		outputs = model(inputs)
		# outputs.shape (batch_size)
		loss = loss_fn(outputs, targets)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		total_loss += loss.item()
	return total_loss / len(dataloader)


def train():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dataloader = get_dataloader()
	tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
	model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
	loss_fn = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
	writer = SummaryWriter(config.LOG_DIR / time.strftime('%Y-%m-%d-%H-%M-%S'))
	# train
	best_loss = float('inf')
	for epoch in tqdm(range(config.EPOCHS), desc='训练'):
		print(f'Epoch: {epoch + 1}')
		loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
		print(f'Loss: {loss:.4f}')
		# 记录到tensorboard
		writer.add_scalar('Loss', loss, epoch)
		# 保存模型
		if loss < best_loss:
			best_loss = loss
			torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
			print("保存模型")
	writer.close()


if __name__ == '__main__':
	train()
