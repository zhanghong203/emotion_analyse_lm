from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

import config


class ReviewAnalyzeDataset(Dataset):
	def __init__(self, path):
		self.data = pd.read_json(path, orient='records', lines=True).to_dict('records')

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
		output_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float32)
		return input_tensor, output_tensor


def get_dataloader(train=True):
	path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
	dataset = ReviewAnalyzeDataset(path)
	return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
	train_loader = get_dataloader()
	test_loader = get_dataloader(train=False)
	print(len(train_loader))
	print(len(test_loader))
	for input_tensor, output_tensor in train_loader:
		print(input_tensor.shape, output_tensor.shape)
		# input shape: (batch_size, seq_len) output shape: (batch_size)
		break


