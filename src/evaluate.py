import torch
import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch

from tokenizer import JiebaTokenizer


def evaluate(model, test_loader, device):
	total_count = 0
	correct_count = 0
	for inputs, targets in test_loader:
		inputs = inputs.to(device)
		# input.shape: (batch_size, seq_len)
		targets = targets.tolist()
		# target.shape (batch_size) e.g: [0, 1, 1]
		batch_result = predict_batch(model, inputs)  # e.g: [0.2, 0.3, 0.4]
		for result, target in zip(batch_result, targets):
			result = 1 if result >= 0.5 else 0
			if result == target:
				correct_count += 1
			total_count += 1
	# 计算准确度
	return correct_count / total_count


def run_evaluate():
	# 1. 确定设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# 2. 加载词表
	tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
	# 3. 加载模型
	model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_index).to(device)
	model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
	# 4. 数据集
	test_loader = get_dataloader(train=False)
	# 5. 评估
	acc = evaluate(model, test_loader, device)
	print(f"准确率: {acc}")


if __name__ == '__main__':
	run_evaluate()
