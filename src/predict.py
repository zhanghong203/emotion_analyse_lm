import torch
import config
from model import ReviewAnalyzeModel

from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
	"""
	批量预测
	:param model: 模型
	:param inputs: 输入 形状：(batch_size, seq_len)
	:return: 预测结果 形状： (batch_size)
	"""
	model.eval()
	with torch.no_grad():
		output = model(inputs)
		# output.shape (batch_size)
	batch_result = torch.sigmoid(output)
	return batch_result.tolist()


def predict(text, tokenizer, model, device):
	# 1. 处理输入
	indexes = tokenizer.encode(text, seq_len=config.SEQ_lEN)
	# input.shape (batch_size, seq_len)
	input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)
	# 2. 预测逻辑
	batch_result = predict_batch(model, input_tensor)
	return batch_result[0]


def run_predict():
	# 1. 确定设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# 2. 加载词表
	tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
	# 3. 加载模型
	model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_index).to(device)
	model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
	print("欢迎使用情感分析模型（输入q或者quit退出）")
	while True:
		user_input = input(">")
		if user_input in ['q', 'quit']:
			print("bye！")
			break
		if user_input.strip() == '':
			print("请输入内容")
			continue
		result = predict(user_input, tokenizer, model, device)
		if result > 0.5:
			print(f"正向(置信度){result}")
		else:
			print(f"负向评价{1 - result}")


if __name__ == '__main__':
	run_predict()

