import jieba
from tqdm import tqdm

import config


class JiebaTokenizer:

	unk_token = '<UNK>'
	pad_token = '<PAD>'

	def __init__(self, vocab_list):
		self.vocab_list = vocab_list
		self.vocab_size = len(vocab_list)
		self.word2index = {word: index for index, word in enumerate(vocab_list)}
		self.index2word = {index: word for index, word in enumerate(vocab_list)}
		self.unk_token_index = self.word2index[self.unk_token]
		self.pad_token_index = self.word2index[self.pad_token]

	@staticmethod
	def tokenize(text):
		return jieba.lcut(text)

	def encode(self, text, seq_len):
		tokens = self.tokenize(text)
		indexes = [self.word2index.get(token, self.unk_token_index) for token in tokens]
		# 填充或截取
		tokens = tokens[:seq_len] if len(tokens) >= seq_len else tokens + [self.pad_token] * (seq_len - len(indexes))
		return [self.word2index.get(token, self.unk_token_index) for token in tokens]

	@classmethod
	def build_vocab(cls, setences, vocab_path):
		"""
		根据路径创建词表
		:param setences: 语料库
		:param vocab_path: 词表路径
		"""
		vocab_set = set()
		for sentence in tqdm(setences, desc="构建词表"):
			vocab_set.update(jieba.lcut(sentence))
		vocab_set = [cls.pad_token, cls.unk_token] + [token for token in vocab_set if token.strip() != '']
		print("词表大小:", len(vocab_set))
		# 保存词表
		with open(vocab_path, 'w', encoding='utf-8') as f:
			f.write('\n'.join(vocab_set))

	@classmethod
	def from_vocab(cls, vocab_path):
		with open(vocab_path, 'r', encoding='utf-8') as f:
			vocab_list = [line.strip() for line in f.readlines()]
		return cls(vocab_list)


if __name__ == '__main__':
	tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / 'vocab.txt')
	print(f'词表大小：{tokenizer.vocab_size}')
	print(tokenizer.unk_token)

