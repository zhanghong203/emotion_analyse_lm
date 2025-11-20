import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import JiebaTokenizer


def process():
	print("开始处理数据")
	# 1. 读取文件
	df = pd.read_csv(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv', usecols=['label', 'review'],
					 encoding='utf-8').dropna()
	# 2. 划分数据集 (分层抽样)
	train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
	# 3. 构建词表
	JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.MODELS_DIR / 'vocab.txt')
	# 创建tokenizer对象
	tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
	# 计算序列长度
	# max_tokens = train_df['review'].apply(lambda x: len(tokenizer.tokenize(x))).quantile(0.95)
	# 4. 构建训练集
	train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_lEN))
	# 5. 保存训练集
	train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)
	test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_lEN))
	# 5. 保存训练集
	test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)
	print("数据处理完成")


if __name__ == '__main__':
	process()
