.PHONY: train-prepare test-prepare bow-prepare fasttext-prepare

data/interim/train.parquet: run.py data/raw/train.csv
	python run.py preprocess-train
train-prepare: data/interim/train.parquet

data/interim/test.parquet: run.py data/raw/test_private_expanded.csv
	python run.py preprocess-test
test-prepare: data/interim/test.parquet

data/interim/fasttext_train.txt data/interim/fasttext_valid.txt: run.py data/interim/train.parquet
	python run.py fasttext-prepare
fasttext-prepare: data/interim/fasttext_train.txt data/interim/fasttext_valid.txt