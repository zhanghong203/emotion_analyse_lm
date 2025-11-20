from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

LOG_DIR = ROOT_DIR / "logs"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "process"
MODELS_DIR = ROOT_DIR / "models"
SEQ_lEN = 128
BATCH_SIZE = 64
EPOCHS = 10

EMBEDDING_DIM = 128
PADDING_IDX = 0
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
