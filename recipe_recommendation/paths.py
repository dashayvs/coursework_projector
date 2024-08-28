from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

FILTER_DATA_PATH = ROOT_DIR / "data" / "filter_data_recipes.csv"
BEST_MODEL_PATH = ROOT_DIR / "model" / "ObjectsTextSimilarityModel.npy"
RECIPES_PATH = ROOT_DIR / "data" / "train_data_text_url.csv"
RAW_RECIPES_PATH = ROOT_DIR / "data" / "data_raw_recipes.csv"
