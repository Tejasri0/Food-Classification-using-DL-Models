
import os
import shutil

from huggingface_hub import hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Hugging Face private repository
REPO_ID = os.environ["HF_MODEL_REPO"]
HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_FILES = [
    "custom_cnn.keras",
    "resnet50.keras",
    "vgg16.keras"
]

for filename in MODEL_FILES:

    model_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(model_path):
        print(f"Already exists: {filename}")
        continue

    print(f"Downloading {filename}...")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        token=HF_TOKEN
    )

    shutil.copy2(downloaded_path, model_path)

    print(f"Downloaded successfully: {filename}")

# Verify all models
for filename in MODEL_FILES:
    path = os.path.join(MODELS_DIR, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model missing: {filename}"
        )

print("\nAll three models are ready!")
