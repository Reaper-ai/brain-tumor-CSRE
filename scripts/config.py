from pathlib import Path
import os

# Get the project root directory (assuming config.py is in the project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Define paths for model weights
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Create weights directory if it doesn't exist
WEIGHTS_DIR.mkdir(exist_ok=True)

# Define model weight paths
MODEL_PATHS = {
    "classifier": WEIGHTS_DIR / "brain_tumor_classifier.pth",
    "t1": WEIGHTS_DIR / "t1_unet.pth",
    "t2": WEIGHTS_DIR / "t2_unet.pth",
    "flair": WEIGHTS_DIR / "flair_unet.pth",
    "t1ce": WEIGHTS_DIR / "t1ce_unet.pth"
}

# Validate paths exist
def validate_model_paths():
    missing_models = []
    for model_name, path in MODEL_PATHS.items():
        if not path.exists():
            missing_models.append(f"{model_name} ({path})")

    if missing_models:
        raise FileNotFoundError(
            f"Missing model weights for: {', '.join(missing_models)}\n"
            f"Please ensure model weights are present in {WEIGHTS_DIR}"
        )