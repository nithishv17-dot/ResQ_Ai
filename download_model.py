import os
import sys
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

DEFAULT_MODEL_FILENAME = "bug_bite_8class_stacked_model.keras"
DEFAULT_MODEL_PATH = os.path.join("models", DEFAULT_MODEL_FILENAME)
DEFAULT_MODEL_URL = "https://github.com/nithishv17-dot/ResQ_Ai/releases/download/v1.0.0/bug_bite_8class_stacked_model.keras"


def ensure_model(model_path: str = None, model_url: str = None) -> str:
    """
    Checks if the model file exists locally and is valid (>100 MB).
    If missing or corrupt/incomplete, stream-downloads the 1.7 GB model from GitHub Releases
    with a tqdm progress bar directly into a .part file, renaming upon completion.
    """
    model_path = model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    model_url = model_url or os.getenv("MODEL_URL", DEFAULT_MODEL_URL)

    # 1. Check if model already exists and is >100 MB
    min_size_bytes = 100 * 1024 * 1024  # 100 MB threshold
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        if file_size > min_size_bytes:
            return model_path
        print(f"[Warning] Found '{model_path}' but size ({file_size / (1024 * 1024):.2f} MB) is <= 100 MB. Re-downloading...")

    # 2. Ensure models/ directory exists
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    part_path = f"{model_path}.part"
    print("\n" + "=" * 60)
    print("📦 Model file not found locally or incomplete.")
    print(f"📁 Target Path : {model_path}")
    print(f"🌐 Downloading from GitHub Releases:")
    print(f"🔗 {model_url}")
    print("=" * 60 + "\n")

    try:
        response = requests.get(model_url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(part_path, "wb") as f, tqdm(
            desc=os.path.basename(model_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        # 3. Verify downloaded file size before atomically replacing
        if os.path.exists(part_path) and os.path.getsize(part_path) > min_size_bytes:
            if os.path.exists(model_path):
                os.remove(model_path)
            os.replace(part_path, model_path)
            print(f"\n✅ Model downloaded and verified successfully: {model_path}\n")
            return model_path
        else:
            downloaded_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            raise RuntimeError(
                f"Downloaded model size ({downloaded_size / (1024 * 1024):.2f} MB) is smaller than required (>100 MB)."
            )

    except Exception as e:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        print(f"\n❌ Error during model download: {e}")
        raise RuntimeError(
            f"Unable to download model from {model_url}. "
            f"Please verify your network connection or manually place the model at '{model_path}'."
        ) from e


if __name__ == "__main__":
    ensure_model()
