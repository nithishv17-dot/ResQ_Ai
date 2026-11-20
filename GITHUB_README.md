# 🦟 ResQ AI: Multimodal Insect Bite & Skin Reaction Classifier

[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![GitHub Release Model](https://img.shields.io/badge/Model%20Release-1.7%20GB-blue?logo=github&logoColor=white)](https://github.com/nithishv17-dot/ResQ_Ai/releases)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas%20Audit-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)

A clinical-grade deep learning decision-support system designed to classify insect bites and dermatological lesions across **8 canonical categories** (`ants`, `bed_bugs`, `chiggers`, `fleas`, `mosquitos`, `no_bites`, `spiders`, `ticks`). ResQ AI combines a **Two-Tier Stacking Ensemble** (MobileNetV2, VGG16, InceptionV3, DenseNet169) with **Bayesian clinical prior fusion**, **conflict-rejection gatekeeping**, and **longitudinal progression tracking**.

---

## 📌 1. Project Overview & Repository Structure

ResQ AI operates in dual modalities via an interactive **Streamlit** interface:
1. **Pure Vision Mode:** Classifies macro lesion imagery with standardized channel-mean calibration ($R=170.88, G=134.77, B=118.84$).
2. **Multimodal Bayesian Context Fusion:** Integrates patient metadata (anatomical bite location, symptom onset timing, and hallmark signs) using rule-based prior distributions, automatically suppressing conflicting text if visual features strongly disagree.

### Repository Layout

```text
ResQ_Ai/
├── app.py                   # Interactive Streamlit web app (vision + multimodal fusion)
├── download_model.py        # Automated stream-downloader for the 1.7 GB GitHub Release model
├── predict.py               # CLI inference script (single file, batch folder, or test_images/)
├── evaluate_test_set.py     # Batch evaluation script for test split benchmarking
├── db_manager.py            # MongoDB Atlas manager for audit logging & longitudinal tracking
├── requirements.txt         # Project dependencies (TensorFlow, Streamlit, requests, tqdm, etc.)
├── .env.example             # Configuration template for MODEL_PATH and MODEL_URL overrides
├── .gitignore               # Strict exclusion rules for models, environments, and caches
├── models/                  # Local model target directory (1.7 GB .keras artifact is gitignored)
└── test_images/             # Sample dermatological lesion images for verification
```

> [!NOTE]
> **Why `models/bug_bite_8class_stacked_model.keras` (1.7 GB) is gitignored:**  
> The trained stacked ensemble artifact is ~1.7 GB, which far exceeds GitHub's 100 MB hard per-file push limit. Rather than bloating git history or requiring complex external LFS configurations, the repository hosts the production model asset directly on **GitHub Releases**. The included `download_model.py` module automatically stream-downloads and verifies the model into `models/` on first execution.

---

## 💻 2. System Requirements

Ensure your host machine meets these requirements before setup:

* **Operating System:** Windows 10/11 (64-bit), macOS (Intel or Apple Silicon), or Linux (Ubuntu 20.04+).
* **Python Version:** **Python 3.10 or 3.11 (64-bit)**.  
  *(Python 3.11 is strongly recommended. Verify that your Python installation is healthy and has standard library modules `tempfile`, `ensurepip`, and `ssl`).*
* **Git:** Git 2.x+ installed.
* **Storage:** At least **3.5 GB of free disk space** (1.7 GB for the `.keras` model archive + TensorFlow and dependencies).
* **Memory (RAM):** **8 GB RAM minimum** (16 GB recommended) to load the multi-backbone stacked ensemble into memory during inference.
* **Network:** High-speed internet connection for initial package installation and one-time 1.7 GB model download.

---

## 🚀 3. Local Laptop Setup (Copy-Paste Commands)

Follow these steps to set up and launch ResQ AI locally:

### Step 1: Clone Repository
```bash
git clone https://github.com/nithishv17-dot/ResQ_Ai.git
cd ResQ_Ai
```

### Step 2: Create & Activate Virtual Environment

* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

* **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```

* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Upgrade Pip & Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the provided `.env.example` file to create your local `.env`:

* **Windows (PowerShell / CMD):**
  ```powershell
  copy .env.example .env
  ```

* **macOS / Linux:**
  ```bash
  cp .env.example .env
  ```

Default values configured in `.env.example`:
```env
MODEL_PATH=models/bug_bite_8class_stacked_model.keras
MODEL_URL=https://github.com/nithishv17-dot/ResQ_Ai/releases/download/v1.0.0/bug_bite_8class_stacked_model.keras
```

### Step 5: Launch Streamlit Web Application
```bash
streamlit run app.py
```

> [!TIP]
> **Zero Manual Model Setup:**  
> On your first launch, `download_model.py` detects that the model is missing from `models/` and automatically downloads `bug_bite_8class_stacked_model.keras` (1.7 GB) with an interactive progress bar. It safely stages the stream inside a `.part` temporary file and verifies the file size before renaming. All subsequent runs detect the healthy model and launch instantly!
>
> If you prefer to pre-download the model before launching Streamlit, run:
> ```bash
> python download_model.py
> ```

Once running, access the local dashboard at:
👉 **`http://localhost:8501`**

---

## 🛠️ 4. Optional CLI Scripts

ResQ AI includes command-line tools for batch prediction and evaluation:

### 1. Standalone Prediction (`predict.py`)
[`predict.py`](predict.py) provides flexible inference options:

* **Automatic Scan (Default):** Evaluates all supported images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) in the local `test_images/` directory:
  ```powershell
  python predict.py
  ```

* **Single Image Prediction:** Specify the path to an individual lesion photograph:
  ```powershell
  python predict.py "test_images/es_spidersimage207 (copy).jpg"
  ```

* **Batch Folder Prediction:** Point to any custom directory of images:
  ```powershell
  python predict.py "C:\path\to\your\lesion_images"
  ```

### 2. Test Set Benchmarking (`evaluate_test_set.py`)
[`evaluate_test_set.py`](evaluate_test_set.py) computes categorical loss, test accuracy, and a per-class scikit-learn `classification_report` (Precision, Recall, F1-Score) across a structured dataset split:
```powershell
python evaluate_test_set.py
```
*(The evaluation directory defaults to `TEST_DIR` configured inside the script).*

---

## 👥 5. GitHub Workflow & Contributing Rules

To maintain a clean repository and protect git history, adhere to the following rules:

### Pulling Updates
Always pull latest updates on the main branch before starting new work:
```bash
git checkout main
git pull origin main
```

### Git Hygiene & Commit Safeguards
* **Never commit large binaries:** Ensure `.keras`, `.part`, `.h5`, or large dataset folders are never staged.
* **Never commit secrets:** Never commit `.env` or personal MongoDB credentials.
* **Pre-commit status verification:**
  ```bash
  git status
  ```
  Ensure untracked files only include intentional code/doc changes. The `.gitignore` file is pre-configured to suppress:
  ```text
  *.keras
  models/*.keras
  *.part
  models/*.part
  venv/
  .venv/
  __pycache__/
  .env
  *.db
  ```

### Publishing a New Model Version
When a new model is trained and ready for release:
1. Navigate to the GitHub repository: **Releases -> Draft a new release**.
2. Create a new tag (e.g., `v1.1.0`) and title.
3. Upload the new `.keras` binary as a release asset attachment.
4. Copy the direct browser download URL for the asset (`https://github.com/nithishv17-dot/ResQ_Ai/releases/download/<tag>/<filename>.keras`).
5. Update `MODEL_URL` in [.env.example](.env.example) and `DEFAULT_MODEL_URL` in [download_model.py](download_model.py).

---

## 🔧 6. Troubleshooting

### 1. Interrupted or Corrupted Model Download
If your internet connection drops during the 1.7 GB model download, `download_model.py` automatically cleans up partial `.part` files. If a partial or corrupted file (`<=100 MB`) remains in `models/`, delete it and trigger the download again:
* **Windows (PowerShell):**
  ```powershell
  Remove-Item -Force models/*.part, models/*.keras -ErrorAction SilentlyContinue
  python download_model.py
  ```
* **macOS / Linux:**
  ```bash
  rm -f models/*.part models/*.keras
  python download_model.py
  ```

### 2. PowerShell Script Execution Policy Error
If activating the virtual environment (`.\venv\Scripts\Activate.ps1`) fails with:
`cannot be loaded because running scripts is disabled on this system`
Run this command in your PowerShell terminal and retry activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Streamlit Port Conflicts
If port `8501` is already in use by another service:
```bash
streamlit run app.py --server.port 8502
```

### 4. MongoDB Atlas Connection Timeout
If you do not have active internet access or MongoDB credentials:
* ResQ AI handles connection failures gracefully. The application will log `[MongoDB Error] Connection failed` and automatically operate in **Local Offline Mode**.
* Core image inference, clinical prior calibration, and first-aid protocols remain 100% operational. Only audit trail persistence and cross-session longitudinal history are bypassed.

---

## 🛡️ Clinical Disclaimer

> [!IMPORTANT]
> **ResQ AI** is an AI-assisted diagnostic decision support prototype developed for clinical research and initial triage screening. It should not replace formal medical evaluation, laboratory serology, or in-person consultation with a board-certified dermatologist or healthcare professional.
