# 🔬 MedVision AI
### Intelligent Medical Scan Analysis Platform

MedVision AI is an AI-powered medical imaging assistant built with Streamlit and Roboflow Inference. It accepts medical scan images (MRI, CT, X-Ray, Retinal) and returns annotated results with a structured clinical report — including per-finding severity levels, confidence scores, and clinical recommendations.

---

## Features

- **Multi-modality support** — Brain MRI, Chest X-Ray, CT Scan, Bone X-Ray, Retinal Scan
- **AI-powered detection** — Roboflow-hosted object detection models
- **Structured report generation** — Findings, severity (HIGH / MEDIUM / LOW / NONE), confidence scores, clinical notes
- **Annotated scan overlay** — Colour-coded bounding boxes by severity
- **Patient ID tracking** — Reference-based session logging
- **Downloadable reports** — Export findings as JSON
- **Session logging** — All analyses are appended to `medvision_log.txt`

---

## Project Structure

```
MedVision-AI/
├── main.py              # Core application
├── requirements.txt     # Python dependencies
├── medvision_log.txt    # Auto-generated session log (gitignored)
└── README.md
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/medvision-ai.git
cd medvision-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Roboflow API key

**Option A — Streamlit Secrets (recommended for deployment)**

Create `.streamlit/secrets.toml`:
```toml
ROBOFLOW_API_KEY = "your_api_key_here"
```

**Option B — Environment Variable**
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

### 4. Run the app
```bash
streamlit run main.py
```

---

## Model Configuration

Each scan type maps to a Roboflow model ID in `SCAN_MODELS` inside `main.py`. Replace these with your own trained model IDs:

```python
SCAN_MODELS = {
    "Brain MRI":        "your-brain-mri-model/1",
    "Chest X-Ray":      "your-chest-xray-model/1",
    "CT Scan (General)":"your-ct-scan-model/1",
    "Bone X-Ray":       "your-bone-xray-model/1",
    "Retinal Scan":     "your-retinal-model/1",
}
```

You can train and host these models for free at [roboflow.com](https://roboflow.com).

---

## Deployment

### Streamlit Community Cloud (free)
1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and add `ROBOFLOW_API_KEY` in the Secrets panel

---

## Clinical Disclaimer

> MedVision AI is an **assistive tool** designed to support — not replace — professional medical judgement. All findings must be reviewed and validated by a qualified radiologist or clinician before any clinical decisions are made. This tool is **not FDA-approved** for diagnostic use.

---

## Built With

| Library | Purpose |
|---|---|
| Streamlit | Web UI framework |
| Pillow | Image processing & annotation |
| Roboflow Inference SDK | AI model inference |
| NumPy | Array & image operations |

---

## Inspired By

This project was inspired by [CircuitSense AI](https://github.com/your-username/circuit-sense) — a computer vision tool for electronics component detection and circuit fault analysis — and applies the same pattern to the medical imaging domain.

---

*College project — for educational and demonstration purposes only.*
