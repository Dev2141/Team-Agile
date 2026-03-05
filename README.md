# Ollama Threat Analyser — Standalone

Completely self-contained VLM threat-detection tool.
Zero dependency on the DVR app.

## What it does

- Send images or a video → Ollama Qwen-VL → structured threat verdict
- Web UI at `http://localhost:8100`

## Setup

```powershell
cd ollama_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
uvicorn main:app --host 0.0.0.0 --port 8100 --reload
```

Open `http://localhost:8100`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3-vl:235b-cloud` | Model to use |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Ollama reachability check |
| `POST` | `/analyse/images` | Send 1–6 JPEG/PNG images |
| `POST` | `/analyse/video` | Upload a video file |

### POST /analyse/images

Form fields:
- `images` — 1 to 6 image files (JPEG/PNG)
- `context` — optional text hint (e.g. `"person near bicycle"`)

### POST /analyse/video

Form fields:
- `file` — video file (MP4/MOV/AVI/MKV)
- `fps` — frames per second to sample (default `2`)
- `max_keyframes` — max frames sent to VLM (default `6`)
- `context` — optional text hint

### Response (both endpoints)

```json
{
  "verdict":    "THREAT" or "BENIGN",
  "type":       "theft|assault|vandalism|trespass|intrusion|none",
  "confidence": 0.87,
  "reason":     "Person crouching near bicycle, looking around",
  "is_threat":  true
}
```
