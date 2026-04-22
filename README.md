# Emotion Pro

Real-time facial emotion detection web app built with Flask, Socket.IO, and ONNX Runtime.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Live Detection** — Real-time webcam emotion recognition via a mini Xception model trained on FER-2013
- **Session Tracking** — Every detection run is stored with a unique session ID
- **Analytics Dashboard** — Emotion distribution charts and 7-day activity graph
- **Session History** — Browse past sessions with per-session timeline and breakdown
- **Alert System** — Triggers when negative emotions (Angry, Sad, Fear, Disgust) persist beyond a threshold
- **Detection Settings** — Tune scale factor, min neighbors, and frame rate on the fly
- **CSV Export** — Download all your emotion data as a spreadsheet

## Emotions Detected

`Angry` · `Disgust` · `Fear` · `Happy` · `Sad` · `Surprise` · `Neutral`

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask + Flask-SocketIO (eventlet) |
| ML Inference | ONNX Runtime — mini Xception (FER-2013) |
| Face Detection | OpenCV Haar Cascade |
| Database | SQLite via Flask-SQLAlchemy |
| Frontend | Vanilla JS + Chart.js |
| Deployment | Docker |

## Project Structure

```
emotion_pro/
├── main.py                        # Flask app, routes, Socket.IO handlers
├── fer2013_mini_XCEPTION.onnx     # Pre-trained emotion model
├── requirements.txt
├── Dockerfile
├── static/
│   └── style.css
└── templates/
    ├── detector.html              # Live detection page
    ├── dashboard.html             # Analytics dashboard
    └── history.html               # Session history
```

## Getting Started

### Prerequisites

- Python 3.10+
- A webcam

### Run Locally

```bash
git clone https://github.com/your-username/emotion-pro.git
cd emotion-pro

pip install -r requirements.txt

python main.py
# Open http://localhost:5002
```

### Run with Docker

```bash
docker build -t emotion-pro .
docker run -p 5002:5002 emotion-pro
# Open http://localhost:5002
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `emotion-pro-2024` | Flask session secret |
| `PORT` | `5002` | Port to run the server on |

## License

MIT
