import base64
import csv
import io
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit, join_room
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = os.environ.get("SECRET_KEY", "emotion-pro-2024")

# Writable DB path: Works on Windows and Linux
_here = os.path.dirname(os.path.abspath(__file__))
_db_dir = os.path.join(_here, "instance")
os.makedirs(_db_dir, exist_ok=True)
_db_path = os.path.join(_db_dir, "emotion_pro.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{_db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── ML Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_here, "fer2013_mini_XCEPTION.onnx")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(_here, "..", "fer2013_mini_XCEPTION.onnx")

ort_session = ort.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

ALERT_RULES = {"Angry": 15, "Sad": 20, "Fear": 15, "Disgust": 20}

# ── In-memory state per socket session ───────────────────────────────────────
session_state: dict = defaultdict(lambda: {
    "smoother": deque(maxlen=6),
    "last_emotion": None,
    "emotion_start": time.time(),
    "alert_fired": False,
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "frame_interval": 50,
})

# ── DB Models (no user accounts — keyed by browser_id) ───────────────────────
class DetectionSession(db.Model):
    id          = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    browser_id  = db.Column(db.String(36), nullable=False, index=True)
    started_at  = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at    = db.Column(db.DateTime, nullable=True)
    total_frames= db.Column(db.Integer, default=0)
    logs        = db.relationship("EmotionLog", backref="det_session", lazy=True)


class EmotionLog(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    session_id  = db.Column(db.String(36), db.ForeignKey("detection_session.id"), nullable=False)
    emotion     = db.Column(db.String(20), nullable=False)
    confidence  = db.Column(db.Float, nullable=False)
    faces_count = db.Column(db.Integer, default=0)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)


class AlertLog(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    browser_id       = db.Column(db.String(36), nullable=False, index=True)
    session_id       = db.Column(db.String(36), nullable=False)
    emotion          = db.Column(db.String(20), nullable=False)
    duration_seconds = db.Column(db.Float, nullable=False)
    triggered_at     = db.Column(db.DateTime, default=datetime.utcnow)


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("detector.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
def history():
    return render_template("history.html")


# ── REST API ──────────────────────────────────────────────────────────────────


# Nayi emotion detection session start karta hai.
@app.route("/api/session/start", methods=["POST"])
def api_session_start():
    browser_id = request.json.get("browser_id", "unknown")
    s = DetectionSession(browser_id=browser_id)
    db.session.add(s)
    db.session.commit()
    return jsonify({"session_id": s.id})

# Current session ko stop karta hai.
@app.route("/api/session/end", methods=["POST"])
def api_session_end():
    
    sid = request.json.get("session_id")
    if sid:
        s = DetectionSession.query.get(sid)
        if s:
            s.ended_at = datetime.utcnow()
            db.session.commit()
        session_state.pop(sid, None)
    return jsonify({"status": "ended"})


# Dashboard ke liye overall analytics bhejti hai. Isme total sessions, detections, alerts, emotion distribution, aur last 7 din ka trend hota hai.
@app.route("/api/stats/overview")
def api_stats_overview():
    
    browser_id = request.args.get("browser_id", "")
    sessions = DetectionSession.query.filter_by(browser_id=browser_id).all()
    sids = [s.id for s in sessions]

    logs = EmotionLog.query.filter(EmotionLog.session_id.in_(sids)).all()
    dist = defaultdict(int)
    for l in logs:
        dist[l.emotion] += 1

    total_alerts = AlertLog.query.filter_by(browser_id=browser_id).count()

    seven_days = []
    for i in range(6, -1, -1):
        day = datetime.utcnow() - timedelta(days=i)
        ds = day.replace(hour=0, minute=0, second=0, microsecond=0)
        de = ds + timedelta(days=1)
        count = EmotionLog.query.filter(
            EmotionLog.session_id.in_(sids),
            EmotionLog.timestamp >= ds,
            EmotionLog.timestamp < de
        ).count()
        seven_days.append({"date": ds.strftime("%b %d"), "count": count})

    return jsonify({
        "total_sessions": len(sessions),
        "total_detections": len(logs),
        "total_alerts": total_alerts,
        "emotion_distribution": dict(dist),
        "last_7_days": seven_days,
    })


# Purani sessions ki history deta hai. Isme pagination hoti hai aur har session ke liye dominant emotion, duration, aur total frames hota hai.
@app.route("/api/history/sessions")
def api_history_sessions():
    browser_id = request.args.get("browser_id", "")
    page = request.args.get("page", 1, type=int)
    per_page = 10
    q = DetectionSession.query.filter_by(browser_id=browser_id)\
        .order_by(DetectionSession.started_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    result = []
    for s in q.items:
        logs = EmotionLog.query.filter_by(session_id=s.id).all()
        dist = defaultdict(int)
        for l in logs:
            dist[l.emotion] += 1
        dominant = max(dist, key=dist.get) if dist else "—"
        duration = int((s.ended_at - s.started_at).total_seconds()) if s.ended_at else None
        result.append({
            "id": s.id,
            "started_at": s.started_at.strftime("%Y-%m-%d %H:%M"),
            "ended_at": s.ended_at.strftime("%H:%M") if s.ended_at else "Active",
            "duration": duration,
            "total_frames": s.total_frames,
            "dominant_emotion": dominant,
        })
    return jsonify({"sessions": result, "total": q.total, "pages": q.pages})

# Ek specific session ke detailed logs deta hai. Isme har detection ka timestamp, emotion, confidence, aur face count hota hai. Saath hi emotion distribution bhi hota hai.
@app.route("/api/history/session/<session_id>")
def api_session_detail(session_id):
    logs = EmotionLog.query.filter_by(session_id=session_id)\
        .order_by(EmotionLog.timestamp.asc()).all()
    dist = defaultdict(int)
    for l in logs:
        dist[l.emotion] += 1
    timeline = [{"time": l.timestamp.strftime("%H:%M:%S"),
                 "emotion": l.emotion,
                 "confidence": round(l.confidence * 100, 1)} for l in logs]
    return jsonify({"timeline": timeline, "distribution": dict(dist)})

# Recent alerts ki list deta hai. Isme har alert ke liye emotion, duration, triggered time, aur session ID hota hai.
@app.route("/api/alerts")
def api_alerts():
    browser_id = request.args.get("browser_id", "")
    alerts = AlertLog.query.filter_by(browser_id=browser_id)\
        .order_by(AlertLog.triggered_at.desc()).limit(20).all()
    return jsonify([{
        "emotion": a.emotion,
        "duration": round(a.duration_seconds),
        "triggered_at": a.triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": a.session_id,
    } for a in alerts])


# Emotion detection data ko CSV format me export karta hai. Isme timestamp, session ID, emotion, confidence, aur face count hota hai.
@app.route("/api/export/csv")
def api_export_csv():
    browser_id = request.args.get("browser_id", "")
    sids = [s.id for s in DetectionSession.query.filter_by(browser_id=browser_id).all()]
    logs = EmotionLog.query.filter(EmotionLog.session_id.in_(sids))\
        .order_by(EmotionLog.timestamp.asc()).all()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["timestamp", "session_id", "emotion", "confidence", "faces_count"])
    for l in logs:
        w.writerow([l.timestamp.strftime("%Y-%m-%d %H:%M:%S"), l.session_id,
                    l.emotion, round(l.confidence * 100, 2), l.faces_count])
    out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=emotion_data.csv"})


# Detection parameters ko get ya update karta hai. Isme scale factor, min neighbors, aur frame interval hota hai.
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    sid = request.json.get("session_id", "default") if request.method == "POST" else \
          request.args.get("session_id", "default")
    state = session_state[sid]
    if request.method == "POST":
        data = request.get_json()
        state["scale_factor"]  = float(data.get("scale_factor", 1.1))
        state["min_neighbors"] = int(data.get("min_neighbors", 5))
        state["frame_interval"]= int(data.get("frame_interval", 50))
        return jsonify({"status": "updated"})
    return jsonify({k: state[k] for k in ("scale_factor", "min_neighbors", "frame_interval")})


# ── Socket.IO ─────────────────────────────────────────────────────────────────
@socketio.on("join_session")
def on_join(data):
    sid = data.get("session_id")
    if sid:
        join_room(sid)


@socketio.on("process_image")
def handle_image(data):
    try:
        det_session_id = data.get("session_id")
        image_data     = data.get("image")
        browser_id     = data.get("browser_id", "unknown")
        if not det_session_id or not image_data:
            return

        state = session_state[det_session_id]

        _, encoded = image_data.split(",", 1)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=state["scale_factor"],
            minNeighbors=state["min_neighbors"], minSize=(30, 30)
        )

        response_data = {
            "faces": [],
            "stats": {"emotion": "Scanning...", "confidence": 0.0, "faces_count": len(faces)},
            "alert": None,
            "frame_interval": state["frame_interval"],
        }

        max_conf = 0.0
        dominant = "No Face Detected" if len(faces) == 0 else "Scanning..."

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            emotion_str, conf_val = "Unknown", 0.0
            if roi.shape[0] >= 48 and roi.shape[1] >= 48:
                roi  = cv2.equalizeHist(roi)
                face = cv2.resize(roi, (48, 48)) / 255.0
                flip = cv2.flip(face, 1)
                batch = np.vstack([np.reshape(face, (1,48,48,1)),
                                   np.reshape(flip, (1,48,48,1))]).astype(np.float32)
                preds = ort_session.run(None, {input_name: batch})[0]
                avg   = np.mean(preds, axis=0)
                state["smoother"].append(avg)
                smoothed   = np.mean(state["smoother"], axis=0)
                idx        = int(np.argmax(smoothed))
                emotion_str= EMOTION_LABELS[idx]
                conf_val   = float(smoothed[idx])
                if conf_val > max_conf:
                    max_conf = conf_val
                    dominant = emotion_str
            response_data["faces"].append(
                {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "emotion": emotion_str})

        response_data["stats"]["emotion"]    = dominant
        response_data["stats"]["confidence"] = max_conf

        # Alert logic
        now = time.time()
        if dominant != state["last_emotion"]:
            state.update({"last_emotion": dominant, "emotion_start": now, "alert_fired": False})
        elif dominant in ALERT_RULES and not state["alert_fired"]:
            elapsed = now - state["emotion_start"]
            if elapsed >= ALERT_RULES[dominant]:
                state["alert_fired"] = True
                response_data["alert"] = {"emotion": dominant, "duration": round(elapsed)}
                a = AlertLog(browser_id=browser_id, session_id=det_session_id,
                             emotion=dominant, duration_seconds=elapsed)
                db.session.add(a)
                db.session.commit()

        # Persist every 10 frames
        s = DetectionSession.query.get(det_session_id)
        if s:
            s.total_frames += 1
            if s.total_frames % 10 == 0 and dominant not in ("Scanning...", "No Face Detected"):
                db.session.add(EmotionLog(session_id=det_session_id, emotion=dominant,
                                          confidence=max_conf, faces_count=len(faces)))
            db.session.commit()

        emit("result", response_data)

    except Exception as e:
        print("Error:", e)


# ── Boot ──────────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    print(f"[DB] Ready at {_db_path}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    print(f"Running on http://localhost:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
