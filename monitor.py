"""Lab monitor engine (no UI). The web UI in app.py drives it.

Timetable says when the lab is booked. During a booked slot, recognised
people are marked present in Attendance/<Subject>_<date>_<start>.csv.
Outside booked slots the monitor is in ALERT mode: any presence (motion or a
face) is identified and logged with a timestamp and snapshot.

The optional WiFi camera runs the same face recognition, counting, attendance
and alerts as the main camera, and also records video to Recordings/ on alert.

Optional ESP32 + PIR sensor over UART (SERIAL_PORT): the camera and face model
sleep until the sensor reports MOTION, saving energy. If the sensor is
unplugged or silent, the camera stays on (fail-safe).

Face pipeline: YuNet detector + SFace embeddings (opencv-python >= 4.8, models
auto-downloaded to models/). A per-face tracker keeps an identity through a
few bad frames; "Unknown" is only logged after it stays unknown for a while.

python monitor.py --selftest | --download
"""
import csv
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

try:
    import serial
except ImportError:  # PIR sensor support is optional
    serial = None

TIMETABLE_CSV = "timetable.csv"
LOG_CSV = "lab_log.csv"
SNAP_DIR = "Intrusions"
REC_DIR = "Recordings"
ATT_DIR = "Attendance"
FACE_DB_DIR = "FaceDB"
STUDENTS_CSV = os.path.join("StudentDetails", "StudentDetails.csv")
SETTINGS_JSON = "settings.json"
MODEL_DIR = "models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
REC_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
MODEL_URLS = {
    DET_MODEL: ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    REC_MODEL: ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
STUDENT_FIELDS = ["Enrollment", "Name", "Role", "Date", "Time"]
ROLES = ["Student", "Faculty"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TT_FIELDS = ["Day", "Start", "End", "Subject", "Faculty", "MaxStudents"]
MODES = ["Auto (timetable)", "Always alert", "Off"]
ENROLL_PROMPTS = ["Look straight", "Turn head left and right", "Tilt head up and down",
                  "Smile / laugh", "Tongue out, funny faces", "Talk, open mouth wide",
                  "Move closer and further, glasses on/off"]

# Calibration knobs (editable on the website's Settings tab, saved to settings.json)
CAMERA_INDEX = 0
DET_SCORE = 0.6
MATCH_THRESHOLD = 0.38
HOLD_FRAMES = 60
UNKNOWN_CONFIRM = 15
MOTION_RATIO = 0.02
MOTION_CONFIRM = 10
MAX_MISSED = 15
ENROLL_SAMPLES = 70
LOG_COOLDOWN_S = 30
CAM2_TYPE = "off"
CAM2_IP = ""
CAM2_PORT = 0
CAM2_USER = ""
CAM2_PASS = ""
CAM2_PATH = ""
CAM2_URL = ""
RECORD_SECONDS = 30
MAX_CLIP_S = 300
WEB_PORT = 5000
SERIAL_PORT = ""
SERIAL_BAUD = 115200
WAKE_SECONDS = 60
STAFF_ABSENT_S = 10

# WiFi camera presets: type -> (scheme, default port, default path)
CAM_PRESETS = {
    "ipwebcam": ("http", 8080, "video"),   # Android "IP Webcam" app
    "rtsp": ("rtsp", 554, "stream1"),      # most IP / CCTV cameras
    "esp32cam": ("http", 81, "stream"),    # ESP32-CAM CameraWebServer example
    "http": ("http", 80, "video"),         # any MJPEG-over-HTTP camera
}
CAM2_TYPES = ["off", *CAM_PRESETS, "usb", "custom"]
HOST_RE = re.compile(r"[A-Za-z0-9.\-]+")

SETTINGS_HELP = {
    "CAMERA_INDEX": "Main camera number (0 = built-in, 1 = first USB cam...)",
    "DET_SCORE": "Face detector confidence 0-1. Lower finds more faces (and more false ones)",
    "MATCH_THRESHOLD": "Recognition strictness 0-1. Raise if strangers match, lower if people are missed",
    "HOLD_FRAMES": "Frames to keep a name through odd expressions/blur (~20 frames = 1 s)",
    "UNKNOWN_CONFIRM": "Frames a face must stay unrecognised before logged as Unknown",
    "MOTION_RATIO": "Fraction of picture that must change to count as motion (0.02 = 2%)",
    "MOTION_CONFIRM": "Frames of faceless motion before it is logged",
    "MAX_MISSED": "Frames a tracked face survives when the detector loses it",
    "ENROLL_SAMPLES": "Face samples captured per enrollment",
    "LOG_COOLDOWN_S": "Seconds before the same person is logged again",
    "CAM2_TYPE": "WiFi camera type",
    "CAM2_IP": "WiFi camera IP address (or camera number for a USB camera)",
    "CAM2_PORT": "WiFi camera port (0 = default for the type)",
    "CAM2_USER": "WiFi camera username (empty = none)",
    "CAM2_PASS": "WiFi camera password (leave empty to keep the saved one)",
    "CAM2_PATH": "Stream path (empty = default for the type)",
    "CAM2_URL": "Full stream URL (only for type 'custom')",
    "RECORD_SECONDS": "Keep recording this many seconds after the last presence",
    "MAX_CLIP_S": "Split recordings into files of at most this many seconds",
    "WEB_PORT": "Website port (restart the app after changing)",
    "SERIAL_PORT": "ESP32 PIR sensor port, e.g. COM5 (or socket://ip:port). Empty = off, camera always on",
    "SERIAL_BAUD": "PIR sensor baud rate (must match the ESP32 sketch)",
    "WAKE_SECONDS": "Keep camera + face recognition on this long after the last motion",
    "STAFF_ABSENT_S": "Seconds without seeing any faculty face before staff shows Absent",
}
DEFAULTS = {k: globals()[k] for k in SETTINGS_HELP}
UNIT_RANGE = ("DET_SCORE", "MATCH_THRESHOLD", "MOTION_RATIO")


# ---------- settings ----------
def read_json():
    try:
        with open(SETTINGS_JSON) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_json(**extra):
    data = read_json()
    data.update({k: globals()[k] for k in DEFAULTS}, **extra)
    with open(SETTINGS_JSON, "w") as f:
        json.dump(data, f, indent=2)


def load_settings():
    for k, v in read_json().items():
        if k in DEFAULTS:
            globals()[k] = type(DEFAULTS[k])(v)


def current_settings():
    hide = {"CAM2_PASS"}  # never send the camera password back to the browser
    return [{"key": k, "value": "" if k in hide else globals()[k], "default": DEFAULTS[k], "help": h}
            for k, h in SETTINGS_HELP.items()]


def build_cam2_url(t, ip, port, user, pw, path, custom):
    if t == "custom":
        return custom.strip()
    if t == "usb":
        return str(ip).strip()
    if t not in CAM_PRESETS or not ip:
        return ""
    scheme, dport, dpath = CAM_PRESETS[t]
    auth = f"{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(pw, safe='')}@" if user else ""
    return f"{scheme}://{auth}{ip}:{port or dport}/{(path or dpath).lstrip('/')}"


def cam2_url(redact=False):
    pw = "***" if redact and CAM2_PASS else CAM2_PASS
    return build_cam2_url(CAM2_TYPE, CAM2_IP, CAM2_PORT, CAM2_USER, pw, CAM2_PATH, CAM2_URL)


def parse_settings(form):
    """Validate a {key: text} dict. Returns (values, error)."""
    new = {}
    for k in DEFAULTS:
        t = type(DEFAULTS[k])
        try:
            new[k] = t(str(form.get(k, globals()[k])).strip())
        except ValueError:
            return None, f"{k}: must be a {t.__name__}"
        if t is not str and (new[k] < 0 or (k in UNIT_RANGE and new[k] > 1)):
            return None, f"{k}: out of range"
    if not new["CAM2_PASS"]:
        new["CAM2_PASS"] = CAM2_PASS  # empty field = keep saved password
    if new["CAM2_TYPE"] not in CAM2_TYPES:
        return None, "Unknown WiFi camera type"
    if new["CAM2_TYPE"] in CAM_PRESETS and not HOST_RE.fullmatch(new["CAM2_IP"]):
        return None, "WiFi camera IP address required (e.g. 192.168.1.50)"
    if new["CAM2_TYPE"] == "usb" and not new["CAM2_IP"].isdigit():
        return None, "USB camera: put the camera number (e.g. 1) in the IP field"
    if new["CAM2_TYPE"] == "custom" and not new["CAM2_URL"]:
        return None, "Custom type needs the full stream URL"
    if new["CAM2_PORT"] > 65535 or new["WEB_PORT"] > 65535:
        return None, "Port must be 0-65535"
    return new, None


# ---------- helpers ----------
def slot_at(rows, dt):
    """Return the timetable row active at dt, or None if lab is free."""
    day, hm = DAYS[dt.weekday()], dt.strftime("%H:%M")
    for r in rows:
        if r["Day"] == day and r["Start"] <= hm < r["End"]:
            return r
    return None


def slot_limit(slot):
    """Max students for a timetable slot (0 = no limit)."""
    v = str((slot or {}).get("MaxStudents") or "").strip()
    return int(v) if v.isdigit() else 0


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def best_match(embeddings, labels, f):
    """Closest enrolled sample by cosine similarity (rows and f are L2-normalised)."""
    if not labels:
        return "", 0.0
    sims = embeddings @ f
    i = int(sims.argmax())
    return labels[i], float(sims[i])


def head_count(track_lists, faculty):
    """(people, faculty, students) over all cameras. Anyone not faculty counts as a student.
    A recognised person counts once even if both cameras see them."""
    known = {t["enr"] for ts in track_lists for t in ts if t["enr"]}
    # ponytail: unknown faces can't be matched across cameras, so take the larger camera's count
    unknown = max((sum(not t["enr"] for t in ts) for ts in track_lists), default=0)
    staff = len(known & faculty)
    return len(known) + unknown, staff, len(known) + unknown - staff


def safe(name):
    return re.sub(r"[^\w\- ()]", "_", name)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if any(r.values())]


def append_csv(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerow(row)


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            print("Downloading", url)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)


def open_source(url):
    return cv2.VideoCapture(int(url) if url.isdigit() else url)


def grab_jpeg(url):
    """One frame from a camera URL as JPEG bytes, or None."""
    cap = open_source(url)
    ok, frame = cap.read()
    cap.release()
    return cv2.imencode(".jpg", frame)[1].tobytes() if ok else None


# ---------- WiFi camera recorder ----------
class Recorder:
    """Writes WiFi camera frames from the first alert until RECORD_SECONDS after the last one."""
    FPS = 10

    def __init__(self):
        self.until = 0.0
        self.writer = None
        self.status = "idle"

    def trigger(self):
        self.until = time.time() + RECORD_SECONDS

    def stop(self):
        if self.writer:
            self.writer.release()
        self.writer, self.status = None, "idle"

    def feed(self, frame):
        """Called with every WiFi camera frame (before boxes are drawn)."""
        now = time.time()
        if now >= self.until:
            if self.writer:
                self.stop()
            return
        if self.writer is None or now - self.started > MAX_CLIP_S:
            self.stop()
            name = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.webm"
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(os.path.join(REC_DIR, name), cv2.VideoWriter_fourcc(*"VP80"), self.FPS, (w, h))
            self.started = self.next_t = now
            self.status = "recording " + name
        # pace to wall clock: slow streams repeat frames, fast ones drop them
        while self.next_t <= now:
            out = frame.copy()
            cv2.putText(out, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.writer.write(out)
            self.next_t += 1 / self.FPS


class LatestFrame:
    """Reads a stream in its own thread and keeps only the newest frame, so a slow
    face model never lets a network stream fall behind real time."""

    def __init__(self, url):
        self.url, self.frame, self.alive = url, None, True
        self.cap = open_source(url)
        self.t = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                break
            self.frame, self.t = f, time.time()
        self.alive = False
        self.cap.release()

    def close(self):
        self.alive = False


class View:
    """Per-camera state: detector, face tracks, motion model, people seen, last JPEG."""

    def __init__(self, name):
        self.name = name
        self.jpeg = None
        self.status = "off"
        self.make_detector()
        self.clear()

    def make_detector(self):
        self.detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), DET_SCORE, 0.3, 5000)

    def clear(self):
        self.tracks, self.motion_frames, self.people = [], 0, []
        self.bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


# ---------- ESP32 PIR sensor over UART ----------
class PirSensor:
    """Reads lines from the ESP32: MOTION = wake the camera, ALIVE = heartbeat.

    awake() is True while motion is recent, and also whenever the sensor is off,
    unplugged or silent for 30 s, so a dead sensor never blinds the monitor.
    """
    STALE_S = 30

    def __init__(self, start=True):
        self.awake_until = 0.0
        self.connected = False
        self.last_line = 0.0
        self.last_motion = None
        self.status = "off"
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def handle_line(self, line):
        self.last_line = time.time()
        if line.strip() == "MOTION":
            self.motion()

    def motion(self):
        self.last_motion = datetime.datetime.now()
        self.awake_until = time.time() + WAKE_SECONDS

    def keep_awake(self):
        self.awake_until = max(self.awake_until, time.time() + WAKE_SECONDS)

    def gating(self):
        """True when the sensor is healthy and allowed to put the camera to sleep."""
        return bool(SERIAL_PORT) and self.connected and time.time() - self.last_line < self.STALE_S

    def awake(self):
        return not self.gating() or time.time() < self.awake_until

    def _run(self):
        while True:
            port, baud = SERIAL_PORT, SERIAL_BAUD
            if not port:
                self.connected, self.status = False, "off"
                time.sleep(1)
                continue
            if serial is None:
                self.status = "pyserial not installed (run install.bat)"
                time.sleep(5)
                continue
            try:
                s = serial.serial_for_url(port, baud, timeout=1, do_not_open=True)
                s.dtr = s.rts = False  # don't reset the ESP32 when the port opens
                s.open()
                with s:
                    self.connected, self.last_line = True, time.time()
                    self.status = f"connected on {port}"
                    while SERIAL_PORT == port and SERIAL_BAUD == baud:
                        line = s.readline()
                        if line:
                            self.handle_line(line.decode(errors="ignore"))
                        elif time.time() - self.last_line > self.STALE_S:
                            self.status = f"no data from sensor on {port} (camera kept on)"
            except (serial.SerialException, OSError, ValueError) as ex:
                self.status = f"{port}: {ex} (camera kept on)"
                self.connected = False
                time.sleep(3)
            self.connected = False


SLEEP_JPEG = None


def sleep_jpeg():
    global SLEEP_JPEG
    if SLEEP_JPEG is None:
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(img, "Camera sleeping", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        cv2.putText(img, "waiting for PIR motion", (150, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        SLEEP_JPEG = cv2.imencode(".jpg", img)[1].tobytes()
    return SLEEP_JPEG


# ---------- engine ----------
class Monitor:
    def __init__(self):
        for d in (SNAP_DIR, REC_DIR, FACE_DB_DIR, ATT_DIR):
            os.makedirs(d, exist_ok=True)
        self.lock = threading.RLock()
        self.timetable = read_csv(TIMETABLE_CSV)
        students = read_csv(STUDENTS_CSV)
        self.names = {r["Enrollment"]: r["Name"] for r in students}
        self.faculty = {r["Enrollment"] for r in students if r.get("Role") == "Faculty"}
        self.staff_seen = None  # last time any faculty face was in view
        self.staff_names = []
        self.main, self.cam2 = View("main"), View("wifi")
        self.views = (self.main, self.cam2)
        self.recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")
        self._load_db()
        self.enrolling = None
        self.att = None
        self.last_logged = {}
        self.log_rows = read_csv(LOG_CSV)[-200:]
        self.mode = MODES[0]
        self.status = ("Starting...", "grey")
        self.notice = ""
        self.camera_ok = False
        self.recorder = Recorder()
        self.pir = PirSensor()
        self.cam_index = CAMERA_INDEX
        self.cam = cv2.VideoCapture(CAMERA_INDEX)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._cam2_loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                self.tick()
            except Exception as ex:  # keep monitoring; show the problem on the website
                self.status = (f"Error: {ex}", "red")
                time.sleep(1)

    def _load_db(self):
        embs, self.labels = [], []
        for fn in os.listdir(FACE_DB_DIR):
            if fn.endswith(".npy"):
                e = np.load(os.path.join(FACE_DB_DIR, fn))
                embs.append(e)
                self.labels += [fn[:-4]] * len(e)
        self.embeddings = np.vstack(embs) if embs else np.zeros((0, 128), np.float32)

    def name_of(self, enr):
        return self.names.get(enr, f"ID {enr}") if enr else "Unknown"

    # ---------- queries for the web UI ----------
    def state(self):
        with self.lock:
            att = None
            if self.att:
                s, present = self.att["slot"], self.att["present"]
                att = {"title": f"{s['Subject']} ({s['Faculty']}) {self.att['date']} {s['Start']}-{s['End']}",
                       "present": len(present)}
            ids = sorted((set(self.labels) - self.faculty) | set(self.att["present"] if self.att else {}))
            total, staff, students = head_count([v.tracks for v in self.views], self.faculty)
            staff_present = bool(self.staff_seen) and time.time() - self.staff_seen < STAFF_ABSENT_S
            return {
                "status": self.status[0], "color": self.status[1], "mode": self.mode, "modes": MODES,
                "camera_ok": self.camera_ok, "notice": self.notice, "enrolling": bool(self.enrolling),
                "recorder": f"{self.cam2.status} | {self.recorder.status}" if cam2_url() else "not configured",
                "in_view": {v.name: head_count([v.tracks], self.faculty)[0] for v in self.views},
                "cam2_url": cam2_url(redact=True),
                "pir": self.pir.status, "pir_gating": self.pir.gating(), "awake": self.pir.awake(),
                "awake_left": max(0, int(self.pir.awake_until - time.time())),
                "last_motion": f"{self.pir.last_motion:%H:%M:%S}" if self.pir.last_motion else "",
                "enrolled": len(set(self.labels)), "samples": len(self.labels), "log_count": len(self.log_rows),
                "attendance": att,
                "count": {"total": total, "faculty": staff, "students": students,
                          "limit": slot_limit(self.att["slot"]) if self.att else 0,
                          "staff": "Present" if staff_present else "Absent",
                          "staff_names": self.staff_names if staff_present else [],
                          "staff_last_seen": f"{datetime.datetime.fromtimestamp(self.staff_seen):%H:%M:%S}"
                                             if self.staff_seen else ""},
                "attendance_rows": [{"id": e, "name": self.name_of(e),
                                     "status": ("Present" if e in self.att["present"] else "Absent") if self.att else "-",
                                     "time": self.att["present"].get(e, "") if self.att else ""} for e in ids],
            }

    def users(self):
        with self.lock:
            return [{"id": e, "name": self.name_of(e), "samples": self.labels.count(e),
                     "role": "Faculty" if e in self.faculty else "Student"}
                    for e in sorted(set(self.names) | set(self.labels))]

    def log(self):
        with self.lock:
            return [{**r, "Snapshot": os.path.basename(r["Snapshot"])} for r in reversed(self.log_rows)]

    # ---------- commands from the web UI (return error text or None) ----------
    def set_mode(self, mode):
        if mode not in MODES:
            return "Unknown mode"
        self.mode = mode

    def add_slot(self, row):
        row = {c: str(row.get(c, "")).strip() for c in TT_FIELDS}
        if row["Day"] not in DAYS:
            return "Pick a day"
        try:
            start = datetime.datetime.strptime(row["Start"], "%H:%M")
            end = datetime.datetime.strptime(row["End"], "%H:%M")
        except ValueError:
            return "Start/End must be HH:MM (24h)"
        if end <= start:
            return "End must be after Start"
        if row["MaxStudents"] and not row["MaxStudents"].isdigit():
            return "Max students must be a whole number (empty = no limit)"
        if not row["Subject"]:
            return "Subject required"
        with self.lock:
            self.timetable.append(row)
            self._save_timetable()

    def timetable_rows(self):
        with self.lock:
            return [{**r, "live": bool(self.att) and self.att["slot"] is r} for r in self.timetable]

    def delete_slot(self, i):
        with self.lock:
            if not 0 <= i < len(self.timetable):
                return "No such slot"
            del self.timetable[i]
            self._save_timetable()

    def _save_timetable(self):
        self.timetable.sort(key=lambda r: (DAYS.index(r["Day"]), r["Start"]))
        write_csv(TIMETABLE_CSV, TT_FIELDS, self.timetable)

    def start_enroll(self, enr, name):
        enr, name = str(enr).strip(), str(name).strip()
        if not enr.isalnum():
            return "ID must be letters/digits only"
        with self.lock:
            if self.enrolling:
                return "Already enrolling someone"
            if not name and enr not in self.names:
                return "Name required for a new ID"
            self.enrolling = {"enr": enr, "name": name or self.names[enr], "samples": [], "frame": 0}

    def cancel_enroll(self):
        self.enrolling = None

    def rename_user(self, enr, name):
        name = str(name).strip()
        if not name:
            return "Name required"
        with self.lock:
            if enr not in self.names and enr not in self.labels:
                return "No such user"
            self.names[enr] = name
            self.save_students()

    def set_role(self, enr, role):
        if role not in ROLES:
            return "Role must be Student or Faculty"
        with self.lock:
            if enr not in self.names:
                return "No such user"
            (self.faculty.add if role == "Faculty" else self.faculty.discard)(enr)
            self.save_students()

    def delete_user(self, enr):
        with self.lock:
            path = os.path.join(FACE_DB_DIR, safe(enr) + ".npy")
            if os.path.exists(path):
                os.remove(path)
            self.names.pop(enr, None)
            self.faculty.discard(enr)
            self.save_students()
            self._load_db()
            for v in self.views:
                v.tracks = []

    def save_students(self):
        now = datetime.datetime.now()
        old = {r["Enrollment"]: r for r in read_csv(STUDENTS_CSV)}
        write_csv(STUDENTS_CSV, STUDENT_FIELDS,
                  [{**old.get(e, {"Date": f"{now:%Y-%m-%d}", "Time": f"{now:%H:%M:%S}"}), "Enrollment": e, "Name": n,
                    "Role": "Faculty" if e in self.faculty else "Student"} for e, n in self.names.items()])

    def apply_settings(self, values):
        with self.lock:
            globals().update(values)
            save_json()
            for v in self.views:
                v.make_detector()
            if CAMERA_INDEX != self.cam_index:
                self.cam.release()
                self.cam_index = CAMERA_INDEX
                self.cam = cv2.VideoCapture(CAMERA_INDEX)

    # ---------- camera loop ----------
    def embed(self, frame, face):
        f = self.recognizer.feature(self.recognizer.alignCrop(frame, face)).flatten()
        return f / (np.linalg.norm(f) + 1e-9)

    def enroll_step(self, frame, faces):
        e = self.enrolling
        e["frame"] += 1
        n = len(e["samples"])
        prompt = ENROLL_PROMPTS[min(n * len(ENROLL_PROMPTS) // ENROLL_SAMPLES, len(ENROLL_PROMPTS) - 1)]
        if len(faces) != 1:
            self.status = (f"Enrolling {e['name']}: exactly one face in view please", "orange")
            return
        if e["frame"] % 3 == 0:
            e["samples"].append(self.embed(frame, faces[0]))
        self.status = (f"Enrolling {e['name']}: {prompt}  ({n}/{ENROLL_SAMPLES})", "blue")
        if len(e["samples"]) >= ENROLL_SAMPLES:
            self.finish_enroll()

    def finish_enroll(self):
        e, self.enrolling = self.enrolling, None
        path = os.path.join(FACE_DB_DIR, e["enr"] + ".npy")
        samples = np.array(e["samples"], np.float32)
        if os.path.exists(path):
            samples = np.vstack([np.load(path), samples])
        np.save(path, samples)
        self.names[e["enr"]] = e["name"]
        self.save_students()
        self._load_db()
        self.notice = f"{e['name']} saved with {len(samples)} samples ({datetime.datetime.now():%H:%M:%S})"

    def update_tracks(self, v, frame, faces):
        old, new = v.tracks, []
        for face in faces:
            box = tuple(face[:4])
            t = max(old, key=lambda t: iou(t["box"], box), default=None)
            if t is not None and iou(t["box"], box) > 0.3:
                old.remove(t)
            else:
                t = {"enr": "", "sim": 0.0, "hold": 0, "unk": 0}
            t.update(box=box, missed=0)
            enr, sim = best_match(self.embeddings, self.labels, self.embed(frame, face))
            if sim >= MATCH_THRESHOLD:
                t.update(enr=enr, sim=sim, hold=HOLD_FRAMES, unk=0)
            elif t["hold"] > 0:
                t["hold"] -= 1  # keep last identity through odd expressions / blur
            else:
                t.update(enr="", sim=sim, unk=t["unk"] + 1)
            new.append(t)
        for t in old:  # face briefly lost by detector
            t["missed"] += 1
            if t["missed"] <= MAX_MISSED:
                t["hold"] = max(0, t["hold"] - 1)
                new.append(t)
        v.tracks = new

    def update_attendance(self, slot, now, people):
        """Switch sheet when the class changes, then mark recognised people present."""
        path = slot and os.path.join(ATT_DIR, f"{safe(slot['Subject'])}_{now:%Y-%m-%d}_{slot['Start'].replace(':', '')}.csv")
        if path != (self.att and self.att["path"]):
            self.att = path and {"path": path, "slot": slot, "date": f"{now:%Y-%m-%d}",
                                 "present": {r["Enrollment"]: r["Time"] for r in read_csv(path)}}
        if not self.att:
            return
        for name, enr, _ in people:
            if enr and enr not in self.faculty and enr not in self.att["present"]:
                self.att["present"][enr] = f"{now:%H:%M:%S}"
                append_csv(path, ["Enrollment", "Name", "Date", "Time"], [enr, name, f"{now:%Y-%m-%d}", f"{now:%H:%M:%S}"])

    def tick(self):
        if not self.pir.awake() and not self.enrolling:
            if self.cam.isOpened():  # sleep: free the camera and skip the face model
                self.cam.release()
                self.main.clear()
            self.camera_ok = True
            self.status = ("Sleeping - camera off until the PIR sensor detects motion", "grey")
            self.main.jpeg = sleep_jpeg()
            time.sleep(0.2)
            return
        if not self.cam.isOpened():
            self.cam = cv2.VideoCapture(self.cam_index)
            self.main.clear()
        ok, frame = self.cam.read()
        self.camera_ok = ok
        if not ok:
            self.status = (f"Main camera {self.cam_index} not available (check Settings)", "red")
            time.sleep(0.5)
            return
        with self.lock:
            self._process(self.main, frame)

    def _cam2_loop(self):
        """WiFi camera: same face pipeline as the main camera, plus recording on alert."""
        v, src, last = self.cam2, None, None
        while True:
            try:
                url = cam2_url() if self.pir.awake() else ""
                if src and (src.url != url or not src.alive or time.time() - src.t > 10):
                    lost = src.url == url
                    src.close()
                    src = None
                    self.recorder.stop()
                    with self.lock:
                        v.clear()
                    v.jpeg = None
                    if lost:
                        v.status = f"not reachable: {cam2_url(redact=True)} (retrying)"
                        time.sleep(3)
                    continue
                if not url:
                    v.status = "sleeping until PIR motion" if cam2_url() else "off"
                    time.sleep(0.5)
                    continue
                if src is None:
                    v.status = "connecting..."
                    src = LatestFrame(url)
                    continue
                frame = src.frame
                if frame is None or frame is last:
                    time.sleep(0.01)
                    continue
                last = frame
                v.status = "watching"
                self.recorder.feed(frame)  # full resolution, no boxes
                if frame.shape[1] > 960:  # big IP camera streams: detect on a smaller copy to keep up
                    frame = cv2.resize(frame, (960, frame.shape[0] * 960 // frame.shape[1]))
                else:
                    frame = frame.copy()
                # ponytail: one lock for both cameras (shared recognizer), so fps drops; per-camera recognizer if that hurts
                with self.lock:
                    self._process(v, frame)
            except Exception as ex:  # keep going; show the problem on the website
                v.status = f"Error: {ex}"
                time.sleep(1)

    def _process(self, v, frame):
        now = datetime.datetime.now()
        slot = slot_at(self.timetable, now)
        armed = self.mode == "Always alert" or (self.mode == MODES[0] and slot is None)
        main = v is self.main
        if main and slot and self.mode == MODES[0]:
            self.status = (f"Scheduled: {slot['Subject']} ({slot['Faculty']}) {slot['Start']}-{slot['End']}", "green")
        elif main and armed:
            self.status = ("ALERT MODE - lab not scheduled", "red")
        elif main:
            self.status = ("Monitoring off", "grey")

        h, w = frame.shape[:2]
        v.detector.setInputSize((w, h))
        faces = v.detector.detect(frame)[1]
        faces = [] if faces is None else list(faces)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = (v.bg.apply(gray) > 0).mean() > MOTION_RATIO

        people = []
        if self.enrolling and main:
            self.enroll_step(frame, faces)
            for face in faces:
                x, y, bw, bh = map(int, face[:4])
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 140, 0), 3)
        else:
            self.update_tracks(v, frame, faces)
            for t in v.tracks:
                if t["missed"]:
                    continue
                x, y, bw, bh = map(int, t["box"])
                if t["enr"] in self.faculty:
                    label, color = f"{self.name_of(t['enr'])} (Faculty) {t['sim']:.2f}", (255, 0, 200)
                    people.append((self.name_of(t["enr"]), t["enr"], f"{t['sim']:.2f}"))
                elif t["enr"]:
                    label, color = f"{self.name_of(t['enr'])} {t['sim']:.2f}", (0, 200, 0)
                    people.append((self.name_of(t["enr"]), t["enr"], f"{t['sim']:.2f}"))
                elif t["unk"] >= UNKNOWN_CONFIRM:
                    label, color = f"Unknown {t['sim']:.2f}", (0, 0, 255)
                    people.append(("Unknown", "", f"{t['sim']:.2f}"))
                else:
                    label, color = "Checking...", (0, 220, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            v.motion_frames = v.motion_frames + 1 if motion and not v.tracks else 0
            if v.motion_frames >= MOTION_CONFIRM:
                people.append(("Unknown (no face)", "", ""))

        staff_here = sorted({self.name_of(t["enr"]) for u in self.views for t in u.tracks if t["enr"] in self.faculty})
        if staff_here:
            self.staff_seen, self.staff_names = time.time(), staff_here
        total, staff, students = head_count([v.tracks], self.faculty)
        limit = slot_limit(slot)
        in_lab = head_count([u.tracks for u in self.views], self.faculty)[2]
        if main and slot and self.mode == MODES[0] and limit and in_lab > limit:
            self.status = (f"OVER LIMIT: {in_lab}/{limit} students - {slot['Subject']} {slot['Start']}-{slot['End']}", "orange")
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"People {total} | Faculty {staff} | Students {students}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if people or v.tracks or self.enrolling:
            self.pir.keep_awake()  # PIR misses people sitting still; the camera doesn't
        v.people = people
        seen = self.main.people + self.cam2.people
        if armed and main and seen:
            self.status = ("INTRUSION: " + ", ".join(sorted({p[0] for p in seen})), "darkred")
        if armed and people:
            self.recorder.trigger()
            for p in people:
                self.log_event(now, frame, *p, cam="" if main else "_wifi")
        self.update_attendance(slot if self.mode != "Off" else None, now, people)
        v.jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def log_event(self, now, frame, name, enr, conf, cam=""):
        if (now - self.last_logged.get(name, datetime.datetime.min)).total_seconds() < LOG_COOLDOWN_S:
            return
        self.last_logged[name] = now
        d, t = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        snap = os.path.join(SNAP_DIR, f"{d}_{t.replace(':', '-')}_{safe(name)}{cam}.jpg")
        cv2.imwrite(snap, frame)
        row = {"Date": d, "Time": t, "Name": name, "Enrollment": enr, "Confidence": conf, "Snapshot": snap}
        append_csv(LOG_CSV, list(row), list(row.values()))
        self.log_rows = (self.log_rows + [row])[-200:]


def selftest():
    tt = [{"Day": "Fri", "Start": "09:00", "End": "10:30", "Subject": "OOP", "Faculty": "X"}]
    fri = datetime.datetime(2026, 10, 16)  # a Friday
    assert slot_at(tt, fri.replace(hour=9))["Subject"] == "OOP"
    assert slot_at(tt, fri.replace(hour=10, minute=29)) is not None
    assert slot_at(tt, fri.replace(hour=10, minute=30)) is None   # end is exclusive
    assert slot_at(tt, fri.replace(hour=8, minute=59)) is None
    assert slot_at(tt, datetime.datetime(2026, 10, 15, 9, 30)) is None  # Thursday
    assert slot_limit(tt[0]) == 0 and slot_limit(None) == 0               # old timetable.csv: no column
    assert slot_limit({"MaxStudents": "20"}) == 20 and slot_limit({"MaxStudents": ""}) == 0

    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1) < 1e-6
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0
    e = np.eye(3, dtype=np.float32)
    assert best_match(e, ["a", "b", "c"], np.array([0.1, 0.9, 0.1], np.float32))[0] == "b"
    assert best_match(np.zeros((0, 3)), [], e[0]) == ("", 0.0)
    assert safe("a/b:c") == "a_b_c"
    # 20 students + 1 faculty, then one student leaves, then the faculty leaves
    room = [{"enr": "F1"}] + [{"enr": f"S{i}"} for i in range(15)] + [{"enr": ""}] * 5
    assert head_count([room], {"F1"}) == (21, 1, 20)
    assert head_count([room[:-1]], {"F1"}) == (20, 1, 19)
    assert head_count([room[1:-1]], {"F1"}) == (19, 0, 19)
    # both cameras see the class: each person counted once; faculty seen by either camera is enough
    assert head_count([room, room[1:]], {"F1"}) == (21, 1, 20)
    assert head_count([room[1:], room[:3]], {"F1"}) == (21, 1, 20)
    assert head_count([[], []], {"F1"}) == (0, 0, 0)

    vals, err = parse_settings({"MATCH_THRESHOLD": "0.5", "CAM2_URL": " rtsp://x "})
    assert err is None and vals["MATCH_THRESHOLD"] == 0.5 and vals["CAM2_URL"] == "rtsp://x"
    assert parse_settings({"MATCH_THRESHOLD": "2"})[1] and parse_settings({"CAMERA_INDEX": "x"})[1]
    assert build_cam2_url("ipwebcam", "192.168.1.5", 0, "", "", "", "") == "http://192.168.1.5:8080/video"
    assert build_cam2_url("rtsp", "10.0.0.2", 8554, "admin", "p@ss:1", "live", "") == "rtsp://admin:p%40ss%3A1@10.0.0.2:8554/live"
    assert build_cam2_url("esp32cam", "1.2.3.4", 0, "", "", "", "") == "http://1.2.3.4:81/stream"
    assert build_cam2_url("usb", "1", 0, "", "", "", "") == "1"
    assert build_cam2_url("off", "1.2.3.4", 0, "", "", "", "x") == ""
    assert parse_settings({"CAM2_TYPE": "rtsp", "CAM2_IP": "bad/host"})[1]
    assert parse_settings({"CAM2_TYPE": "rtsp", "CAM2_IP": "192.168.1.9"})[1] is None

    global SERIAL_PORT
    pir = PirSensor(start=False)
    assert pir.awake()                       # no sensor configured: always on
    SERIAL_PORT, pir.connected, pir.last_line = "COM9", True, time.time()
    assert not pir.awake()                   # healthy sensor, no motion: sleep
    pir.handle_line("MOTION\r\n")
    assert pir.awake()                       # motion wakes
    pir.awake_until = 0
    pir.last_line = time.time() - 60
    assert pir.awake()                       # silent sensor: fail-safe on
    SERIAL_PORT = ""
    print("selftest ok")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1:] == ["--download"]:
        ensure_models()
    else:
        selftest()
