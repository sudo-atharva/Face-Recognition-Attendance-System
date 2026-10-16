"""Lab occupancy monitor.

Timetable says when the lab is booked. During a booked slot, recognised
people are marked present in Attendance/<Subject>_<date>_<start>.csv. Outside booked slots the monitor is in
ALERT mode: any presence (motion or a face) on camera is identified and logged
with a timestamp and snapshot.

Face pipeline: YuNet detector + SFace embeddings (deep models built into
opencv-python >= 4.8, auto-downloaded to models/). Much more robust to
expressions, pose and lighting than the old Haar + LBPH. A per-face tracker
keeps an identity through a few bad frames, and "Unknown" is only logged after
it stays unknown for a while.

Run: run.bat (after install.bat), or python lab_monitor.py
     python lab_monitor.py --selftest | --download
"""
import csv
import json
import datetime
import os
import re
import sys
import tkinter as tk
import urllib.request
from tkinter import ttk, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

TIMETABLE_CSV = "timetable.csv"
LOG_CSV = "lab_log.csv"
SNAP_DIR = "Intrusions"
ATT_DIR = "Attendance"
FACE_DB_DIR = "FaceDB"
STUDENTS_CSV = os.path.join("StudentDetails", "StudentDetails.csv")
MODEL_DIR = "models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
REC_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
MODEL_URLS = {
    DET_MODEL: ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    REC_MODEL: ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TT_FIELDS = ["Day", "Start", "End", "Subject", "Faculty"]
ENROLL_PROMPTS = ["Look straight", "Turn head left and right", "Tilt head up and down",
                  "Smile / laugh", "Tongue out, funny faces", "Talk, open mouth wide",
                  "Move closer and further, glasses on/off"]

# Calibration knobs
CAMERA_INDEX = 0
DET_SCORE = 0.6          # YuNet face confidence; lower finds more (and more false) faces
MATCH_THRESHOLD = 0.38   # SFace cosine similarity; higher = stricter. Watch the on-screen score to tune
HOLD_FRAMES = 60         # keep a recognised identity this many frames without a fresh match (~3 s)
UNKNOWN_CONFIRM = 15     # frames a face must stay unrecognised before it is logged as Unknown
MOTION_RATIO = 0.02      # fraction of frame that must change to count as presence
MOTION_CONFIRM = 10      # consecutive faceless motion frames before logging
MAX_MISSED = 15          # frames a track survives without a detection
ENROLL_SAMPLES = 70      # embeddings captured per enrollment (spread over the prompts)
LOG_COOLDOWN_S = 30      # don't re-log the same identity more often than this

SETTINGS_JSON = "settings.json"
SETTINGS_HELP = {
    "CAMERA_INDEX": "Camera number (0 = built-in, 1 = first USB cam...)",
    "DET_SCORE": "Face detector confidence 0-1. Lower finds more faces (and more false ones)",
    "MATCH_THRESHOLD": "Recognition strictness 0-1. Raise if strangers match, lower if people are missed",
    "HOLD_FRAMES": "Frames to keep a name through odd expressions/blur (~20 frames = 1 s)",
    "UNKNOWN_CONFIRM": "Frames a face must stay unrecognised before logged as Unknown",
    "MOTION_RATIO": "Fraction of picture that must change to count as motion (0.02 = 2%)",
    "MOTION_CONFIRM": "Frames of faceless motion before it is logged",
    "MAX_MISSED": "Frames a tracked face survives when the detector loses it",
    "ENROLL_SAMPLES": "Face samples captured per enrollment",
    "LOG_COOLDOWN_S": "Seconds before the same person is logged again",
}
DEFAULTS = {k: globals()[k] for k in SETTINGS_HELP}


def load_settings():
    try:
        with open(SETTINGS_JSON) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return
    for k, v in saved.items():
        if k in DEFAULTS:
            globals()[k] = type(DEFAULTS[k])(v)


def slot_at(rows, dt):
    """Return the timetable row active at dt, or None if lab is free."""
    day, hm = DAYS[dt.weekday()], dt.strftime("%H:%M")
    for r in rows:
        if r["Day"] == day and r["Start"] <= hm < r["End"]:
            return r
    return None


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


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            print("Downloading", url)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)


class LabMonitor:
    def __init__(self, root):
        self.root = root
        root.title("Lab Occupancy Monitor")
        root.geometry("1280x640")
        os.makedirs(SNAP_DIR, exist_ok=True)
        os.makedirs(FACE_DB_DIR, exist_ok=True)
        os.makedirs(ATT_DIR, exist_ok=True)

        self.timetable = read_csv(TIMETABLE_CSV)
        self.names = {r["Enrollment"]: r["Name"] for r in read_csv(STUDENTS_CSV)}
        self.detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), DET_SCORE, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")
        self._load_db()
        self.tracks = []
        self.motion_frames = 0
        self.enrolling = None
        self.att = None
        self.last_logged = {}
        self.bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.cam_index = CAMERA_INDEX
        self.cam = cv2.VideoCapture(CAMERA_INDEX)

        self._build_ui()
        self._refresh_timetable()
        self._refresh_users()
        for r in read_csv(LOG_CSV)[-200:]:
            if not self.log_tree.exists(r["Snapshot"]):
                self.log_tree.insert("", 0, iid=r["Snapshot"],
                                     values=(r["Date"], r["Time"], r["Name"], r["Enrollment"], r["Confidence"]))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

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

    # ---------- UI ----------
    def _build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side="left", fill="both", padx=8, pady=8)
        right = tk.Frame(self.root)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        bar = tk.Frame(left)
        bar.pack(fill="x", pady=(0, 4))
        big = {"font": ("Segoe UI", 10, "bold"), "padx": 8}
        tk.Button(bar, text="Enroll person", bg="royal blue", fg="white", command=self.enroll_dialog, **big).pack(side="left")
        tk.Button(bar, text="Attendance", bg="green4", fg="white",
                  command=lambda: self.tabs.select(self.att_tab), **big).pack(side="left", padx=6)
        tk.Label(bar, text="Mode:").pack(side="left", padx=(10, 0))
        self.mode = tk.StringVar(value="Auto (timetable)")
        ttk.Combobox(bar, textvariable=self.mode, state="readonly", width=16,
                     values=["Auto (timetable)", "Always alert", "Off"]).pack(side="left", padx=4)
        tk.Button(bar, text="Settings", command=self.settings_dialog, **big).pack(side="left")
        tk.Button(bar, text="Snapshots", command=lambda: os.startfile(os.path.abspath(SNAP_DIR))).pack(side="right")

        self.status = tk.Label(left, font=("Segoe UI", 15, "bold"), fg="white", width=48, height=2)
        self.status.pack(fill="x")
        self.video = tk.Label(left, bg="black", width=640, height=480)
        self.video.pack()
        self.info = tk.Label(left, anchor="w", fg="grey30")
        self.info.pack(fill="x")

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        tt_tab, self.att_tab, log_tab = tk.Frame(self.tabs), tk.Frame(self.tabs), tk.Frame(self.tabs)
        self.tabs.add(tt_tab, text="Timetable")
        self.tabs.add(self.att_tab, text="Attendance")
        self.tabs.add(log_tab, text="Alert log")
        users_tab = tk.Frame(self.tabs)
        self.tabs.add(users_tab, text="Users")

        # Users
        cols = ("ID", "Name", "Face samples")
        self.user_tree = ttk.Treeview(users_tab, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.user_tree.heading(c, text=c)
            self.user_tree.column(c, width=120)
        self.user_tree.pack(fill="both", expand=True)
        ub = tk.Frame(users_tab)
        ub.pack(fill="x", pady=4)
        tk.Button(ub, text="Add user", command=self.enroll_dialog).pack(side="left")
        tk.Button(ub, text="Add face samples", command=self.add_samples).pack(side="left", padx=4)
        tk.Button(ub, text="Rename", command=self.rename_user).pack(side="left")
        tk.Button(ub, text="Delete", command=self.delete_user).pack(side="left", padx=4)

        # Attendance
        self.att_title = tk.Label(self.att_tab, font=("Segoe UI", 12, "bold"), anchor="w")
        self.att_title.pack(fill="x")
        cols = ("Enrollment", "Name", "Status", "Time")
        self.att_tree = ttk.Treeview(self.att_tab, columns=cols, show="headings")
        for c in cols:
            self.att_tree.heading(c, text=c)
            self.att_tree.column(c, width=100)
        self.att_tree.tag_configure("Present", background="#c8f7c5")
        self.att_tree.tag_configure("Absent", background="#f7d0d0")
        self.att_tree.pack(fill="both", expand=True)
        tk.Button(self.att_tab, text="Open attendance sheets",
                  command=lambda: os.startfile(os.path.abspath(ATT_DIR))).pack(anchor="e", pady=4)

        # Timetable
        self.tt_tree = ttk.Treeview(tt_tab, columns=TT_FIELDS, show="headings")
        for c in TT_FIELDS:
            self.tt_tree.heading(c, text=c)
            self.tt_tree.column(c, width=90)
        self.tt_tree.pack(fill="both", expand=True)
        form = tk.Frame(tt_tab)
        form.pack(fill="x", pady=4)
        self.tt_vars = {}
        for c in TT_FIELDS:
            tk.Label(form, text=c).pack(side="left")
            v = tk.StringVar(value={"Day": "Mon", "Start": "09:00", "End": "10:00"}.get(c, ""))
            w = (ttk.Combobox(form, textvariable=v, values=DAYS, width=5, state="readonly") if c == "Day"
                 else tk.Entry(form, textvariable=v, width=8 if c in ("Start", "End") else 12))
            w.pack(side="left", padx=(0, 6))
            self.tt_vars[c] = v
        btns = tk.Frame(tt_tab)
        btns.pack(fill="x")
        tk.Button(btns, text="Add slot", command=self.add_slot).pack(side="left")
        tk.Button(btns, text="Delete selected", command=self.delete_slot).pack(side="left", padx=4)

        # Detection log
        cols = ("Date", "Time", "Name", "Enrollment", "Confidence")
        self.log_tree = ttk.Treeview(log_tab, columns=cols, show="headings")
        for c in cols:
            self.log_tree.heading(c, text=c)
            self.log_tree.column(c, width=90)
        self.log_tree.pack(fill="both", expand=True)
        self.log_tree.bind("<Double-1>", self.open_snapshot)
        tk.Label(log_tab, text="Double-click a detection to view its snapshot", fg="grey40").pack(anchor="w")

    def enroll_dialog(self):
        if self.enrolling:
            return
        d = tk.Toplevel(self.root)
        d.title("Enroll person")
        d.transient(self.root)
        d.grab_set()
        enr, name = tk.StringVar(), tk.StringVar()
        tk.Label(d, text="ID / Enrollment").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        e = tk.Entry(d, textvariable=enr, width=24)
        e.grid(row=0, column=1, padx=8)
        tk.Label(d, text="Name").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        tk.Entry(d, textvariable=name, width=24).grid(row=1, column=1, padx=8)
        tk.Label(d, text="Existing ID adds more samples (name can be left empty).\n"
                         "Follow the prompts on the main screen; one face in view only.",
                 fg="grey30", justify="left").grid(row=2, column=0, columnspan=2, padx=8)

        def start():
            if self.start_enroll(enr.get().strip(), name.get().strip()):
                d.destroy()
        tk.Button(d, text="Start capture", bg="royal blue", fg="white", command=start).grid(
            row=3, column=0, columnspan=2, pady=8)
        e.focus_set()
        d.bind("<Return>", lambda _e: start())

    def _refresh_attendance(self):
        if not self.att:
            self.att_title.config(text="No class scheduled right now")
        else:
            s = self.att["slot"]
            self.att_title.config(text=f"{s['Subject']} ({s['Faculty']}) {self.att['date']} {s['Start']}-{s['End']}   "
                                       f"Present {len(self.att['present'])}/{len(set(self.labels) | set(self.att['present']))}")
        self.att_tree.delete(*self.att_tree.get_children())
        present = self.att["present"] if self.att else {}
        for enr in sorted(set(self.labels) | set(present)):
            status = "Present" if enr in present else "Absent"
            self.att_tree.insert("", "end", values=(enr, self.name_of(enr), status if self.att else "-",
                                                    present.get(enr, "")), tags=(status,) if self.att else ())

    def update_attendance(self, slot, now, people):
        """Switch sheet when the class changes, then mark recognised people present."""
        path = slot and os.path.join(ATT_DIR, f"{safe(slot['Subject'])}_{now:%Y-%m-%d}_{slot['Start'].replace(':', '')}.csv")
        if path != (self.att and self.att["path"]):
            self.att = path and {"path": path, "slot": slot, "date": f"{now:%Y-%m-%d}",
                                 "present": {r["Enrollment"]: r["Time"] for r in read_csv(path)}}
            self._refresh_attendance()
        if not self.att:
            return
        new = False
        for name, enr, _ in people:
            if enr and enr not in self.att["present"]:
                self.att["present"][enr] = f"{now:%H:%M:%S}"
                append_csv(path, ["Enrollment", "Name", "Date", "Time"], [enr, name, f"{now:%Y-%m-%d}", f"{now:%H:%M:%S}"])
                new = True
        if new:
            self._refresh_attendance()

    def _refresh_timetable(self):
        self.timetable.sort(key=lambda r: (DAYS.index(r["Day"]), r["Start"]))
        self.tt_tree.delete(*self.tt_tree.get_children())
        for r in self.timetable:
            self.tt_tree.insert("", "end", values=[r[c] for c in TT_FIELDS])

    def _save_timetable(self):
        with open(TIMETABLE_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TT_FIELDS)
            w.writeheader()
            w.writerows(self.timetable)
        self._refresh_timetable()

    def add_slot(self):
        row = {c: v.get().strip() for c, v in self.tt_vars.items()}
        try:
            start = datetime.datetime.strptime(row["Start"], "%H:%M")
            end = datetime.datetime.strptime(row["End"], "%H:%M")
        except ValueError:
            return messagebox.showerror("Timetable", "Start/End must be HH:MM (24h)")
        if end <= start:
            return messagebox.showerror("Timetable", "End must be after Start")
        row["Start"], row["End"] = start.strftime("%H:%M"), end.strftime("%H:%M")
        self.timetable.append(row)
        self._save_timetable()

    def delete_slot(self):
        sel = {self.tt_tree.index(i) for i in self.tt_tree.selection()}
        self.timetable = [r for i, r in enumerate(self.timetable) if i not in sel]
        self._save_timetable()

    def open_snapshot(self, _event):
        snap = self.log_tree.focus()
        if snap and os.path.exists(snap):
            os.startfile(os.path.abspath(snap))

    # ---------- Enrollment ----------
    def start_enroll(self, enr, name):
        if not enr.isalnum():
            messagebox.showerror("Enroll", "ID must be letters/digits only")
            return False
        if not name and enr not in self.names:
            messagebox.showerror("Enroll", "Name required for a new ID")
            return False
        self.enrolling = {"enr": enr, "name": name or self.names[enr], "samples": [], "frame": 0}
        return True

    def enroll_step(self, frame, faces):
        e = self.enrolling
        e["frame"] += 1
        n = len(e["samples"])
        prompt = ENROLL_PROMPTS[min(n * len(ENROLL_PROMPTS) // ENROLL_SAMPLES, len(ENROLL_PROMPTS) - 1)]
        if len(faces) != 1:
            self.status.config(text=f"Enrolling {e['name']}: exactly one face in view please", bg="dark orange")
            return
        if e["frame"] % 3 == 0:
            e["samples"].append(self.embed(frame, faces[0]))
        self.status.config(text=f"Enrolling {e['name']}: {prompt}  ({n}/{ENROLL_SAMPLES})", bg="royal blue")
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
        self._refresh_attendance()
        self._refresh_users()
        # deferred so the camera loop keeps running while the box is open
        self.root.after(0, lambda: messagebox.showinfo("Enroll", f"{e['name']} saved with {len(samples)} samples"))

    # ---------- Users ----------
    def save_students(self):
        now = datetime.datetime.now()
        old = {r["Enrollment"]: r for r in read_csv(STUDENTS_CSV)}
        rows = [{**old.get(e, {"Date": f"{now:%Y-%m-%d}", "Time": f"{now:%H:%M:%S}"}), "Enrollment": e, "Name": n}
                for e, n in self.names.items()]
        with open(STUDENTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Enrollment", "Name", "Date", "Time"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def _refresh_users(self):
        self.user_tree.delete(*self.user_tree.get_children())
        for enr in sorted(set(self.names) | set(self.labels)):
            n = self.labels.count(enr)
            self.user_tree.insert("", "end", iid=enr, values=(enr, self.name_of(enr), n or "0 (enroll face!)"))

    def selected_user(self):
        sel = self.user_tree.selection()
        if not sel:
            messagebox.showinfo("Users", "Select a user first")
        return sel[0] if sel else None

    def add_samples(self):
        enr = self.selected_user()
        if enr and not self.enrolling:
            self.start_enroll(enr, self.name_of(enr))

    def rename_user(self):
        enr = self.selected_user()
        if not enr:
            return
        name = simpledialog.askstring("Rename", f"New name for {enr}:", initialvalue=self.name_of(enr), parent=self.root)
        if name and name.strip():
            self.names[enr] = name.strip()
            self.save_students()
            self._refresh_users()
            self._refresh_attendance()

    def delete_user(self):
        enr = self.selected_user()
        if not enr or not messagebox.askyesno("Delete", f"Delete {self.name_of(enr)} ({enr}) and their face data?\n"
                                                          "Past attendance sheets and logs are kept."):
            return
        path = os.path.join(FACE_DB_DIR, enr + ".npy")
        if os.path.exists(path):
            os.remove(path)
        self.names.pop(enr, None)
        self.save_students()
        self._load_db()
        self.tracks = []
        self._refresh_users()
        self._refresh_attendance()

    # ---------- Settings ----------
    def settings_dialog(self):
        d = tk.Toplevel(self.root)
        d.title("Settings")
        d.transient(self.root)
        d.grab_set()
        vars_ = {}
        for i, (k, help_) in enumerate(SETTINGS_HELP.items()):
            tk.Label(d, text=k.replace("_", " ").title()).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            vars_[k] = tk.StringVar(value=str(globals()[k]))
            tk.Entry(d, textvariable=vars_[k], width=10).grid(row=i, column=1, padx=4)
            tk.Label(d, text=help_, fg="grey30").grid(row=i, column=2, sticky="w", padx=8)

        def save():
            new = {}
            for k, v in vars_.items():
                try:
                    new[k] = type(DEFAULTS[k])(v.get().strip())
                except ValueError:
                    return messagebox.showerror("Settings", f"{k}: must be a {type(DEFAULTS[k]).__name__}", parent=d)
                if new[k] < 0 or (k in ("DET_SCORE", "MATCH_THRESHOLD", "MOTION_RATIO") and new[k] > 1):
                    return messagebox.showerror("Settings", f"{k}: out of range", parent=d)
            globals().update(new)
            with open(SETTINGS_JSON, "w") as f:
                json.dump(new, f, indent=2)
            self.apply_settings()
            d.destroy()

        row = len(SETTINGS_HELP)
        tk.Button(d, text="Reset defaults",
                  command=lambda: [vars_[k].set(str(v)) for k, v in DEFAULTS.items()]).grid(row=row, column=0, pady=8)
        tk.Button(d, text="Save", bg="green4", fg="white", width=10, command=save).grid(row=row, column=1, pady=8)

    def apply_settings(self):
        self.detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), DET_SCORE, 0.3, 5000)
        if CAMERA_INDEX != self.cam_index:
            self.cam.release()
            self.cam_index = CAMERA_INDEX
            self.cam = cv2.VideoCapture(CAMERA_INDEX)

    # ---------- Camera loop ----------
    def embed(self, frame, face):
        f = self.recognizer.feature(self.recognizer.alignCrop(frame, face)).flatten()
        return f / (np.linalg.norm(f) + 1e-9)

    def update_tracks(self, frame, faces):
        old, new = self.tracks, []
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
        self.tracks = new

    def tick(self):
        ok, frame = self.cam.read()
        now = datetime.datetime.now()
        slot = slot_at(self.timetable, now)
        mode = self.mode.get()
        armed = mode == "Always alert" or (mode.startswith("Auto") and slot is None)

        if slot and mode.startswith("Auto"):
            self.status.config(text=f"Scheduled: {slot['Subject']} ({slot['Faculty']}) {slot['Start']}-{slot['End']}", bg="green4")
        elif armed:
            self.status.config(text="ALERT MODE - lab not scheduled", bg="red3")
        else:
            self.status.config(text="Monitoring off", bg="grey40")

        people = []
        if ok:
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            faces = self.detector.detect(frame)[1]
            faces = [] if faces is None else list(faces)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = (self.bg.apply(gray) > 0).mean() > MOTION_RATIO

            if self.enrolling:
                self.enroll_step(frame, faces)
            else:
                self.update_tracks(frame, faces)

            for t in self.tracks if not self.enrolling else []:
                if t["missed"]:
                    continue
                x, y, bw, bh = map(int, t["box"])
                if t["enr"]:
                    label, color = f"{self.name_of(t['enr'])} {t['sim']:.2f}", (0, 200, 0)
                    people.append((self.name_of(t["enr"]), t["enr"], f"{t['sim']:.2f}"))
                elif t["unk"] >= UNKNOWN_CONFIRM:
                    label, color = f"Unknown {t['sim']:.2f}", (0, 0, 255)
                    people.append(("Unknown", "", f"{t['sim']:.2f}"))
                else:
                    label, color = "Checking...", (0, 220, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for face in faces if self.enrolling else []:
                x, y, bw, bh = map(int, face[:4])
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 140, 0), 3)

            self.motion_frames = self.motion_frames + 1 if motion and not self.tracks else 0
            if self.motion_frames >= MOTION_CONFIRM:
                people.append(("Unknown (no face)", "", ""))
            cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if armed and people and not self.enrolling:
                self.status.config(text="INTRUSION: " + ", ".join(p[0] for p in people), bg="red4")
                for p in people:
                    self.log(now, frame, *p)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 480))
            self.video.imgtk = ImageTk.PhotoImage(img)
            self.video.config(image=self.video.imgtk)
        self.update_attendance(slot if mode != "Off" else None, now, people)
        self.info.config(text=("" if ok else "Camera not available | ")
                         + f"{len(set(self.labels))} people enrolled, {len(self.labels)} samples")
        self.root.after(30, self.tick)

    def log(self, now, frame, name, enr, conf):
        if (now - self.last_logged.get(name, datetime.datetime.min)).total_seconds() < LOG_COOLDOWN_S:
            return
        self.last_logged[name] = now
        d, t = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        snap = os.path.join(SNAP_DIR, f"{d}_{t.replace(':', '-')}_{safe(name)}.jpg")
        cv2.imwrite(snap, frame)
        append_csv(LOG_CSV, ["Date", "Time", "Name", "Enrollment", "Confidence", "Snapshot"],
                   [d, t, name, enr, conf, snap])
        if not self.log_tree.exists(snap):
            self.log_tree.insert("", 0, iid=snap, values=(d, t, name, enr, conf))
        self.root.bell()

    def close(self):
        self.cam.release()
        self.root.destroy()


def selftest():
    tt = [{"Day": "Fri", "Start": "09:00", "End": "10:30", "Subject": "OOP", "Faculty": "X"}]
    fri = datetime.datetime(2026, 10, 16)  # a Friday
    assert slot_at(tt, fri.replace(hour=9))["Subject"] == "OOP"
    assert slot_at(tt, fri.replace(hour=10, minute=29)) is not None
    assert slot_at(tt, fri.replace(hour=10, minute=30)) is None   # end is exclusive
    assert slot_at(tt, fri.replace(hour=8, minute=59)) is None
    assert slot_at(tt, datetime.datetime(2026, 10, 15, 9, 30)) is None  # Thursday

    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1) < 1e-6
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0
    e = np.eye(3, dtype=np.float32)
    assert best_match(e, ["a", "b", "c"], np.array([0.1, 0.9, 0.1], np.float32))[0] == "b"
    assert best_match(np.zeros((0, 3)), [], e[0]) == ("", 0.0)
    assert safe("a/b:c") == "a_b_c"
    print("selftest ok")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif sys.argv[1:] == ["--download"]:
        ensure_models()
    else:
        load_settings()
        root = tk.Tk()
        try:
            ensure_models()
            LabMonitor(root)
        except Exception as ex:  # run.bat has no console, so show startup errors
            messagebox.showerror("Lab Monitor", f"Startup failed:\n{ex}")
            raise
        root.mainloop()
