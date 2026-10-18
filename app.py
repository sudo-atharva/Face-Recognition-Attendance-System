"""Lab monitor website. Run with run.bat, or: python app.py [--open]

Open http://localhost:5000 on the lab PC, or http://<lab-pc-ip>:5000 from a
phone/laptop on the same network. The admin password is created on first visit
(only allowed from the lab PC itself).
"""
import base64
import os
import secrets
import socket
import sys
import time
import webbrowser

from flask import Flask, Response, jsonify, redirect, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash

import monitor as m

app = Flask(__name__, static_folder="static")
mon = None  # set in main

LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Lab Monitor Login</title>
<style>body{{font-family:system-ui,sans-serif;background:#f1f3f5;display:grid;place-items:center;height:100vh;margin:0}}
form{{background:#fff;padding:28px;border-radius:12px;box-shadow:0 2px 12px #0002;width:min(320px,90vw)}}
input,button{{width:100%;padding:10px;margin-top:10px;font-size:16px;box-sizing:border-box}}
button{{background:#2563eb;color:#fff;border:0;border-radius:6px}}.e{{color:#c00}}</style></head>
<body><form method="post"><h2>{title}</h2><p>{hint}</p>
<input type="password" name="password" placeholder="Password" autofocus required>
<button>{button}</button><p class="e">{error}</p></form></body></html>"""


def is_local():
    return request.remote_addr in ("127.0.0.1", "::1")


def ok(err=None):
    return jsonify({"error": err} if err else {"ok": True}), 400 if err else 200


@app.before_request
def require_login():
    if request.endpoint != "login" and not session.get("ok"):
        if request.path.startswith("/api/"):
            return jsonify({"error": "login required"}), 401
        return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    pw_hash = m.read_json().get("WEB_PASSWORD_HASH")
    error = ""
    if request.method == "POST":
        pw = request.form.get("password", "")
        if not pw_hash:
            if not is_local():
                error = "Create the password on the lab PC first (http://localhost)"
            elif len(pw) < 6:
                error = "Use at least 6 characters"
            else:
                m.save_json(WEB_PASSWORD_HASH=generate_password_hash(pw))
                session["ok"] = True
                return redirect("/")
        elif check_password_hash(pw_hash, pw):
            session["ok"] = True
            session.permanent = True
            return redirect("/")
        else:
            time.sleep(1)  # slow down guessing
            error = "Wrong password"
    if pw_hash:
        page = LOGIN_PAGE.format(title="Lab Monitor", hint="Admin login", button="Log in", error=error)
    else:
        page = LOGIN_PAGE.format(title="Create admin password", hint="First run: choose the password for this website.",
                                 button="Create", error=error)
    return page


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/video")
@app.route("/video2")
def video():
    view = mon.cam2 if request.path == "/video2" else mon.main

    def frames():
        last = None
        while True:
            j = view.jpeg
            if j is not None and j is not last:
                last = j
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.04)
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def state():
    return jsonify(mon.state())


@app.route("/api/mode", methods=["POST"])
def set_mode():
    return ok(mon.set_mode(request.json.get("mode")))


@app.route("/api/timetable", methods=["GET", "POST"])
def timetable():
    if request.method == "POST":
        return ok(mon.add_slot(request.json))
    return jsonify(mon.timetable)


@app.route("/api/timetable/<int:i>", methods=["DELETE"])
def delete_slot(i):
    return ok(mon.delete_slot(i))


@app.route("/api/users")
def users():
    return jsonify(mon.users())


@app.route("/api/enroll", methods=["POST"])
def enroll():
    return ok(mon.start_enroll(request.json.get("id", ""), request.json.get("name", "")))


@app.route("/api/enroll/cancel", methods=["POST"])
def cancel_enroll():
    return ok(mon.cancel_enroll())


@app.route("/api/users/<enr>", methods=["PATCH", "DELETE"])
def user(enr):
    if request.method == "DELETE":
        return ok(mon.delete_user(enr))
    if "role" in request.json:
        return ok(mon.set_role(enr, request.json["role"]))
    return ok(mon.rename_user(enr, request.json.get("name", "")))


@app.route("/api/log")
def log():
    return jsonify(mon.log())


@app.route("/api/files/<kind>")
def files(kind):
    folder = {"recordings": m.REC_DIR, "sheets": m.ATT_DIR}.get(kind)
    if not folder:
        return ok("unknown list")
    names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    names.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return jsonify([{"name": f, "size": os.path.getsize(os.path.join(folder, f))} for f in names])


@app.route("/files/<kind>/<path:name>")
def get_file(kind, name):
    folder = {"recordings": m.REC_DIR, "sheets": m.ATT_DIR, "snapshots": m.SNAP_DIR}.get(kind)
    if not folder:
        return "not found", 404
    return send_from_directory(os.path.abspath(folder), name, as_attachment=kind == "sheets")


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        values, err = m.parse_settings(request.json)
        if err:
            return ok(err)
        mon.apply_settings(values)
        return ok()
    return jsonify(m.current_settings())


@app.route("/api/cam2/test", methods=["POST"])
def cam2_test():
    """Grab one frame using the (possibly unsaved) WiFi camera fields from the form."""
    values, err = m.parse_settings(request.json)
    if err:
        return err, 400
    cam = [values[k] for k in ("CAM2_TYPE", "CAM2_IP", "CAM2_PORT", "CAM2_USER", "CAM2_PASS", "CAM2_PATH", "CAM2_URL")]
    url = m.build_cam2_url(*cam)
    if not url:
        return "WiFi camera is off", 400
    jpg = m.grab_jpeg(url)
    shown = m.build_cam2_url(*cam[:4], "***" if cam[4] else "", *cam[5:])
    return Response(jpg, mimetype="image/jpeg") if jpg else ("Could not read a frame from " + shown, 502)


@app.route("/api/cameras")
def cameras():
    """Probe camera numbers 0-5 for the main-camera picker, with a thumbnail of each."""
    found = []
    for i in range(6):
        if i == mon.cam_index and mon.cam.isOpened() and mon.main.jpeg is not None:  # in use; can't open twice
            jpg = mon.main.jpeg
        else:
            jpg = m.grab_jpeg(str(i))
        if jpg:
            found.append({"index": i, "in_use": i == mon.cam_index,
                          "thumb": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})
    return jsonify(found)


@app.route("/api/serial/ports")
def serial_ports():
    if m.serial is None:
        return jsonify([])
    from serial.tools import list_ports
    return jsonify([{"port": p.device, "desc": p.description} for p in list_ports.comports()])


@app.route("/api/wake", methods=["POST"])
def wake():
    mon.pir.motion()
    return ok()


@app.route("/api/password", methods=["POST"])
def change_password():
    old, new = request.json.get("old", ""), request.json.get("new", "")
    if not check_password_hash(m.read_json().get("WEB_PASSWORD_HASH", ""), old):
        return ok("Current password is wrong")
    if len(new) < 6:
        return ok("Use at least 6 characters")
    m.save_json(WEB_PASSWORD_HASH=generate_password_hash(new))
    return ok()


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    global mon
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.stdout is None:  # pythonw (run.bat) has no console
        sys.stdout = sys.stderr = open("server.log", "a", buffering=1)
    m.load_settings()
    url = f"http://localhost:{m.WEB_PORT}"
    if port_in_use(m.WEB_PORT):  # already running: just open it
        webbrowser.open(url)
        return
    m.ensure_models()
    app.secret_key = m.read_json().get("SECRET_KEY") or secrets.token_hex(32)
    m.save_json(SECRET_KEY=app.secret_key)
    mon = m.Monitor()
    mon.start()
    if "--open" in sys.argv:
        webbrowser.open(url)
    # ponytail: Flask dev server is fine for a few viewers on a LAN; use waitress if many people watch at once
    app.run(host="0.0.0.0", port=m.WEB_PORT, threaded=True)


if __name__ == "__main__":
    main()
