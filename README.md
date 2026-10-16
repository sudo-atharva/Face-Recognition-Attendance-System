# Lab Occupancy Monitor + Face Recognition Attendance

Watches a lab with a webcam:
- **Booked slot (timetable):** recognised people are marked **present** for that class.
- **No class booked:** **alert mode**. Any presence (face or motion) is identified, timestamped and saved with a snapshot.

Face pipeline: OpenCV YuNet detector + SFace recognizer, with tracking. It copes with expressions (tongue out, laughing), head turns and brief blur far better than the original Haar + LBPH.

### Install & run (Windows)
1. Install Python 3.9+ from python.org (tick **Add python.exe to PATH**).
2. Double-click **`install.bat`**. It creates `venv`, installs packages and downloads the face models.
3. Double-click **`run.bat`**.

Everything else is done in the GUI.

### Using it
| Where | What |
|---|---|
| **Enroll person** button / Users tab → Add user | Enter ID + name, then follow on-screen prompts (straight, turn, tilt, smile, tongue out, talk...). 70 samples are captured. |
| Users tab | Add more face samples, rename, delete users |
| Timetable tab | Add / delete lab slots (Day, Start, End, Subject, Faculty) |
| Attendance tab | Present / absent list for the current class. Sheets are saved in `Attendance/<Subject>_<date>_<start>.csv` |
| Alert log tab | Detections while the lab is not booked. Double-click to open the snapshot (`Intrusions/`) |
| Mode | `Auto (timetable)`, `Always alert`, `Off` |
| Settings | Camera, recognition strictness, motion sensitivity, timings (saved to `settings.json`) |

### Screenshots
(Users, timetable and attendance rows are demo data.)

| | |
|---|---|
| Alert mode: lab not booked, stranger detected<br><img src="screenshots/1_alert_mode_timetable.png" width="480"> | Class running: attendance<br><img src="screenshots/2_class_attendance.png" width="480"> |
| Alert log<br><img src="screenshots/3_alert_log.png" width="480"> | Users<br><img src="screenshots/4_users.png" width="480"> |
| Enroll dialog<br><img src="screenshots/5_enroll_dialog.png" width="300"> | Enrolling with guided prompts<br><img src="screenshots/6_enrolling.png" width="480"> |
| Settings<br><img src="screenshots/7_settings.png" width="480"> | |

### Limits
- Not spoof-proof: a photo or phone screen of an enrolled face can fool it.
- Settings and Users have no password.

---

# Original project: Face-Recognition-Attendance-Management-System
Attendance Management System based on Face Recognition using Python  and OpenCv  

### Sourcerer
<img src="https://avatars.githubusercontent.com/u/84435079?v=4" height="50px" width="50px" alt=""/>

### Code Requirements
- Opencv(`pip install opencv-python`)
- Tkinter(Available in python)
- PIL (`pip install Pillow`)
- Pandas(`pip install pandas`)

### What steps you have to follow??
- Download my Repository 
- Create a `TrainingImage` folder in a project.
- Open a `AMS_Run.py` and change the all paths with your system path
- Run `AMS_Run.py`.

### Project Structure

- After run you need to give your face data to system so enter your ID and name in box than click on `Take Images` button.
- It will collect 200 images of your faces, it save a images in `TrainingImage` folder
- After that we need to train a model(for train a model click on `Train Image` button.
- It will take 5-10 minutes for training(for 10 person data).
- After training click on `Automatic Attendance` ,it can fill attendance by your face using our trained model (model will save in `TrainingImageLabel` )
- it will create `.csv` file of attendance according to time & subject.
- You can store data in database (install wampserver),change the DB name according to your in `AMS_Run.py`.
- `Manually Fill Attendance` Button in UI is for fill a manually attendance (without facce recognition),it's also create a `.csv` and store in a database.

### Screenshots

### Basic UI
<img src="https://github.com/Pragya9ps/Face-Recognition-Attendance-System/blob/main/Screenshot%20(31).png">

### When it Recognises me
<img src="https://github.com/Pragya9ps/Face-Recognition-Attendance-System/blob/main/Screenshot%20(33).png">

### While filling automatic attendance
<img src="https://github.com/Pragya9ps/Face-Recognition-Attendance-System/blob/main/Screenshot%20(38).png">

### Manually attendance filling UI
<img src="https://github.com/Pragya9ps/Face-Recognition-Attendance-System/blob/main/Screenshot%20(35).png">


### Video demo

[Youtube](https://youtu.be/onms2KDOTtY)


### Notes
- It will require high processing power(I have 8 GB RAM)
- Noisy image can reduce the accuracy, so quality of images should be good.


