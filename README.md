# 🚦 Deep Learning Vehical Safety Monitoring System using YOLOv8

## 📌 Project Overview

This project is a Deep Learning based Vehical Safety System that analyzes traffic videos using YOLOv8 and Computer Vision techniques.

The system allows users to upload a traffic video through a Flask web application. The uploaded video is processed frame by frame to detect vehicles such as cars, bikes, buses, and trucks. The system generates a processed output video with bounding boxes, vehicle count, basic speed estimation, danger alerts, and downloadable traffic logs in Excel format.

---

## 🎯 Objectives

- Detect vehicles from traffic/CCTV videos
- Identify cars, bikes, buses, and trucks using YOLOv8
- Generate bounding boxes around detected vehicles
- Estimate vehicle speed using pixel movement
- Detect unsafe vehicle proximity
- Display danger alerts on processed video
- Generate Excel traffic logs
- Provide a Flask-based frontend for user interaction

---

## 🧠 Technologies Used

- Python
- Flask
- YOLOv8
- OpenCV
- Pandas
- NumPy
- HTML / CSS
- Deep Learning
- Computer Vision

---

## ⚙️ Features

- Upload traffic video from local system
- Process the same uploaded video
- Vehicle detection using YOLOv8
- Bounding box visualization
- Vehicle count display
- Basic speed estimation
- Danger detection for close vehicles
- Processed video output
- Download processed video
- Download traffic logs in Excel format

---

## 📁 Project Structure

```text
traffic_project/
│
├── app.py
├── model.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── uploads/
│   └── outputs/
│
└── README.md
````

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Hithashree2004/traffic-monitoring-system.git
cd traffic-monitoring-system
```

### 2. Install Required Packages

```bash
python -m pip install flask ultralytics opencv-python pandas openpyxl
```

### 3. Run the Flask Application

```bash
python app.py
```

### 4. Open in Browser

```text
http://127.0.0.1:5000
```

---

## 🖥️ How It Works

1. User uploads a traffic video.
2. Flask stores the video in the uploads folder.
3. YOLOv8 processes the video frame by frame.
4. Vehicles are detected using bounding boxes.
5. Vehicle count and speed are displayed on the video.
6. Unsafe vehicle distance is marked as danger.
7. Processed video is saved in the outputs folder.
8. Traffic logs are saved in Excel format.
9. User can download both processed video and Excel logs.

---

## 📊 Output

The system provides:

* Processed traffic video
* Vehicle detection results
* Vehicle count
* Speed estimation
* Danger warning
* Excel traffic logs

---

## 📈 Model Evaluation Metrics

YOLOv8 object detection models are evaluated using:

* Precision
* Recall
* F1 Score
* mAP50
* mAP50-95
* Confusion Matrix

---

## 📌 Algorithm

1. Start the Flask application
2. Upload traffic video
3. Read video using OpenCV
4. Extract video frames
5. Apply YOLOv8 detection
6. Filter vehicle classes
7. Draw bounding boxes
8. Estimate speed using pixel displacement
9. Calculate distance between vehicles
10. Detect danger condition
11. Save processed video
12. Generate Excel traffic logs
13. Display and download results

---

## 🔮 Future Enhancements

* Real-time CCTV stream support
* Accurate vehicle tracking with unique IDs
* Advanced lane segmentation
* Number plate recognition
* Email and sound alerts
* SQLite database integration
* Dashboard with graphs and analytics
* Deployment on cloud platform

---

## 👩‍💻 Developed By

**Hithashree K S**

GitHub: [Hithashree2004](https://github.com/Hithashree2004)

---

## 📌 Project Title

**Deep Learning Based Vehical Safety  Monitoring System using YOLOv8**

```
```
