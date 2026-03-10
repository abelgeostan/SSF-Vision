# SSF-Vision 👁️🛡️
**Intelligent Surveillance System for Real-Time Threat Detection & Tracking**

![SSF-Vision Overview](https://img.shields.io/badge/Status-Active-brightgreen) ![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)

**SSF-Vision** is a comprehensive, self-hosted smart surveillance and security framework. Leveraging state-of-the-art computer vision models and machine learning techniques, it transforms standard camera feeds into an intelligent monitoring system capable of anomaly detection, automated logging, and multi-camera person re-identification (ReID).

The entire system is orchestrated through a central dashboard (`main.py`), making it easy to manage multiple surveillance feeds and security tools from a single interface.

---

## 🚀 Key Features

* **Centralized Dashboard (`main.py`):** Acts as the command center to launch and manage all other security modules (anomaly detection, tracking, etc.) from one unified interface.
* **Real-Time Object & Threat Detection:** Powered by **YOLOv8** (`yolov8n.pt`), ensuring lightning-fast and highly accurate detection of people, vehicles, and objects.
* **Behavioral Anomaly Detection (`anomaly.py`):** AI-driven logic to monitor feeds for suspicious activities, unauthorized access, or unusual behavioral patterns.
* **Person Re-Identification & Tracking (`reid.py` & `suspect_finder.py`):** Advanced tracking that extracts visual features to re-identify and track a specific suspect across different camera feeds or disconnected frames.
* **Hardware Acceleration via DirectML (`test_directml.py`):** Experimental support for Microsoft DirectML, allowing the system to utilize GPU acceleration across diverse hardware (AMD, Intel, NVIDIA) on Windows.
* **Automated Security Logging:** All security events, identified threats, and anomalies are automatically logged with timestamps and metadata into `security_logs.csv` and the `analytics_output/` directory for post-event analysis.

---

## 🛠️ Frameworks & Technologies Used

* **Computer Vision & AI:**
  * [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - For real-time object detection and tracking.
  * [OpenCV (cv2)](https://opencv.org/) - For video capture, frame processing, and image manipulations.
  * [PyTorch / Torchvision](https://pytorch.org/) - Backend for deep learning feature extraction and model inference.
* **Hardware Acceleration:**
  * **DirectML** - Extends hardware-accelerated AI to a broader range of GPUs via `onnxruntime-directml` or PyTorch-DirectML.
* **Data Handling & Analytics:**
  * **Pandas / CSV** - For logging security events and handling structured data.
  * **NumPy / SciPy** - For calculating cosine similarities and spatial distances in the ReID modules.
* **Dashboard & Orchestration:**
  * Python's standard OS/Subprocess or GUI libraries (Tkinter/PyQt/Streamlit) utilized in `main.py` to route and execute the modular scripts.

---

## 📂 Project Architecture

```text
SSF-Vision/
│
├── main.py                  # The main dashboard & entry point to the system
├── anomaly.py               # Core engine for detecting anomalous events
├── suspect_finder.py        # Logic to locate a specific individual in historical/live feeds
├── reid.py / reid1.0.py     # Person Re-Identification (ReID) implementation
├── surveillance_emy.py      # Camera/Feed-specific surveillance configuration 1
├── surveillance_nashwa.py   # Camera/Feed-specific surveillance configuration 2
├── test_directml.py         # Hardware acceleration diagnostics script
├── yolov8n.pt               # Pre-trained YOLOv8 Nano model weights
├── security_logs.csv        # Database of automated security alerts and event logs
├── requirements.txt         # Project dependencies
├── model/                   # Directory containing additional specific ML models
└── analytics_output/        # Directory for generated analytics, snapshots, and reports

```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/Mr-Joseph-Jo/SSF-Vision.git](https://github.com/Mr-Joseph-Jo/SSF-Vision.git)
cd SSF-Vision

```


2. **Create a Virtual Environment (Recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


*(Note: If you plan to use DirectML on Windows, ensure you install the specific `torch-directml` or `onnxruntime-directml` packages as required by your environment).*

---

## 💻 Usage

### Starting the Dashboard

To start the primary dashboard which controls the entire surveillance suite, run:

```bash
python main.py

```

From the dashboard, you can route to individual feeds, initialize the ReID system, or toggle anomaly detection.

### Running Modules Independently

If you prefer to run specific modules directly from the CLI for testing:

* **Test GPU/DirectML Capabilities:**
```bash
python test_directml.py

```


* **Run Anomaly Detection System:**
```bash
python anomaly.py

```


* **Run Suspect Search/ReID:**
```bash
python suspect_finder.py

```



---

## 📊 Analytics & Event Logging

SSF-Vision relies heavily on accountability and auditing.

* Any time a rule is broken or a suspect is flagged, a record is appended to **`security_logs.csv`**.
* Visual proof (such as bounding box snapshots or tracking paths) is saved in the **`analytics_output/`** folder.
This allows security personnel to review exact timestamps and visual evidence of events without watching hours of footage.

---

## 🤝 Contributing

Contributions are always welcome! If you have suggestions for improving model accuracy, optimizing the dashboard, or adding new deployment methods (like Docker), please fork the repository and create a pull request.

For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This system relies on open-source frameworks. Please review the licenses of YOLOv8 (AGPL-3.0) and other utilized libraries. Ensure you comply with all local privacy, surveillance, and data-protection laws (e.g., GDPR) prior to deploying this software in private or public spaces.

```

```