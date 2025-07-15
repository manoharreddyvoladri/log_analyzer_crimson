## 📊 Log Analyzer Crimson

A Django-based web application for **uploading, analyzing, and forecasting log data** using LSTM models. It provides real-time dashboard insights, visualizations, and predictions on system log messages.

---

### 📁 Project Structure

```bash
LogAnalaysis_CIT/
├── .vscode/
├── data/
├── LogFile_Container/
├── Logfile_Operations/
├── logs/
├── models/
├── Myapp/                  # Main Django app
├── PhaseI/                 # LSTM and forecasting logic
├── db.sqlite3              # Default database
├── manage.py               # Django management script
└── requirement.txt         # Python dependencies
```

---

### 🚀 Features

* 🔍 Upload and parse system log files.
* 📈 Forecast errors using LSTM models (TensorFlow/Keras).
* 📊 Visualizations of logs (INFO, WARNING, ERROR).
* 🧠 Keyword-based meter values and error tracking.
* 📁 Session-based log storage and analysis.
* 📬 Forgot password with OTP mail verification.

---

### ⚙️ Setup & Installation

#### ✅ Prerequisites

* Python **3.11.3**
* Git
* Virtual Environment (`venv`)

#### 📦 Install Dependencies

```bash
# Clone the repository
git clone https://github.com/manoharreddyvoladri/log_analyzer_crimson.git
cd log_analyzer_crimson

# Setup virtual environment
python -m venv .venv
source .venv/Scripts/activate  # For Windows
# Or use: source .venv/bin/activate  # For Linux/macOS

# Install required packages
pip install --upgrade pip
pip install -r requirement.txt
```

---

### ⚙️ Running the Project

```bash
# Apply migrations
python manage.py makemigrations
python manage.py migrate

# Run the Django server
python manage.py runserver
```

Now open your browser and visit:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 📤 How to Use

1. **Login/Register** from the home page.
2. Upload a log file via the `Prediction` tab.
3. View predictions and logs on the `Results` and `Landing` pages.
4. Train new LSTM models using the `Train` button.
5. Visualize error trends month-wise.
6. Check real-time data in the `Healthy` section.

---

### 🧪 Sample Log Format

```
2023-09-12 10:33:56,123 - ERROR [MainThread] - Application crashed due to null pointer.
```

---

### 🧠 Tech Stack

* **Backend**: Django, SQLite
* **Frontend**: HTML, Bootstrap, jQuery
* **ML Model**: LSTM (Keras + TensorFlow)
* **Data**: pandas, numpy
* **Others**: CSRF-secured views, Session-based storage, OTP mailer

---

### 📂 Key Modules

* `PhaseI/views.py`: All forecasting, preprocessing, training.
* `Myapp/models.py`: User management and OTP verification.
* `templates/`: Frontend HTML files.
* `settings.py`: Define `KEYWORD_FILE_PATH` and `CSV_FILE_PATH`.

---

### 🛠 Developer Notes

* Ensure that `KEYWORD_FILE_PATH` and `CSV_FILE_PATH` are defined in `settings.py`.
* All models are saved in `PhaseI/models/`.
* TensorFlow ONEDNN warnings are disabled for speed.
* CSV logs are used for dynamic dashboards.

---

