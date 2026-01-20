# 🚗🔌 EVChargingModel

A Python-based Electric Vehicle (EV) Charging Model with an API layer to run inference using a pre-trained Q-table. This project simulates and serves EV charging-related predictions and can be integrated into larger EV charging platforms or research workflows.

<img width="1461" height="992" alt="EVParking drawio (1)" src="https://github.com/user-attachments/assets/fa18117c-51d1-47c9-b755-6c8fadf6d1ba" />
---


## 📌 Overview

EVChargingModel exposes a simple Flask API that loads a trained Q-table model and returns predictions based on input parameters. It is useful for:

* Simulating EV charging decisions
* Testing EV charging strategies
* Integrating EV charging intelligence into backend systems

---

## 🧠 Features

* Pre-trained EV charging model (`q_table_ev_charging.pkl`)
* Flask-based REST API
* Modular inference helper for easy extension
* Clean and understandable project structure

---

## 🖼️ Workflow Diagram

The following diagram represents the end-to-end workflow of the EVChargingModel:

![EV Charging Workflow](./assets/ev-charging-workflow.png)

### 🔄 Workflow Steps

1. **Client / User** sends a request
2. **Flask API (`api.py`)** receives the request
3. **Inference Helper (`inference_helper.py`)** processes input
4. **Q-table Model** performs prediction
5. **API Response** is returned to the client

---

## 📁 Project Structure

```
EVChargingModel/
│
├── api.py                  # Flask API endpoints
├── app.py                  # Application entry point
├── inference_helper.py     # Model loading and inference logic
├── q_table_ev_charging.pkl # Pre-trained Q-table model
├── requirements.txt        # Python dependencies
├── assets/
│   └── ev-charging-workflow.png
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.8 or higher
* pip

Check your Python version:

```
python --version
```

---

### 🧰 Installation

1. Clone the repository:

```
git clone https://github.com/chinmay97531/EVChargingModel.git
cd EVChargingModel
```

2. (Optional) Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start the Flask API server:

```
python api.py
```

The server will start at:

```
http://127.0.0.1:5000/
```

---

## 🔗 API Usage

You can test the API using a browser, Postman, or curl.

Example:

```
curl http://127.0.0.1:5000/predict
```

(Modify endpoints and parameters as per your implementation.)

---

## 🧪 Model Inference

* The trained model is stored in `q_table_ev_charging.pkl`
* Inference logic is implemented in `inference_helper.py`
* The inference module can be reused independently in other Python projects

---

## 🤝 Contributing

Contributions are welcome.

* Fork the repository
* Create a feature branch
* Commit your changes
* Open a pull request

---

## 📜 License

This project is open-source and available for educational and development purposes.

---
