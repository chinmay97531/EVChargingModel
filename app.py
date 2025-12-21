from flask import Flask, request, jsonify
from ev_q_model import EVChargingQModel
from flask_cors import CORS
import numpy as np
import os
import pickle

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"])  # Enable CORS for frontend and backend

# Load your trained model (or create a new one)
model_path = 'q_model.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        q_table = pickle.load(f)
    model = EVChargingQModel()
    model.q_table = q_table
else:
    model = EVChargingQModel()
    # Optionally train it if not loaded
    from dummy_env import DummyEnv
    env = DummyEnv()
    model.train(env, episodes=500)
    model.save_model(model_path)

@app.route('/predict', methods=['POST'])
def predict_action():
    data = request.get_json()

    try:
        hour = int(data['hour'])
        demand = int(data['demand'])
        solar = int(data['solar'])

        action = model.predict((hour, demand, solar))

        return jsonify({
            'action': int(action),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Expose to local network
