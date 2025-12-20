from flask import Flask, request, jsonify
from ev_q_model import EVChargingQModel
from flask_cors import CORS
import numpy as np
import os
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    # By default skip training at startup to keep the dev server responsive.
    # To force training on startup set environment variable `EVMODEL_TRAIN=1`.
    do_train = os.getenv('EVMODEL_TRAIN', '0') == '1'
    if not do_train:
        print('Skipping training on startup (dev). Set EVMODEL_TRAIN=1 to enable.')
        model.save_model(model_path)
    else:
        # Use a smaller number of episodes by default to avoid long startup during development.
        train_episodes = int(os.getenv('EVMODEL_EPISODES', '50'))
        print(f"Training model for {train_episodes} episodes (dev mode)")
        model.train(env, episodes=train_episodes)
        model.save_model(model_path)
        print("Training completed and model saved.")

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
