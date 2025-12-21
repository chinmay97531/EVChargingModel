from flask import Flask, request, jsonify
from flask_cors import CORS
from ev_q_model import EVChargingQModel
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = EVChargingQModel()
model_path = 'q_model.pkl'
if os.path.exists(model_path):
    model.load_model(model_path)
else:
    # If no saved model, train a new one (this might take time)
    from dummy_env import DummyEnv
    env = DummyEnv()
    model.train(env_function=env, episodes=1000)
    model.save_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        hour = data.get('hour')
        demand = data.get('demand')
        solar = data.get('solar')

        if hour is None or demand is None or solar is None:
            return jsonify({'error': 'hour, demand, and solar are required'}), 400

        # Assuming the state is (hour, demand, solar)
        state = (int(hour), int(demand), int(solar))
        action = model.predict(state)

        return jsonify({'action': int(action), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)