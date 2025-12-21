from ev_q_model import EVChargingQModel
from dummy_env import DummyEnv

def main():
    # Initialize
    env = DummyEnv()
    model = EVChargingQModel()

    # Train
    model.train(env_function=env, episodes=100)

    # Save the model
    model.save_model('q_model.pkl')
    print("Model saved to q_model.pkl")

    # Plot rewards
    model.plot_rewards()

    # Predict
    test_state = (12, 1, 3)
    action = model.predict(test_state)
    print(f"Predicted action for state {test_state}: {action}")

if __name__ == "__main__":
    main()
