# /kaggle/working/inference_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import time

from inference_helper import load_q_table, discretize_inputs, get_action_for_state, ACTIONS

app = Flask(__name__)
CORS(app)  # dev: allow all origins. For production lock origins down.

QTABLE_PATH = "q_table_ev_charging.pkl"
q_table = None

def ensure_q_table():
    global q_table
    if q_table is None:
        q_table = load_q_table(QTABLE_PATH)

@app.route("/infer", methods=["POST", "OPTIONS"])
def infer():
    try:
        ensure_q_table()

        # safer JSON parsing: don't raise HTTP 400 inside werkzeug; handle missing/invalid JSON
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "message": "Request body must be valid JSON and Content-Type must be application/json"}), 400

        # required fields with validation
        try:
            current_soc = float(data.get("current_soc"))
        except (TypeError, ValueError):
            return jsonify({"status":"error","message":"`current_soc` required and must be numeric"}), 400

        try:
            required_soc = float(data.get("required_soc"))
        except (TypeError, ValueError):
            return jsonify({"status":"error","message":"`required_soc` required and must be numeric"}), 400

        preference = data.get("preference", None)

        # hours_remaining parsing
        hours_remaining = None
        if "hours_remaining" in data and data["hours_remaining"] is not None:
            try:
                hours_remaining = float(data["hours_remaining"])
            except (TypeError, ValueError):
                return jsonify({"status":"error","message":"`hours_remaining` must be numeric if provided"}), 400
        elif "plug_out_time" in data and data["plug_out_time"]:
            try:
                plug = str(data["plug_out_time"])
                parts = plug.split(":")
                h = int(parts[0]); m = int(parts[1]) if len(parts)>1 else 0
                now = time.localtime()
                now_minutes = now.tm_hour*60 + now.tm_min
                plug_minutes = h*60 + m
                hrs = max(0, (plug_minutes - now_minutes)/60.0)
                hours_remaining = hrs
            except Exception:
                hours_remaining = 2.0
        else:
            hours_remaining = 2.0

        # optional extras
        solar_kw = data.get("solar_kw", None)
        price = data.get("price", None)
        station_battery_kwh = data.get("station_battery_kwh", None)

        if "time_slot" in data:
            try:
                time_slot = int(data["time_slot"])
            except (TypeError, ValueError):
                time_slot = int(time.localtime().tm_hour)
        else:
            time_slot = int(time.localtime().tm_hour)

        state = discretize_inputs(current_soc, required_soc, hours_remaining, time_slot,
                                  solar_kw=solar_kw, price=price, station_battery_kwh=station_battery_kwh)

        action_id, qvals = get_action_for_state(q_table, state, preference=preference)

        # estimate time to complete (approx)
        mode, source = ACTIONS[action_id]
        rate = 7.0 if mode == "slow" else 22.0 if mode == "fast" else 0.0
        soc_gap = max(required_soc - current_soc, 0.0)
        if rate > 0:
            increments_per_hour = (rate * 0.5)  # keep your quantization assumption if that's how you trained
            est_hours = (soc_gap / increments_per_hour) if increments_per_hour > 0 else None
        else:
            est_hours = None

        result = {
            "state": state,
            "recommended_action_id": int(action_id),
            "recommended_action": ACTIONS[int(action_id)],
            "q_values": qvals,
            "estimated_hours_to_complete": est_hours
        }
        return jsonify({"status":"ok","result": result})
    except Exception as e:
        app.logger.exception("Internal server error in /infer")
        traceback.print_exc()
        return jsonify({"status":"error","message":"internal server error","detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
