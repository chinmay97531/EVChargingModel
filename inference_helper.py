# Cell 8: Write inference helper file (inference_helper.py) to /kaggle/working
import pickle
import numpy as np
import os

ACTIONS = {
    0: ("slow", "solar"),
    1: ("fast", "solar"),
    2: ("slow", "battery"),
    3: ("fast", "battery"),
    4: ("slow", "grid"),
    5: ("fast", "grid"),
    6: ("wait", None)
}

# load q-table (path should be adjusted if moved)
def load_q_table(path="q_table_ev_charging.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def discretize_inputs(current_soc, required_soc, hours_remaining, time_slot,
                      solar_kw=None, price=None, station_battery_kwh=None):
    # bins must match training discretization
    def _discretize(val, bins):
        for i,b in enumerate(bins):
            if val <= b:
                return i
        return len(bins)
    soc_gap = max(required_soc - current_soc, 0)
    soc_gap_level = _discretize(soc_gap, [20,50])
    urgency_level = _discretize(hours_remaining, [1,3])
    # solar / price / battery were derived from dataset during training - pass proxies if unknown
    if solar_kw is None: solar_level = 1
    else: solar_level = 0 if solar_kw < 20 else 1 if solar_kw < 60 else 2
    if price is None: grid_price_level = 1
    else: grid_price_level = 0 if price < 0.15 else 1 if price < 0.30 else 2
    if station_battery_kwh is None: station_battery_level = 1
    else: station_battery_level = 0 if station_battery_kwh < 30 else 1 if station_battery_kwh < 70 else 2

    return (int(time_slot), int(solar_level), int(grid_price_level), int(station_battery_level), int(soc_gap_level), int(urgency_level))

def get_action_for_state(q_table, state_tuple, preference=None):
    # preference: "cheapest","fastest","solar","grid" or None
    if state_tuple in q_table:
        qvals = np.array(q_table[state_tuple], dtype=float).copy()
    else:
        # unseen state fallback: return 'wait' action
        qvals = np.zeros(len(ACTIONS), dtype=float)
    # apply small bias for preference
    if preference:
        pref = preference.lower()
        for aid, (mode, source) in ACTIONS.items():
            if pref == "fastest" and mode == "fast":
                qvals[aid] += 2.0
            if pref == "cheapest" and source == "solar":
                qvals[aid] += 1.5
            if pref == "solar" and source == "solar":
                qvals[aid] += 2.0
            if pref == "grid" and source == "grid":
                qvals[aid] += 1.0
    best = int(np.argmax(qvals))
    return best, qvals.tolist()
