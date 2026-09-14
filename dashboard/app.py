"""FedTrap Flask monitoring dashboard.

Displays real-time trap status, predictions, servo state, FL round, FPS.
NOT a dependency for core operation — the robot works without it.

Usage:  python dashboard/app.py [--port 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, jsonify
from fedtrap.config import load_config, get_project_root

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state (in-memory; updated by the main control loop or API calls)
# ---------------------------------------------------------------------------
_state = {
    "traps": {
        "trap_a": {
            "status": "OFFLINE",
            "last_prediction": None,
            "confidence": 0.0,
            "action": "NONE",
            "servo_state": "HOME",
            "fl_round": 0,
            "model_version": "initial",
            "fps": 0.0,
            "inference_latency_ms": 0.0,
            "recent_events": [],
        },
        "trap_b": {
            "status": "OFFLINE",
            "last_prediction": None,
            "confidence": 0.0,
            "action": "NONE",
            "servo_state": "HOME",
            "fl_round": 0,
            "model_version": "initial",
            "fps": 0.0,
            "inference_latency_ms": 0.0,
            "recent_events": [],
        },
    },
    "fl_status": {
        "current_round": 0,
        "total_rounds": 10,
        "method": "fedavg",
        "global_accuracy": 0.0,
        "communication_bytes": 0,
    },
}


def update_trap_state(trap_id: str, **kwargs) -> None:
    """Update trap state (called from control loop or API)."""
    if trap_id in _state["traps"]:
        _state["traps"][trap_id].update(kwargs)


def update_fl_state(**kwargs) -> None:
    """Update FL status."""
    _state["fl_status"].update(kwargs)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(_state)


@app.route("/api/trap/<trap_id>")
def api_trap(trap_id: str):
    if trap_id in _state["traps"]:
        return jsonify(_state["traps"][trap_id])
    return jsonify({"error": f"Unknown trap: {trap_id}"}), 404


@app.route("/api/fl")
def api_fl():
    return jsonify(_state["fl_status"])


@app.route("/api/events/<trap_id>")
def api_events(trap_id: str):
    """Read recent events from CSV log if available."""
    try:
        cfg = load_config()
        csv_path = Path(get_project_root()) / cfg.get("logging", {}).get(
            "csv_path", "results/events.csv"
        )
        if csv_path.is_file():
            import csv
            events = []
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("trap_id") == trap_id:
                        events.append(row)
            return jsonify(events[-20:])  # last 20
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FedTrap Dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"FedTrap Dashboard starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
