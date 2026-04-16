# run_sla.py

from pathlib import Path
from SLA import evaluate_states_file

states_path = Path("processed_state/states.jsonl")
output_path = Path("processed_state/sla_results.json")

evaluate_states_file(states_path, output_path)