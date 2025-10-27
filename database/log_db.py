import json
import os
from datetime import datetime

LOG_PATH = 'data/logs/ppe_violations.jsonl'

def log_ppe_violation(emp_id, missing_items):
    if not emp_id:
        return
    log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "employee_id": emp_id,
        "missing_ppe": list(missing_items)
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(log) + "\n")
