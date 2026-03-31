import json
import os

def save(results, output_path="results/report.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"JSON report saved to {output_path}")