import pandas as pd
import os

def save(results, output_path="results/report.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"CSV report saved to {output_path}")