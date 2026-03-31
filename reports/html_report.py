import pandas as pd
import os

def save(results, output_path="results/report.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    html = df.to_html(index=False, border=1)
    with open(output_path, "w") as f:
        f.write(f"""
        <html>
        <head>
            <title>ML Inference Bench Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th {{ background-color: #4CAF50; color: white; padding: 8px; }}
                td {{ padding: 8px; text-align: center; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>ML Inference Bench Report</h1>
            {html}
        </body>
        </html>
        """)
    print(f"HTML report saved to {output_path}")