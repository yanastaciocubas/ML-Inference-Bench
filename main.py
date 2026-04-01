import click
from models import resnet50, efficientnet, yolov8, bert, whisper
from exporters import resnet50 as export_resnet50
from exporters import efficientnet as export_efficientnet
from exporters import yolov8 as export_yolov8
from exporters import bert as export_bert
from exporters import whisper as export_whisper
from engines import fp32, fp16, int8
from benchmarks import latency, throughput, memory, accuracy
from reports import json_report, csv_report, html_report

MODELS = {
    "resnet50": (resnet50, export_resnet50),
    "efficientnet": (efficientnet, export_efficientnet),
    "yolov8": (yolov8, export_yolov8),
    "bert": (bert, export_bert),
    "whisper": (whisper, export_whisper),
}

ENGINES = {
    "fp32": fp32,
    "fp16": fp16,
    "int8": int8,
}

REPORTS = {
    "json": json_report,
    "csv": csv_report,
    "html": html_report,
}

@click.command()
@click.option("--model", type=click.Choice(MODELS.keys()), default="resnet50", help="Model to benchmark")
@click.option("--precision", type=click.Choice(ENGINES.keys()), default="fp32", help="Precision mode")
@click.option("--report", type=click.Choice(REPORTS.keys()), default="json", help="Report format")
@click.option("--all-models", is_flag=True, help="Run benchmark on all models")
@click.option("--dashboard", is_flag=True, help="Launch the dashboard")
def main(model, precision, report, all_models, dashboard):

    if dashboard:
        from dashboard import app
        app.run()
        return

    models_to_run = list(MODELS.keys()) if all_models else [model]
    results = []

    for model_name in models_to_run:
        print(f"\n--- Benchmarking {model_name} with {precision} ---")

        model_loader, exporter = MODELS[model_name]
        pytorch_model, sample_input = model_loader.load()

        # Export to ONNX
        onnx_path = f"results/{model_name}.onnx"
        exporter.export(onnx_path)

        # Build TensorRT engine
        engine_path = f"results/{model_name}_{precision}.trt"
        if precision == "int8":
            ENGINES[precision].build(onnx_path, engine_path, sample_input)
        else:
            ENGINES[precision].build(onnx_path, engine_path)

        # Run benchmarks
        latency_results = latency.measure(engine_path, sample_input)
        throughput_results = throughput.measure(engine_path, sample_input)
        memory_results = memory.measure(engine_path, sample_input)
        accuracy_results = accuracy.measure(engine_path, pytorch_model, sample_input)

        results.append({
            "model": model_name,
            "precision": precision,
            **latency_results,
            **throughput_results,
            **memory_results,
            **accuracy_results,
        })

    # Save report
    report_path = f"results/report.{report}"
    REPORTS[report].save(results, report_path)
    print(f"\nDone! Report saved to {report_path}")

if __name__ == "__main__":
    main()