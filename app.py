"""Run the local sentiment-analysis interface from the VS Code terminal."""

import argparse
from pathlib import Path

from inference import SentimentPredictor
from kaggle_interface import build_sentiment_interface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None, help="Optional fixed port; otherwise use the next available Gradio port.")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check-models", action="store_true")
    parser.add_argument("--backend", choices=["numpy", "tensorflow"], default="numpy")
    args = parser.parse_args()

    predictor = SentimentPredictor(backend=args.backend)
    if args.check_models:
        count = predictor.verify_export()
        print(f"Verified {count} reference reviews against Kaggle outputs.")
        return

    try:
        predictor.check_setup()
        ready, message = True, None
    except RuntimeError as error:
        ready = False
        message = (
            "### Model setup needed\n\n"
            f"{error}\n\n"
            "See **README.md** for the Kaggle export and local setup steps. "
            "Restart the app after setup."
        )

    demo = build_sentiment_interface(
        predictor.predict,
        ready=ready,
        setup_message=message,
        test_metrics=predictor.test_metrics(),
    )
    print(f"Project folder: {Path(__file__).resolve().parent}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=not args.no_browser,
        show_error=False,
    )


if __name__ == "__main__":
    main()
