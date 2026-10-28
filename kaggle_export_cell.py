"""Copy this entire file into a final code cell in the trained Kaggle notebook."""

import importlib.metadata
import json
import platform
from pathlib import Path
import zipfile


def export_for_vscode(namespace):
    required = ("tfidf", "svm_model", "cnn_vectorizer", "clean_text", "preprocess_review")
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError(
            "Run this cell in the trained notebook. Missing: " + ", ".join(missing)
        )

    import joblib
    import tensorflow as tf

    source_dir = Path(namespace.get("checkpoint_dir", "/kaggle/working/sentiment_project"))
    best_checkpoint = source_dir / "best_cnn.keras"
    if not best_checkpoint.is_file():
        raise FileNotFoundError("The validation-selected best_cnn.keras checkpoint is missing.")

    # Reconstruct the text-input model from the selected checkpoint.
    # No model is trained or fitted here.
    best_cnn = tf.keras.models.load_model(best_checkpoint, compile=False)
    text_input = tf.keras.Input(shape=(), dtype=tf.string, name="preprocessed_review")
    output = best_cnn(namespace["cnn_vectorizer"](text_input))
    text_model = tf.keras.Model(text_input, output)

    export_dir = source_dir.parent / "vscode_models"
    export_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(namespace["tfidf"], export_dir / "tfidf_vectorizer.joblib")
    joblib.dump(namespace["svm_model"], export_dir / "svm_model.joblib")
    text_model.save(export_dir / "cnn_with_vectorizer.keras")

    package_names = ("scikit-learn", "numpy", "joblib", "nltk", "keras")
    packages = {name: importlib.metadata.version(name) for name in package_names}
    packages["tensorflow"] = tf.__version__

    metadata = {
        "python_version": platform.python_version(),
        "packages": packages,
        "cnn_threshold": 0.5,
        "cnn_selection": "minimum validation loss; best_cnn.keras",
        "preprocessing": "HTML cleaning, lowercase, negation expansion, regex tokens, POS-aware WordNet lemmatization",
        "labels": {"0": "Negative", "1": "Positive"},
        "test_metrics": (
            namespace["test_comparison"].to_dict(orient="records")
            if "test_comparison" in namespace else []
        ),
        "test_size": len(namespace["y_test"]) if "y_test" in namespace else None,
    }
    (export_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (export_dir / "requirements-inference.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in packages.items()) + "\n",
        encoding="utf-8",
    )

    # Reference predictions check preprocessing and model parity after moving.
    examples = [
        "The acting was excellent and I enjoyed every minute.",
        "The movie was boring and I would not recommend it.",
        "The film was decent, but I didn\u2019t like the way it ended; I believe there should have been a different, more logical conclusion.",
        "The actors were watching movies, but I didn't enjoy them.",
    ]
    references = []
    labels = {0: "Negative", 1: "Positive"}
    for review in examples:
        processed = namespace["preprocess_review"](namespace["clean_text"](review))
        features = namespace["tfidf"].transform([processed])
        svm_label = int(namespace["svm_model"].predict(features)[0])
        svm_score = float(namespace["svm_model"].decision_function(features)[0])
        cnn_score = float(text_model(tf.constant([processed]), training=False).numpy()[0, 0])
        references.append({
            "review": review,
            "processed": processed,
            "svm_label": labels[svm_label],
            "svm_score": svm_score,
            "cnn_label": labels[int(cnn_score >= 0.5)],
            "cnn_score": cnn_score,
        })

    (export_dir / "reference_predictions.json").write_text(
        json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    filenames = (
        "tfidf_vectorizer.joblib", "svm_model.joblib", "cnn_with_vectorizer.keras",
        "training_metadata.json", "requirements-inference.txt", "reference_predictions.json",
    )
    zip_path = source_dir.parent / "vscode_models.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in filenames:
            archive.write(export_dir / name, arcname=name)

    print("Export created:", zip_path)
    print("Training Python:", metadata["python_version"])
    print("Runtime versions:", json.dumps(packages, indent=2))
    print("Download the ZIP and extract its files directly into PROJECT_NLP/models/.")
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(zip_path)))
    except ImportError:
        pass


if __name__ == "__main__":
    export_for_vscode(globals())
