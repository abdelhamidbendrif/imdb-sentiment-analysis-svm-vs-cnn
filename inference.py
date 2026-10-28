"""Load exported models once and predict locally, without any training."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


class ModelSetupError(RuntimeError):
    """Model files or their matching runtime are not ready."""


class SentimentPredictor:
    REQUIRED_FILES = (
        "tfidf_vectorizer.joblib",
        "svm_model.joblib",
        "cnn_with_vectorizer.keras",
        "training_metadata.json",
        "reference_predictions.json",
    )

    def __init__(self, model_dir=None, backend="numpy"):
        if backend not in ("numpy", "tensorflow"):
            raise ValueError("Choose the numpy or tensorflow CNN backend.")
        self.backend = backend
        self.model_dir = (
            Path(model_dir) if model_dir else Path(__file__).resolve().parent / "models"
        )
        self.tfidf = None
        self.svm = None
        self.cnn = None
        self.tf = None

    def test_metrics(self):
        path = self.model_dir / "training_metadata.json"
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("test_metrics", [])

    def check_setup(self):
        missing = [name for name in self.REQUIRED_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise ModelSetupError(
                "Export the trained models from Kaggle and place the ZIP contents "
                "inside this project's models folder. Missing: " + ", ".join(missing)
            )

        metadata = json.loads((self.model_dir / "training_metadata.json").read_text(encoding="utf-8"))
        for package in ("scikit-learn", "numpy", "nltk"):
            expected = metadata["packages"][package]
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as error:
                raise ModelSetupError("Install requirements.txt before running the app.") from error
            if actual != expected:
                raise ModelSetupError(
                    f"The model uses {package} {expected}; installed version is {actual}. "
                    "Install requirements.txt before loading it."
                )
        dependency = "h5py" if self.backend == "numpy" else "tensorflow"
        if importlib.util.find_spec(dependency) is None:
            raise ModelSetupError(
                f"The {self.backend} backend requires {dependency}. "
                "Install requirements.txt, or the exported full runtime for TensorFlow."
            )
        from preprocessing import check_resources

        check_resources()

    def _load(self):
        if self.cnn is not None:
            return
        self.check_setup()
        import joblib

        from threadpoolctl import threadpool_limits
        self._thread_limiter = threadpool_limits(limits=2)
        # These files are the student's own trusted Kaggle exports.
        tfidf = joblib.load(self.model_dir / "tfidf_vectorizer.joblib")
        svm = joblib.load(self.model_dir / "svm_model.joblib")
        if self.backend == "numpy":
            from keras_numpy import KerasNumpyCNN
            cnn = KerasNumpyCNN(self.model_dir / "cnn_with_vectorizer.keras")
            tf = None
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
            cnn = tf.keras.models.load_model(
                self.model_dir / "cnn_with_vectorizer.keras", compile=False
            )
        self.tfidf, self.svm, self.cnn, self.tf = tfidf, svm, cnn, tf

    def cnn_score(self, processed):
        self._load()
        if self.backend == "numpy":
            return self.cnn.predict_score(processed)
        return float(self.cnn(self.tf.constant([processed]), training=False).numpy()[0, 0])

    def predict_details(self, review):
        from preprocessing import preprocess_raw_review

        processed = preprocess_raw_review(review)
        self._load()
        features = self.tfidf.transform([processed])
        svm_label = int(self.svm.predict(features)[0])
        svm_score = float(self.svm.decision_function(features)[0])
        cnn_score = self.cnn_score(processed)
        labels = {0: "Negative", 1: "Positive"}
        return {
            "processed": processed,
            "svm_label": labels[svm_label],
            "svm_score": svm_score,
            "cnn_label": labels[int(cnn_score >= 0.5)],
            "cnn_score": cnn_score,
        }

    def predict(self, review):
        result = self.predict_details(review)
        return {
            "TF-IDF + SVM": result["svm_label"],
            "GloVe + CNN": result["cnn_label"],
        }

    def verify_export(self):
        """Compare local outputs to predictions saved in the Kaggle runtime."""
        self.check_setup()
        references = json.loads(
            (self.model_dir / "reference_predictions.json").read_text(encoding="utf-8")
        )
        if not references:
            raise ModelSetupError("The export contains no reference predictions.")
        for reference in references:
            actual = self.predict_details(reference["review"])
            for key in ("processed", "svm_label", "cnn_label"):
                if actual[key] != reference[key]:
                    raise ModelSetupError(f"Local/Kaggle mismatch in {key}: {reference['review']}")
            for key in ("svm_score", "cnn_score"):
                if abs(actual[key] - reference[key]) > 1e-4:
                    raise ModelSetupError(f"Local/Kaggle score mismatch in {key}.")
        return len(references)
