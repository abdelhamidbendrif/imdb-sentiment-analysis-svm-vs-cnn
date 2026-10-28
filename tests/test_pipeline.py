"""Check preprocessing fidelity and prevent misleading interface behavior."""

from pathlib import Path
import tempfile
import unittest

from inference import ModelSetupError, SentimentPredictor
from kaggle_interface import analyze_review, analyze_review_for_ui, format_test_results
from preprocessing import clean_text, normalize_negation, tokenize_text


class PipelineTests(unittest.TestCase):
    def test_html_cleanup_keeps_negation(self):
        self.assertEqual(
            clean_text("The movie was <br /> NOT good &amp; very boring!"),
            "the movie was not good & very boring!",
        )

    def test_negative_contractions_preserve_not(self):
        text = "i didn\u2019t enjoy it and can't recommend it; it won't improve"
        result = tokenize_text(normalize_negation(text))
        self.assertEqual(
            result,
            ["i", "did", "not", "enjoy", "it", "and", "can", "not", "recommend", "it", "it", "will", "not", "improve"],
        )

    def test_blank_input_never_reaches_models(self):
        def must_not_predict(_):
            self.fail("The models must not run for empty input.")

        for value in (None, "", "   "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze_review(value, must_not_predict)

    def test_disagreement_is_not_changed_into_agreement(self):
        result = analyze_review(
            "A mixed review",
            lambda _: {"TF-IDF + SVM": "Positive", "GloVe + CNN": "Negative"},
        )
        self.assertEqual(result[:2], ("Positive", "Negative"))
        self.assertEqual(result[2], "The models disagree on this review.")

    def test_missing_bundle_explains_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            predictor = SentimentPredictor(Path(directory))
            with self.assertRaisesRegex(ModelSetupError, "Export the trained models"):
                predictor.check_setup()

    def test_interface_uses_current_export_metrics(self):
        records = [{
            "Model": "TF-IDF + SVM", "Accuracy": 0.8,
            "F1 (macro)": 0.79, "ROC-AUC": 0.9,
        }]
        table = format_test_results(records)
        self.assertIn("80.00%", table)
        self.assertNotIn("90.86%", table)
        self.assertNotIn("90.86%", format_test_results([]))

    def test_blank_input_feedback_keeps_prediction_fields_empty(self):
        def must_not_predict(_):
            self.fail("Blank input must not reach the models.")
        svm, cnn, message = analyze_review_for_ui("  ", must_not_predict)
        self.assertEqual((svm, cnn), ("", ""))
        self.assertIn("Please enter", message)


if __name__ == "__main__":
    unittest.main()
