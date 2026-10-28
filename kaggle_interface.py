"""Gradio interface for the trained IMDb sentiment-analysis notebook.

Run these definitions after the notebook's predict_sentiment() function.
Then call build_sentiment_interface(predict_sentiment) and launch the result.
"""


def analyze_review(review, predict_fn):
    """Validate input and format the existing models' predictions for display."""
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Please enter an English movie review before analyzing.")

    results = predict_fn(review.strip())
    svm_result = results["TF-IDF + SVM"]
    cnn_result = results["GloVe + CNN"]

    if svm_result not in ("Positive", "Negative") or cnn_result not in (
        "Positive", "Negative"
    ):
        raise ValueError("The prediction function returned an unexpected label.")

    if svm_result == cnn_result:
        comparison = f"Both models predict {svm_result.lower()} sentiment."
    else:
        comparison = "The models disagree on this review."

    return svm_result, cnn_result, comparison


def format_test_results(records):
    """Use metrics from the exported run rather than hard-coding old scores."""
    if not records:
        return "Test results will appear when the saved experiment results are added."
    lines = [
        "Held-out test results for the saved experiment. These are dataset-level metrics.",
        "",
        "| Model | Accuracy | Macro F1 | ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['Model']} | {float(row['Accuracy']):.2%} | "
            f"{float(row['F1 (macro)']):.4f} | {float(row['ROC-AUC']):.4f} |"
        )
    return "\n".join(lines)


def analyze_review_for_ui(review, predict_fn):
    """Keep invalid-input feedback visible without showing model error cards."""
    try:
        return analyze_review(review, predict_fn)
    except ValueError as error:
        return "", "", str(error)


def build_sentiment_interface(predict_fn, *, ready=True, setup_message=None, test_metrics=None):
    """Build an English interface without loading or retraining any models."""
    import gradio as gr

    def run_analysis(review):
        try:
            return analyze_review_for_ui(review, predict_fn)
        except RuntimeError as error:
            raise gr.Error(str(error)) from error

    def reset_results():
        return "", "", ""

    with gr.Blocks(title="Movie Review Sentiment Analysis", analytics_enabled=False) as demo:
        gr.Markdown("# Movie Review Sentiment Analysis")
        gr.Markdown(
            "Enter an English movie review to compare **TF-IDF + SVM** "
            "and **GloVe + CNN**. Choose an example or write your own, "
            "then click **Analyze Sentiment**."
        )

        if setup_message:
            gr.Markdown(setup_message)

        review_input = gr.Textbox(
            label="Movie review",
            placeholder="What did you think of the movie?",
            lines=5,
        )

        with gr.Row():
            analyze_button = gr.Button(
                "Analyze Sentiment", variant="primary", interactive=ready
            )
            clear_button = gr.ClearButton(value="Clear")

        with gr.Row():
            svm_output = gr.Textbox(label="TF-IDF + SVM", interactive=False)
            cnn_output = gr.Textbox(label="GloVe + CNN", interactive=False)

        comparison_output = gr.Textbox(label="Model comparison", interactive=False)
        outputs = [svm_output, cnn_output, comparison_output]

        gr.Examples(
            examples=[
                ["The acting was excellent and I enjoyed every minute."],
                ["The movie was boring and I would not recommend it."],
                [
                    "The film was decent, but I didn't like the way it ended. "
                    "There should have been a different, more logical conclusion."
                ],
            ],
            inputs=[review_input],
            label="Example reviews",
            cache_examples=False,
        )

        with gr.Accordion("Held-out test results", open=False):
            gr.Markdown(format_test_results(test_metrics))

        # Clear previous results whenever the review changes.
        review_input.change(fn=reset_results, inputs=None, outputs=outputs, queue=False)

        analyze_button.click(
            fn=run_analysis,
            inputs=review_input,
            outputs=outputs,
            concurrency_limit=1,
        )

        clear_button.add([review_input, *outputs])

    return demo
