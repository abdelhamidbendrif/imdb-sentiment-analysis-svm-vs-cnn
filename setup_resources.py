"""Download the two small NLTK resources into this project folder."""

from pathlib import Path

import nltk


def main():
    resource_dir = Path(__file__).resolve().parent / "nltk_data"
    resource_dir.mkdir(parents=True, exist_ok=True)
    for name in ("wordnet", "averaged_perceptron_tagger_eng"):
        if not nltk.download(name, download_dir=str(resource_dir)):
            raise RuntimeError(f"Could not download NLTK resource: {name}")
    from preprocessing import check_resources

    check_resources()
    print("NLTK resources are ready.")


if __name__ == "__main__":
    main()
