"""CPU inference for this project's saved Keras CNN, using its original weights.

This is deliberately limited to the exported Embedding -> Conv1D -> GlobalMax
-> Dense -> Dropout -> Dense architecture. It is not a general Keras loader.
Training and model selection remain in the TensorFlow/Keras notebook.
"""

import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np


class KerasNumpyCNN:
    def __init__(self, path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            config = json.loads(archive.read("config.json"))
            outer = config["config"]["layers"]
            if [layer["class_name"] for layer in outer] != [
                "InputLayer", "TextVectorization", "Sequential"
            ]:
                raise ValueError("Unsupported Keras text-model architecture.")
            vectorizer = outer[1]["config"]
            expected = {
                "standardize": None, "split": "whitespace", "ngrams": None,
                "output_mode": "int", "sparse": False, "ragged": False,
            }
            if any(vectorizer.get(key) != value for key, value in expected.items()):
                raise ValueError("The exported text vectorizer uses unsupported settings.")
            self.sequence_length = int(vectorizer["output_sequence_length"])
            layers = outer[2]["config"]["layers"]
            if [layer["class_name"] for layer in layers] != [
                "InputLayer", "Embedding", "Conv1D", "GlobalMaxPooling1D",
                "Dense", "Dropout", "Dense"
            ]:
                raise ValueError("Unsupported CNN layers; use the TensorFlow backend.")
            conv = layers[2]["config"]
            pool = layers[3]["config"]
            if not (
                conv["padding"] == "valid" and conv["strides"] == [1]
                and conv["dilation_rate"] == [1] and conv["groups"] == 1
                and conv["data_format"] == "channels_last"
                and conv["activation"] == "relu" and conv["use_bias"]
                and pool["data_format"] == "channels_last" and not pool["keepdims"]
                and not layers[1]["config"].get("mask_zero", False)
                and layers[4]["config"]["activation"] == "relu"
                and layers[6]["config"]["activation"] == "sigmoid"
                and layers[6]["config"]["units"] == 1
            ):
                raise ValueError("Unsupported CNN configuration; use the TensorFlow backend.")
            self.kernel_size = int(conv["kernel_size"][0])
            vocabulary = archive.read(
                "assets/layers/text_vectorization/vocabulary.txt"
            ).decode("utf-8").splitlines()
            if vocabulary[:2] != ["", "[UNK]"]:
                raise ValueError("Unexpected padding and unknown-token IDs.")
            self.word_to_id = {word: index for index, word in enumerate(vocabulary)}
            # The large initializer in config is unnecessary: use learned HDF5 weights.
            del config, outer, layers
            with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
                base = "layers/sequential/layers/"
                self.embedding = weights[base + "embedding/vars/0"][:]
                self.conv_kernel = weights[base + "conv1d/vars/0"][:]
                self.conv_bias = weights[base + "conv1d/vars/1"][:]
                self.hidden_kernel = weights[base + "dense/vars/0"][:]
                self.hidden_bias = weights[base + "dense/vars/1"][:]
                self.output_kernel = weights[base + "dense_1/vars/0"][:]
                self.output_bias = weights[base + "dense_1/vars/1"][:]

        if (
            self.embedding.shape[0] != len(vocabulary)
            or self.conv_kernel.shape != (self.kernel_size, self.embedding.shape[1], len(self.conv_bias))
            or self.hidden_kernel.shape != (len(self.conv_bias), len(self.hidden_bias))
            or self.output_kernel.shape != (len(self.hidden_bias), 1)
            or self.output_bias.shape != (1,)
        ):
            raise ValueError("Saved weights do not match the model configuration.")
        self.flat_kernel = self.conv_kernel.reshape(-1, len(self.conv_bias))

    def vectorize(self, processed_text):
        ids = np.zeros(self.sequence_length, dtype=np.int32)
        tokens = processed_text.split()[:self.sequence_length]
        ids[:len(tokens)] = [self.word_to_id.get(word, 1) for word in tokens]
        return ids

    def predict_score(self, processed_text):
        embedded = self.embedding[self.vectorize(processed_text)]
        windows = np.lib.stride_tricks.sliding_window_view(
            embedded, self.kernel_size, axis=0
        ).transpose(0, 2, 1)
        convolution = windows.reshape(len(windows), -1) @ self.flat_kernel + self.conv_bias
        np.maximum(convolution, 0, out=convolution)
        pooled = convolution.max(axis=0)
        hidden = np.maximum(pooled @ self.hidden_kernel + self.hidden_bias, 0)
        # Dropout is inactive during inference, just as training=False in Keras.
        logit = float((hidden @ self.output_kernel + self.output_bias)[0])
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exponential = np.exp(logit)
        return float(exponential / (1.0 + exponential))
