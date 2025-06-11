from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.utils import to_categorical


class ModelEncapsulator:
    """Encapsulates models with train, fit and predict methods."""

    def __init__(self, model, name: str = "DefaultModelName"):
        self.model = model
        self.name = name

    def predict(self, X):
        """Predict labels for features X."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def fit(self, X, y):
        """Train the model."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def perform_pipeline(self, X, y):
        """
        Perform the training and evaluation pipeline.
        Returns accuracy and classification report.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class BasicModelEncapsulator(ModelEncapsulator):
    """Encapsulates models with train, fit and predict methods."""

    def __init__(self, model, name: str = "BasicModel"):
        self.model = model
        self.name = name

    def predict(self, X):
        """Predict labels for features X."""
        return self.model.predict(X)

    def fit(self, X, y):
        """Alias for train method."""
        self.model.fit(X, y)

    def perform_pipeline(self, X, y):
        """"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.fit(X_train, y_train)
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        return accuracy, report


class NeuralNetworkModel(ModelEncapsulator):
    """Neural network model designed for TF-IDF features."""

    def __init__(
        self, input_dim: int, num_classes: int = 3, name: str = "NeuralNetworkModel"
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_model()
        self.name = name

    def _build_model(self):
        """Build neural network for TF-IDF features."""
        model = Sequential()
        # Explicitly define the input shape using an Input layer
        model.add(Input(shape=(self.input_dim,)))  # ✅ Replaces input_dim in Dense

        # Dense layers for TF-IDF input
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Dropout(0.4))
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=1e-4),
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        """Train the neural network."""
        # Convert sparse matrix to dense if needed
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Convert labels to categorical (one-hot encoding)

        y_categorical = to_categorical(y, num_classes=self.num_classes)

        history = self.model.fit(
            X,
            y_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
        )
        return history

    def predict(self, X):
        """Predict labels for features X."""
        # Convert sparse matrix to dense if needed
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Get predictions and convert back to class labels
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=-1)

    def perform_pipeline(self, X, y, epochs=10, batch_size=32):
        """
        Perform the training and evaluation pipeline.
        Returns accuracy and classification report.
        """
        label_mapping = {0: "neutral", 1: "positive", 2: "negative"}
        ohe_labels = [
            0 if label == "neutral" else 1 if label == "positive" else 2 for label in y
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X, ohe_labels, test_size=0.3, random_state=42
        )

        self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        predictions = self.predict(X_test)

        # Map integer predictions back to string labels
        y_test_str = [label_mapping[label] for label in y_test]
        predictions_str = [label_mapping[label] for label in predictions]

        accuracy = accuracy_score(y_test_str, predictions_str)
        report = classification_report(y_test_str, predictions_str, output_dict=True)

        return accuracy, report
