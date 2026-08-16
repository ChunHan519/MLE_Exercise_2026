import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from xgboost import XGBClassifier


class BaseClassifier:
    """Base class for classifiers without using the ABC module."""

    def __init__(self, name: str):
        self.name = name
        self.pipeline: Pipeline | None = None

    def _build_tfidf_features(self) -> FeatureUnion:
        return FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=15000)),
            ]
        )

    def build_pipeline(self) -> Pipeline:
        raise NotImplementedError("Subclasses must implement build_pipeline().")

    def fit(self, X_train: pd.Series, y_train: pd.Series):
        if self.pipeline is None:
            self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

    def predict(self, X: pd.Series):
        if self.pipeline is None:
            raise ValueError(f"Model '{self.name}' is not trained.")
        return self.pipeline.predict(X)


class LogisticRegressionClassifier(BaseClassifier):

    def __init__(self):
        super().__init__(name="TF-IDF + Logistic Regression")

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("features", self._build_tfidf_features()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=500,
                        tol=1e-4,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )


class LinearSVMClassifier(BaseClassifier):

    def __init__(self):
        super().__init__(name="TF-IDF + Linear SVM")

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("features", self._build_tfidf_features()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        max_iter=1000,
                        tol=1e-3,
                        early_stopping=True,
                        n_iter_no_change=5,
                        random_state=42,
                    ),
                ),
            ]
        )


class XGBoostClassifier(BaseClassifier):

    def __init__(self, label_mapping: dict[str, int]):
        super().__init__(name="TF-IDF + XGBoost")
        self.label_mapping = label_mapping
        self.inv_mapping = {v: k for k, v in label_mapping.items()}

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "features",
                    TfidfVectorizer(ngram_range=(1, 2), max_features=2000),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, X_train: pd.Series, y_train: pd.Series):
        y_train_encoded = y_train.map(self.label_mapping)
        super().fit(X_train, y_train_encoded)

    def predict(self, X: pd.Series):
        preds_encoded = super().predict(X)
        return [self.inv_mapping[p] for p in preds_encoded]