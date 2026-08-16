# Model Training & Evaluation Analysis Report

## 1. Overview & Methodology
This report summarizes the training, tracking, and final evaluation results for the product category classification pipeline. Three distinct machine learning architectures were implemented, trained, and logged using a consolidated local **MLflow** tracking store (`mlruns/` with SQLite metadata storage):

* **TF-IDF + Logistic Regression**: Baseline linear model utilizing word and character n-gram features.
* **TF-IDF + Linear SVM (`SGDClassifier`)**: Fast linear classifier optimized with hinge loss.
* **TF-IDF + XGBoost**: Gradient boosting decision tree model mapping categorical labels through explicit integer encoding.

---

## 2. Experimental Results Summary

| Model Name | MLflow Run ID | Internal Validation Accuracy | Final Evaluation Accuracy | Final Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | `70845eddbc6341a985a2d7aa1727be3b` | 93.75% | **94.75%** | **0.9306** |
| **TF-IDF + Linear SVM** | `50bc233dbdf0453c8211bd8b9cf5f18f` | **93.93%** | 94.54% | 0.9291 |
| **TF-IDF + XGBoost** | `59bfd4eb0fd04e36a16e5dd6cc4e72a7` | 87.37% | 88.12% | 0.8765 |

---

## 3. Data Lineage & Phase-Wise Workflow

| Phase | Input Dataset / File | Script / Module | Output / Details |
| :--- | :--- | :--- | :--- |
| **1. Model Training** | `data/processed/Training_data.csv` | `src.product_classifier.models.train` | Internal train/validation split; logs `run_id`, `accuracy`, and `f1_macro` to MLflow |
| **2. Model Evaluation** | `data/processed/Query_and_Validation_data.csv` | `src.product_classifier.evaluation.evaluate` | Filters marked/labeled query records for out-of-sample evaluation |
| **3. Load Testing & Unlabeled Inference** | `data/processed/Query_and_Validation_data.csv` | `src.product_classifier.evaluation.load_test` | Filters unlabeled query records (`category` is empty) and outputs batch predictions to `data/processed/load_test_result_<run_id>.csv` |
| **4. Artifact Export** | Tracked MLflow run artifacts | `src.product_classifier.evaluation.export` | Exports serialized deployment bundle to `model/` (`model.skops`, `MLmodel`) |
| **5. Runtime Inference** | Real-time JSON payloads (`POST /predict`) | `src.product_classifier.api.app` | Generates batch product category predictions & latency metrics |

---

## 4. Phase Breakdown & Execution Flow

### Model Training (`data/processed/Training_data.csv`)
* **Input**: Ingests cleaned training data from `data/processed/Training_data.csv`.
* **Split Strategy**: Applies an internal stratified train/validation split strictly within `Training_data.csv` to fit the TF-IDF vectorizer and train candidate classifiers without data leakage.
* **Output**: Generates an active MLflow `run_id` and logs training metrics (`accuracy`, `f1_macro`).

### Model Evaluation (`data/processed/Query_and_Validation_data.csv`)
* **Input**: Ingests holdout data from `data/processed/Query_and_Validation_data.csv`.
* **Filtering**: Filters and extracts only records with pre-assigned, ground-truth categories (ignoring unlabeled query entries).
* **Output**: Computes holdout evaluation metrics (`accuracy`, `macro_f1`) against the specified `--run-id` and logs metrics plus prediction comparisons back to MLflow.

### Load Testing & Unlabeled Inference (`data/processed/Query_and_Validation_data.csv`)
* **Input**: Ingests holdout data from `data/processed/Query_and_Validation_data.csv`.
* **Filtering**: Extracts 3,754 unlabeled records where the `category` column is `None` or empty.
* **Output**: Runs offline batch prediction using the specified `--run-id` model and saves a 2-column CSV (`product_name`, `category`) to `data/processed/load_test_result_<run_id>.csv`.

### Artifact Export & Runtime Serving (`/model` Artifacts)
* **Export**: Pulls serialized model pipeline files from MLflow for `--run-id` and writes them to the runtime `/model` directory (`model.skops`, `MLmodel`).
* **Serving**: Ingests JSON request payloads (`{"products": [...]}`) via `POST /predict` for real-time inference.

---

## 5. Model Performance Analysis

### TF-IDF + Logistic Regression
* **Performance:** Achieved the highest overall performance on the final evaluation split with an accuracy of **94.75%** and a macro F1-score of **0.9306**.
* **Characteristics:** Highly stable, well-calibrated probability outputs, and extremely efficient training times. It handles high-dimensional sparse TF-IDF spaces gracefully without overfitting.

### TF-IDF + Linear SVM
* **Performance:** Performed neck-and-neck with Logistic Regression, yielding an internal validation accuracy of **93.93%** and a final evaluation accuracy of **94.54%**.
* **Characteristics:** Excellent margin maximization makes it robust against noisy text tokens. It serves as a strong, lightweight alternative for text classification tasks.

### TF-IDF + XGBoost
* **Performance:** Secured a final evaluation accuracy of **88.12%** and a macro F1-score of **0.8765**.
* **Characteristics:** While gradient boosting excels on dense tabular data, high-dimensional sparse TF-IDF text features limit its comparative performance. Tree-based splits struggle to capture linear interactions across tens of thousands of orthogonal token features as efficiently as linear separators.

---

## 6. Key Recommendations & Next Steps
1. **Model Selection:** Deploy **Linear SVM** (`run_id: 50bc233dbdf0453c8211bd8b9cf5f18f`), as it achieved the highest internal validation performance (**93.93%**) and robust generalization for production inference.
2. **Batch Prediction Export:** Run `load_test.py` with the selected run ID to generate the required unlabeled dataset inferences stored in `data/processed/load_test_result_50bc233dbdf0453c8211bd8b9cf5f18f.csv`.
3. **Artifact Tracking:** Retain the unified `mlruns/` workspace configuration containing `mlflow.db` and run artifact directories for reproducible auditing and model version control.