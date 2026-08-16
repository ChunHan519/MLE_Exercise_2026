## Data Lineage & Phase-Wise Dataset Usage

| Phase | Input Dataset / File | Script / Module | Output / Details |
| :--- | :--- | :--- | :--- |
| **1. Model Training** | `data/processed/Training_data.csv` | `src.product_classifier.models.train` | Internal train/validation split; logs `run_id`, `accuracy`, and `f1_macro` to MLflow |
| **2. Model Evaluation** | `data/processed/Query_and_Validation_data.csv` | `src.product_classifier.evaluation.evaluate` | Filters marked/labeled query records for out-of-sample evaluation |
| **3. Artifact Export** | Tracked MLflow run artifacts | `src.product_classifier.evaluation.export` | Exports serialized deployment bundle to `model/` (`model.skops`, `MLmodel`) |
| **4. Runtime Inference** | Real-time JSON payloads (`POST /predict`) | `src.product_classifier.api.app` | Generates batch product category predictions & latency metrics |

---

### Phase Breakdown & Data Handling

* **Model Training (`data/processed/Training_data.csv`)**:
  * **Input**: Ingests `data/processed/Training_data.csv`.
  * **Split Strategy**: Applies an internal train/validation split (e.g., 80/20 stratified split) strictly within `Training_data.csv` to fit the TF-IDF vectorizer and train candidate classifiers without data leakage.
  * **Output**: Generates an active MLflow `run_id` and logs training performance metrics (`accuracy`, `f1_macro`).

* **Model Evaluation (`data/processed/Query_and_Validation_data.csv`)**:
  * **Input**: Ingests `data/processed/Query_and_Validation_data.csv`.
  * **Filtering**: Filters and extracts only records with pre-assigned, ground-truth categories (ignoring unlabeled query entries).
  * **Output**: Computes holdout evaluation metrics against the specified `--run-id` and logs final validation results back to MLflow.

* **Artifact Export & Runtime Serving (`/model` Artifacts)**:
  * **Export**: Pulls serialized model pipeline files from MLflow for `--run-id` and writes them to the runtime `/model` directory.
  * **Serving**: Ingests JSON request payloads (`{"products": [...]}`) via `POST /predict` for real-time inference.