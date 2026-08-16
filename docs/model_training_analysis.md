# Model Training & Evaluation Analysis Report

## 1. Overview & Methodology
This report summarizes the training, tracking, and final evaluation results for the product category classification pipeline. Three distinct machine learning architectures were implemented, trained, and logged using a consolidated local **MLflow** tracking store (`mlruns/` with SQLite metadata storage):

* **TF-IDF + Logistic Regression**: Baseline linear model utilizing word and character n-gram features.
* **TF-IDF + Linear SVM (`SGDClassifier`)**: Fast linear classifier optimized with hinge loss.
* **TF-IDF + XGBoost**: Gradient boosting decision tree model mapping categorical labels through explicit integer encoding.

---

## 2. Experimental Results Summary

| Model Name | Internal Validation Accuracy | Final Evaluation Accuracy | Final Macro F1-Score |
| :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 93.75% | **94.75%** | **0.9306** |
| **TF-IDF + Linear SVM** | **93.93%** | 94.54% | 0.9291 |
| **TF-IDF + XGBoost** | 87.37% | 88.12% | 0.8765 |

---

## 3. Model Performance Analysis

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

## 4. Key Recommendations & Next Steps
1. **Model Selection:** Deploy **Linear SVM** exclusively, as it achieved the highest internal validation performance (**93.93%**) and robust generalization for production inference.
2. **Artifact Tracking:** Retain the unified `mlruns/` workspace configuration containing both `mlflow.db` and run artifact directories for reproducible auditing and model version control.