# NLP-Based-Review-Classification-Pipeline-for-Fashion-Product-Recommendations
This project uses customer reviews and profile features to predict whether a fashion product will be recommended. The goal is to help e-commerce retailers like StyleSense extract meaningful insights from customer feedback and automate sentiment labeling using natural language processing (NLP) and machine learning techniques.

## Project Overview
1. **Context & Objective**  
   Fashion retailers often receive large volumes of unstructured review data. This project builds an NLP-enabled machine learning pipeline to automatically classify whether a product is recommended, improving feedback analysis and customer insights.

2. **Data Familiarisation**  
   - Explored the class distribution and observed a slight imbalance favoring positive recommendations.  
   - Assessed vocabulary size, numeric feature distributions, and categorical coverage.  
   - Confirmed the dataset was clean and free of missing values.

3. **Feature Engineering & Pipeline Design**  
   - Text data was processed using a **custom spaCy tokenizer** followed by **TF-IDF vectorization**.  
   - Numeric features were scaled and imputed.  
   - Categorical variables were imputed and one-hot encoded.  
   These components were combined using `ColumnTransformer` and encapsulated within a single end-to-end pipeline.

4. **Modeling & Hyperparameter Tuning**  
   - A **Logistic Regression classifier** was selected for its speed and interpretability.  
   - The model was optimized using **GridSearchCV**, enabling simultaneous tuning of hyperparameters and preprocessing components.

5. **Evaluation & Insight Extraction**  
   - Performance was measured using accuracy, precision, recall, and F1 score on a hold-out test set.  
   - A **confusion matrix heatmap** was created to visualize prediction accuracy.  
   - **Top TF-IDF terms** for positive and negative recommendations were identified and visualized to improve interpretability.

6. **Packaging & Deployment Readiness**  
   - The trained pipeline was exported as a `.pkl` file, enabling seamless reuse in production systems or interactive dashboards without requiring additional preprocessing.

## Technologies Used

**Environment:**
- Python 3.11
- Google Colab

**Libraries:**
- `pandas`, `numpy` – Data handling
- `matplotlib`, `seaborn` – Visualization
- `scikit-learn` – Preprocessing, modeling, evaluation
- `spaCy` – NLP tokenization & lemmatization
- `joblib` – Model serialization

## Dataset
 
**File:** `reviews.csv`  
**Size:** ~23,000 records  
**Features used:**
- `Review Text` – Freeform customer review text
- `Age` – Reviewer age
- `Positive Feedback Count` – Number of helpful votes
- `Clothing ID`, `Division Name`, `Department Name`, `Class Name` – Product metadata
- `Recommended IND` – Target variable (1 = recommended, 0 = not recommended)

## Modeling Process

### Model Type: Logistic Regression (Binary Classification)  
**Target Variable:** `Recommended IND`  
**Input Features:**
- Text data (via TF-IDF on lemmatized review text)
- Numeric: Age, Positive Feedback Count
- Categorical: Clothing ID, Division Name, Department Name, Class Name

**Pipeline:**
- Custom spaCy tokenizer with lemmatization
- ColumnTransformer to process mixed data types
- Logistic Regression classifier wrapped in scikit-learn pipeline

## Evaluation Metrics

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | ~87%      |
| Precision (1)| 0.89      |
| Recall (1)   | 0.96      |
| F1-Score (1) | 0.92      |

- **Confusion matrix heatmap** visualized prediction accuracy
- **TF-IDF token importance** helped interpret model decisions
- 
## Key Findings

- Review text is a highly predictive feature for product recommendation.
- Structured metadata (e.g., department name, age) adds complementary signal.
- Logistic regression is effective and interpretable, with excellent class 1 recall.
- Token-level insights provide transparency into the model’s predictions.

## Conclusion

This project successfully demonstrates how NLP and structured data can be combined in a machine learning pipeline to predict product recommendations based on customer reviews. The final model achieved strong performance, particularly in identifying positive recommendations, and offers a scalable solution for automating sentiment analysis in fashion retail.
