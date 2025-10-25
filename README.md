# Task 4: Spam Detection using Logistic Regression and TF-IDF

## Objective
Classify SMS messages as **Spam** or **Ham (not spam)** using machine learning and text feature extraction.

## Steps Performed
1. Loaded the "spam.csv" dataset and cleaned column names.
2. Selected relevant columns: `v1` (label) and `v2` (message), renamed them to `label` and `message`.
3. Label encoded the target variable `label` (Ham=0, Spam=1).
4. Split the dataset into training (80%) and testing (20%) sets.
5. Converted text messages to numerical features using **TF-IDF Vectorizer** (max 3000 features, English stop words removed).
6. Trained **Logistic Regression** on the TF-IDF features.
7. Evaluated model performance using:
   - **Accuracy**
   - **Classification Report** (precision, recall, F1-score)
   - **Confusion Matrix**

## Tools / Technologies Used
- Python 3.12
- Pandas, scikit-learn (`train_test_split`, `LabelEncoder`, `TfidfVectorizer`, `LogisticRegression`, `accuracy_score`, `classification_report`, `confusion_matrix`)  
- Jupyter Notebook or Google Colab  

## Outcome / Results
- **Accuracy:** [96.41 %]  
- **Classification Report:** Shows precision, recall, and F1-score for both Ham and Spam.  
- **Confusion Matrix:** Displays correct and misclassified predictions.  
- Logistic Regression with TF-IDF features performs well in detecting spam messages.

