## Training Models on Imbalanced Data 
This project demonstrates how to handle class imbalance in a binary classification problem for insurance claim prediction. It covers exploratory data analysis (EDA), balancing the dataset through oversampling, selecting key features via a Random Forest, training a Random Forest classifier, and evaluating model performance with precision, recall, and F1-score metrics  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/)).

## Project Description  
The goal is to predict `claim_status` (0 = no claim, 1 = claim) from policy and vehicle attributes. The raw data exhibits a severe imbalance: far fewer claim instances than non-claims  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/)). This imbalance can bias models toward the majority class, degrading performance on the minority (claim) class.

## Installation  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/OyewoleRasheed/Training-Models-on-Imbalanced-Data/.git
   cd Training-Models-on-Imbalanced-Data
   ```
2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Requirements include:
   - Python 3.8+  
   - pandas, numpy  
   - matplotlib, seaborn  
   - scikit-learn >=1.0  ([resample — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html?utm_source=chatgpt.com))  


## Methodology  

### 1. Exploratory Data Analysis  
Visualize class distribution and feature summaries:
- Distribution of `claim_status` highlights the imbalance.  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/))  
- Histograms for numerical features and count plots for categorical features to assess their relationship with the target.  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/))  

### 2. Handling Class Imbalance  
Use **random oversampling** of the minority class via `sklearn.utils.resample` to balance the dataset to a 1:1 ratio:  
```python
from sklearn.utils import resample
minority_oversampled = resample(minority, replace=True,
                                n_samples=len(majority),
                                random_state=42)
oversampled_data = pd.concat([majority, minority_oversampled])
```  

### 3. Feature Selection  
Encode categorical variables with `LabelEncoder`, then fit a `RandomForestClassifier` to the original (imbalanced) data to extract feature importances. Drop unhelpful features (e.g., `policy_id`):  
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_encoded, y)
importances = rf.feature_importances_
```  
Top features: `policy_id`, `subscription_length`, `customer_age`, `vehicle_age`, etc.  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/))  

### 4. Model Training  
Train a Random Forest on the **oversampled** dataset:  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X_oversampled_encoded, y_oversampled, test_size=0.25, random_state=42)

rf_os = RandomForestClassifier(random_state=42)
rf_os.fit(X_train, y_train)
y_pred = rf_os.predict(X_test)
```  
`RandomForestClassifier` averages multiple decision trees for robust classification  ([RandomForestClassifier — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?utm_source=chatgpt.com)).

### 5. Evaluation  
Generate a classification report:
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```  
Key metrics on test data:
- **Accuracy:** 98%  
- **Precision & Recall:** ≥ 0.96 for both classes  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/))

## Results  
On the original imbalanced dataset, the model maintained high performance, correctly classifying over 99% of instances. A pie chart visualizes overall accuracy and misclassifications  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/)).

## Contributing  
1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature-name`).  
3. Commit your changes (`git commit -m 'Add feature'`).  
4. Push to the branch (`git push origin feature-name`).  
5. Open a pull request.

## License  
This project is licensed under the MIT License. See `LICENSE` for details.

## References  
1. Aman Kharwal, “Classification on Imbalanced Data using Python,” The Clever Programmer, Apr. 1 2024.  ([Classification on Imbalanced Data using Python | Aman Kharwal](https://thecleverprogrammer.com/2024/04/01/classification-on-imbalanced-data-using-python/))
