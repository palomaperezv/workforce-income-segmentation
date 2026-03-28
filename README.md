Workforce Income Segmentation: Predictive Classification Model

A machine learning classification model that predicts income level from census demographic data — enabling workforce segmentation, labor market analysis, and operational planning in consulting and supply chain contexts.

TL;DR: Random Forest classifier trained on 48,000+ records achieves 82% accuracy and captures 85% of high-income individuals — 2.4x more efficient than random baseline.


Business Problem
Organizations in consulting, logistics, and operations need to understand workforce composition and income distribution to make decisions about talent allocation, compensation benchmarking, and market segmentation.
This model answers: Which socioeconomic and demographic factors best predict whether a worker earns above a threshold — and how can that inform operational decisions?

Key Results
MetricValueTest Accuracy82%Recall (high-income class)85%F1-Score (high-income class)0.69Efficiency vs. random baseline2.4x

Visualizations
1. Class Distribution
(insert: class_distribution.png)

75.7% of records fall below $50K — class imbalance handled via balanced weights in the classifier.

2. Feature Importance
(insert: feature_importance.png)

Household stability (marital status, relationship) and education years are the strongest income predictors — ahead of occupation type.

3. Confusion Matrix
(insert: confusion_matrix.png)

High recall on the minority class (85%) confirms the model prioritizes capturing high-income individuals over avoiding false positives — appropriate for discovery-driven use cases.


Methodology
Data: UCI Adult Census dataset — 48,842 records, 14 features, binary target (>$50K / ≤$50K)
Preprocessing highlights:

Hidden missing values (?) preserved as "Unknown" category — missingness treated as signal, not noise
Categorical consolidation: marital status → 3 groups, occupation → 4 tiers, country → 5 regional clusters
One-hot encoding for nominal variables; balanced class weights for imbalance

Model: Random Forest Classifier (n_estimators=100, max_depth=20, class_weight='balanced')
Key decision: Optimized for recall over precision — in workforce segmentation, missing a high-value profile is costlier than a false positive.

Project Structure
income-prediction-model/
├── src/
│   ├── preprocessing.py       # Feature engineering pipeline
│   ├── train.py               # Model training and evaluation
│   └── predict.py             # Inference on unlabeled data
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── eda_exploration.ipynb
├── docs/
│   └── img/                   # Visualizations
├── submission.csv
├── requirements.txt
└── README.md

Quickstart
bashgit clone https://github.com/palomaperezv/income-prediction-model.git
cd income-prediction-model
pip install -r requirements.txt
python src/train.py

Next Steps

Hyperparameter tuning via GridSearchCV
Benchmark against XGBoost and LightGBM
Threshold optimization for precision/recall trade-off
Apply cohort analysis by occupation tier for deeper workforce segmentation


Author
Paloma Perez Valdenegro — Geologist & Data Scientist
GitHub · LinkedIn · palomprz@gmail.com

References

UCI Adult Census Dataset
Scikit-learn Documentation


