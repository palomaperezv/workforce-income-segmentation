
#Model Training and Evaluation
X = df.drop(['income_binary', 'income'], axis=1)
y = df['income_binary']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary')

nominals_columns = X_train.select_dtypes(include='object').columns.tolist()
numeric_columns = X_train.select_dtypes(include=['int', 'float']).columns.tolist()

#column selection
nominals_columns, numeric_columns

# Fit OneHotEncoder on X_train's categorical columns
ohe.fit(X_train[nominals_columns])
X_train_cat_encoded = ohe.transform(X_train[nominals_columns])

encoded_cols = ohe.get_feature_names_out(nominals_columns)

# Create DataFrames for encoded categorical features for X_train
X_train_cat_encoded_df = pd.DataFrame(X_train_cat_encoded, columns=encoded_cols, index=X_train.index)

# Transform X_test using the encoder fitted on X_train
X_test_cat_encoded = ohe.transform(X_test[nominals_columns])
X_test_cat_encoded_df = pd.DataFrame(X_test_cat_encoded, columns=encoded_cols, index=X_test.index)

# Concatenate numerical and encoded categorical features for X_train, dropping original categorical columns
X_train = pd.concat([X_train[numeric_columns], X_train_cat_encoded_df], axis=1)
X_train = X_train.drop(columns=nominals_columns, errors='ignore')

# Concatenate numerical and encoded categorical features for X_test, dropping original categorical columns
X_test = pd.concat([X_test[numeric_columns], X_test_cat_encoded_df], axis=1)
X_test = X_test.drop(columns=nominals_columns, errors='ignore')

print('X_train info:')
X_train.info()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('\nX_train info:')
X_train.info()

print('\nX_test info:')
X_test.info()

from sklearn.ensemble import RandomForestClassifier

"""We will use the class_weight='balanced' parameter because of our imbalanced target."""

rf_clf = RandomForestClassifier(max_depth=10, n_estimators=200, n_jobs=-1, random_state=0, class_weight='balanced')

rf_clf.fit(X_train, y_train)

"""We now make the predictions:"""

train_pred = rf_clf.predict(X_train)

test_pred= rf_clf.predict(X_test)

from sklearn.metrics import classification_report

print('metrics in train')
print(classification_report(y_train, train_pred))
print('metrics in test')
print(classification_report(y_test, test_pred))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

rf_clf.feature_importances_

fi = pd.DataFrame(columns=["FEATURE", "IMPORTANCE"])

fi["FEATURE"] = X_train.columns
fi["IMPORTANCE"] = rf_clf.feature_importances_

fi.sort_values("IMPORTANCE", ascending=False, inplace=True)

fi

sns.barplot(x=fi['IMPORTANCE'], y=fi['FEATURE'])
plt.title(label='Feature importances')
plt.show()


df_predict = pd.read_csv('/content/sample_data/test.csv')
ids_test = df_predict['ID']

print("Original shape of df_predict:", df_predict.shape)

df_predict.replace('?', 'Unknown', inplace=True)

df_predict.drop(columns=['ID', 'index'], inplace=True, errors='ignore')

df_predict.drop(columns=['education'], inplace=True, errors='ignore')

#reducing categorical variables cardinality
df_predict = preprocess_marital_status(df_predict)
df_predict = preprocess_relationship(df_predict)
df_predict = preprocess_groupcountry(df_predict)
df_predict = preprocess_occupation(df_predict)
df_predict = preprocess_workclass(df_predict)
df_predict = preprocess_race(df_predict)

print("df_predict después de la ingeniería de características:")
df_predict.info()

df_predict

nominal_cols_to_transform_predict = [col for col in nominals_columns if col in df_predict.columns]

X_predict_cat_encoded = ohe.transform(df_predict[nominal_cols_to_transform_predict])
X_predict_cat_encoded_df = pd.DataFrame(X_predict_cat_encoded, columns=encoded_cols, index=df_predict.index)

numeric_cols_predict = [col for col in df_predict.columns if col not in nominal_cols_to_transform_predict]

X_predict = pd.concat([df_predict[numeric_cols_predict], X_predict_cat_encoded_df], axis=1)

print("X_predict shape after One-Hot Encoding:", X_predict.shape)
print("X_predict info after One-Hot Encoding:")
X_predict.info()


missing_cols_in_predict = set(X_train.columns) - set(X_predict.columns)
for c in missing_cols_in_predict:
    X_predict[c] = 0

extra_cols_in_predict = set(X_predict.columns) - set(X_train.columns)
X_predict.drop(columns=list(extra_cols_in_predict), inplace=True)

X_predict = X_predict[X_train.columns]

print("X_predict final shape (aligned with X_train):", X_predict.shape)
print("X_predict columns aligned:\n", X_predict.columns.tolist())

predictions = rf_clf.predict(X_predict)

print("Generated predictions:", predictions[:10])

rf_clf.predict_proba(X_predict)

prediction_counts = pd.Series(predictions).value_counts()
prediction_pct = pd.Series(predictions).value_counts(normalize=True) * 100

print("--- Distribution of Predictions ---")
for label, pct in prediction_pct.items():
    class_name = ">50K" if label == 1 else "<=50K"
    count = prediction_counts[label]
    print(f"Class {class_name}: {count} individuals ({pct:.2f}%)")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))
sns.barplot(x=[">50K", "<=50K"], y=[prediction_pct[1], prediction_pct[0]], palette='viridis')
plt.title('Predicted Income Distribution (Unlabeled Test Set)')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.show()

submission_df = pd.DataFrame({'ID': ids_test, 'PRED': predictions})

submission_df['PRED'] = submission_df['PRED'].map({1: '>50K', 0: '<=50K'})

submission_df.to_csv('submission.csv', index=False)

print("File: 'submission.csv' succesfully created!.")
print("First 5 rows of the submission file:\n", submission_df.head())
