
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

"""Ensure that the columns in X_predict match those in X_train."""

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
