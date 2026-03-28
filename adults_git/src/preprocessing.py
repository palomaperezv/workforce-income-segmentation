import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

df = pd.read_csv('/content/sample_data/train.csv')

df.describe().T


df.isnull().sum()

print((df == '?').sum())

df.replace('?', 'Unknown', inplace=True)
print((df == '?').sum())


df['income_binary'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

sns.countplot(data=df, x='income_binary')
plt.title('Income distribution: >50K vs <=50K')
plt.xlabel('Income level')
plt.ylabel('Number of people')
plt.xticks(ticks=[0, 1], labels=['<=50K', '>50K'])
plt.show()

proporción_target = df['income'].value_counts(normalize=True)

print(f'Proportion of the target classes: \n {proporción_target *100}:')

object_cols = df.select_dtypes(include='object').columns
object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
sorted(d.items(), key=lambda x: x[1])

fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))

# education
ax[0,0].set_title('Education distribution by Income')
sns.countplot(data=df, x='education', hue='income', ax=ax[0,0])
ax[0,0].set_xlabel('Education')
ax[0,0].set_ylabel('Number of people')
ax[0,0].set_xticklabels(ax[0,0].get_xticklabels(), rotation=45, ha='right')
# marital status
ax[0,1].set_title('Marital Status distribution by Income')
sns.countplot(data=df, x='marital.status', hue='income', ax=ax[0,1])
ax[0,1].set_xlabel('Marital Status')
ax[0,1].set_ylabel('Number of people')
ax[0,1].set_xticklabels(ax[0,1].get_xticklabels(), rotation=45, ha='right')
#relationship
ax[1,0].set_title('Relationship distribution by Income')
sns.countplot(data=df, x='relationship', hue='income', ax=ax[1,0])
ax[1,0].set_xlabel('Relationship')
ax[1,0].set_ylabel('Number of people')
ax[1,0].set_xticklabels(ax[1,0].get_xticklabels(), rotation=45, ha='right')
#native.country
ax[1,1].set_title('Native Country distribution by Income')
sns.countplot(data=df, x='native.country', hue='income', ax=ax[1,1])
ax[1,1].set_xlabel('Native Country')
ax[1,1].set_ylabel('Number of people')
ax[1,1].set_xticklabels(ax[1,1].get_xticklabels(), rotation=45, ha='right')
# workclass
ax[2,0].set_title('Workclass distribution by Income')
sns.countplot(data=df, x='workclass', hue='income', ax=ax[2,0])
ax[2,0].set_xlabel('Workclass')
ax[2,0].set_ylabel('Number of people')
ax[2,0].set_xticklabels(ax[2,0].get_xticklabels(), rotation=45, ha='right')
#occupation
ax[2,1].set_title('Occupation distribution by Income')
sns.countplot(data=df, x='occupation', hue='income', ax=ax[2,1])
ax[2,1].set_xlabel('Occupation')
ax[2,1].set_ylabel('Number of people')
ax[2,1].set_xticklabels(ax[2,1].get_xticklabels(), rotation=45, ha='right')
#race
ax[3,0].set_title('Race distribution by Income')
sns.countplot(data=df, x='race', hue='income', ax=ax[3,0])
ax[3,0].set_xlabel('Race')
ax[3,0].set_ylabel('Number of people')
ax[3,0].set_xticklabels(ax[3,0].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['age', 'education.num', 'hours.per.week']])
plt.title("Outliers on Numeric Variables")
plt.show()

print("People with maximum capital gain:", len(df[df['capital.gain'] == 99999]))


df_poreduc = df.groupby('education')['education.num'].mean()
print(df_poreduc)

# Verify the bijective relationship before deleting
print(df.groupby('education')['education.num'].unique().sort_values())

# Elimination of the redundant column
df.drop('education', axis=1, inplace=True)

plt.figure(figsize=(12,6))
sns.barplot(x='marital.status', y='income_binary', data=df, errorbar=('ci', 95))
plt.xticks(rotation=90)
plt.title('Percentage of People with Income >50K by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Percentage of Income >50K')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# We group and compute the mean (p) and the sample size (n)
stats = df.groupby('marital.status')['income_binary'].agg(['mean', 'count']).copy()

# We calculate the Standard Error (SE) for proportions
stats['se'] = np.sqrt(stats['mean'] * (1 - stats['mean']) / stats['count'])

# We calculate the interval limits at 95% (z = 1.96)
stats['lower_limit'] = stats['mean'] - 1.96 * stats['se']
stats['upper_limit'] = stats['mean'] + 1.96 * stats['se']

# We sort by the mean to view the hierarchy
stats = stats.sort_values('mean', ascending=False)

print(stats[['mean', 'lower_limit', 'upper_limit']])

def preprocess_marital_status(df):
    marital_mapping = {
        'Married-civ-spouse': 'Married',
        'Married-AF-spouse': 'Married',
        'Married-spouse-absent': 'Alone',
        'Never-married': 'Alone',
        'Divorced': 'Alone',
        'Widowed': 'Alone',
        'Separated': 'Alone'
    }

    df = df.copy()
    df['marital_grouped'] = df['marital.status'].map(marital_mapping)
    df.drop(columns='marital.status', inplace=True)

    return df

df = preprocess_marital_status(df)


print(df['marital_grouped'].value_counts())

plt.figure(figsize=(12,6))
sns.barplot(x='relationship', y='income_binary', data=df, errorbar=('ci', 95))
plt.xticks(rotation=90)
plt.title('Percentage of People with Income >50K by Education Level')
plt.xlabel('Relationship')
plt.ylabel('Porcentaje de Ingresos >50K')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()

stats = df.groupby('relationship')['income_binary'].agg(['mean', 'count']).copy()

stats['se'] = np.sqrt(stats['mean'] * (1 - stats['mean']) / stats['count'])

stats['lower_limit'] = stats['mean'] - 1.96 * stats['se']
stats['upper_limit'] = stats['mean'] + 1.96 * stats['se']

stats = stats.sort_values('mean', ascending=False)

print(stats[['mean', 'lower_limit', 'upper_limit']].map(lambda x: f"{x:.2%}"))

def preprocess_relationship(df):

  relationship_mapping= {
    'Wife' : 'Partner',
    'Husband' : 'Partner',
    'Not-in-family' : 'Other',
    'Unmarried' : 'Other',
    'Other-relative' : 'Other',
    'Own-child' : 'Other'
}

  df = df.copy()
  df['relationship_grouped'] = df['relationship'].map(relationship_mapping)
  df.drop(columns='relationship', inplace=True)

  return df

df = preprocess_relationship(df)
print(df['relationship_grouped'].value_counts())

print(df['relationship_grouped'].value_counts())


plt.figure(figsize=(12,6))
sns.barplot(x='native.country', y='income_binary', data=df, errorbar=('ci', 95))
plt.xticks(rotation=90)
plt.title('Percentage of People with Income >50K by native country')
plt.xlabel('native country')
plt.ylim(0,1)
plt.ylabel('Porcentaje de Ingresos >50K')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()

stats = df.groupby('native.country')['income_binary'].agg(['mean', 'count']).copy()

stats['se'] = np.sqrt(stats['mean'] * (1 - stats['mean']) / stats['count'])

stats['lower_limit'] = stats['mean'] - 1.96 * stats['se']
stats['upper_limit'] = stats['mean'] + 1.96 * stats['se']

stats = stats.sort_values('mean', ascending=False)

print(stats[['mean', 'lower_limit', 'upper_limit']].map(lambda x: f"{x:.2%}"))

def preprocess_groupcountry(df):
  groupcountry_mapping = {
    'United-States':'United-States',
    'Jamaica': 'Other', 'India': 'Other',
    'Mexico': 'Other', 'Philippines': 'Other',
    'Dominican-Republic': 'Other',
    'El-Salvador' : 'Other',
    'China': 'Other',
    'Thailand' : 'Other',
    'Ireland' : 'Other',
    'Laos' : 'Other',
    'Iran' : 'Other',
    'France' : 'Other',
    'Guatemala' : 'Other',
    'South' : 'Other',
    'Puerto-Rico' : 'Other',
    'Japan': 'Other',
    'Portugal': 'Other',
    'Greece': 'Other',
    'Canada': 'Other',
    'Poland' : 'Other',
    'Peru' : 'Other',
    'Cuba': 'Other',
    'Columbia' : 'Other',
    'Germany': 'Other',
    'Italy' : 'Other', 'Hong': 'Other', 'Haiti': 'Other',
    'Ecuador': 'Other', 'England': 'Other', 'Nicaragua': 'Other', 'Cambodia': 'Other', 'Trinadad&Tobago': 'Other',
    'Vietnam': 'Other', 'Honduras': 'Other', 'Taiwan': 'Other', 'Scotland': 'Other', 'Yugoslavia': 'Other',
    'Unknown' : 'Other'
  }

  df = df.copy()
  df['native.country_grouped'] = df['native.country'].map(groupcountry_mapping)
  df.drop(columns='native.country', inplace=True)

  return df

df = preprocess_groupcountry(df)


plt.figure(figsize=(12,6))
sns.barplot(x='occupation', y='income_binary', data=df, errorbar=('ci', 95))
plt.xticks(rotation=90)
plt.title('Percentage of People with Income >50K by Occupation')
plt.xlabel('occupation')
plt.ylim(0,1)
plt.ylabel('Percentage of Income >50K')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()

stats = df.groupby('occupation')['income_binary'].agg(['mean', 'count']).copy()

stats['se'] = np.sqrt(stats['mean'] * (1 - stats['mean']) / stats['count'])

stats['lower_limit'] = stats['mean'] - 1.96 * stats['se']
stats['upper_limit'] = stats['mean'] + 1.96 * stats['se']

stats = stats.sort_values('mean', ascending=False)

print(stats[['mean', 'lower_limit', 'upper_limit']])


def preprocess_occupation(df):
  occupation_mapping = {

   'Exec-managerial':'high-tier',
    'Prof-specialty': 'high-tier',
    'Protective-serv': 'middle-tier',
  'Tech-support' :'middle-tier', 'Sales':'middle-tier',
    'Craft-repair': 'middle-tier', 'Transport-moving':'middle-tier', 'Adm-clerical':'middle-tier',
    'Machine-op-inspct': 'low_tier/unknown', 'farming-fishing' : 'low_tier/unknown', 'Handlers-cleaners' : 'low_tier/unknown', 'Other-service':'low_tier/unknown',
    'Priv-house-serv': 'low_tier/unknown',
    'Armed-Forces' : 'low_tier/unknown',
    'Unknown' : 'low_tier/unknown'
  }

  df = df.copy()
  df['occupation_grouped'] = df['occupation'].map(occupation_mapping)
  df.drop(columns='occupation', inplace=True)
  return df

df = preprocess_occupation(df)


plt.figure(figsize=(12,6))
sns.barplot(x='workclass', y='income_binary', data=df, errorbar=('ci', 95))
plt.xticks(rotation=90)
plt.title('Percentage of People with Income >50K by Workclass')
plt.xlabel('workclass')
plt.ylim(0,1)
plt.ylabel('Percentage of Income >50K')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()

stats = df.groupby('workclass')['income_binary'].agg(['mean', 'count']).copy()

stats['se'] = np.sqrt(stats['mean'] * (1 - stats['mean']) / stats['count'])

stats['lower_limit'] = stats['mean'] - 1.96 * stats['se']
stats['upper_limit'] = stats['mean'] + 1.96 * stats['se']

stats = stats.sort_values('mean', ascending=False)

print(stats[['mean', 'lower_limit', 'upper_limit']])

def preprocess_workclass(df):
  workclass_mapping = { 'Local-gov' : 'Mid-tier-workclass', 'Self-emp-not-inc' : 'Mid-tier-workclass',
                     'State-gov' : 'Mid-tier-workclass', 'Never-worked' : 'Without-income',
                      'Without-pay' : 'Without-income', 'Private' : 'Private',
                      'Self-emp-inc' : 'Self-emp-inc', 'Federal-gov': 'Federal-gov',
                      'Unknown' : 'Unknown'
                 }

  df = df.copy()
  df['workclass_grouped'] = df['workclass'].map(workclass_mapping)
  df.drop(columns='workclass', inplace=True)
  return df

df = preprocess_workclass(df)

df['race'].describe()

plt.figure(figsize=(12,6))
sns.barplot(x='race', y='income_binary', data=df, errorbar=('ci', 95))
plt.xticks(rotation=90)
plt.title('Percentage of People with Income >50K by Race')
plt.xlabel('race')
plt.ylim(0,1)
plt.ylabel('Percentage of Income >50K')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()

stats = df.groupby('race')['income_binary'].agg(['mean', 'count']).copy()

stats['se'] = np.sqrt(stats['mean'] * (1 - stats['mean']) / stats['count'])

stats['lower_limit'] = stats['mean'] - 1.96 * stats['se']
stats['upper_limit'] = stats['mean'] + 1.96 * stats['se']

stats = stats.sort_values('mean', ascending=False)

print(stats[['mean', 'lower_limit', 'upper_limit']])

def preprocess_race(df):
  race_mapping = {
     'Asian-Pac-Islander': 'Segmento I', 'White': 'Segmento I',
      'Black' : 'Segmento II' , 'Amer-Indian-Eskimo': 'Segmento II', 'Other': 'Segmento II'
}

  df = df.copy()
  df['race_grouped'] = df['race'].map(race_mapping)
  df.drop(columns='race', inplace=True)
  return df

df = preprocess_race(df)


df.drop(columns=['ID', 'index'], inplace=True)


from scipy.stats import chi2_contingency

def cramers_v(x ,y):
  contingency_table = pd.crosstab(x,y)
  chi2, __, __, _ = chi2_contingency(contingency_table)
  n = contingency_table.sum().sum()
  return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

categoricas = df.select_dtypes(include=['object']).columns

cramers_v_matrix = pd.DataFrame(np.zeros((len(categoricas), len(categoricas))), columns=categoricas, index=categoricas)

for col1 in categoricas:
  for col2 in categoricas:
    cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

mask = np.triu(np.ones_like(cramers_v_matrix, dtype=bool), k=1)

plt.figure(figsize=(10,8))
ax = sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": .8}, mask=mask)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.title("Correlation matrix of Cramer's V for Categorical Variables post-treatment", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

df_corr = df.copy()
numericas_con_target = df_corr.select_dtypes(include=['int64', 'float64'])
corr = numericas_con_target.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix for numerical variables", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

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

rf_clf = RandomForestClassifier(max_depth=10, n_estimators=200, n_jobs=-1, random_state=0, class_weight='balanced')

rf_clf.fit(X_train, y_train)

train_pred = rf_clf.predict(X_train)

test_pred= rf_clf.predict(X_test)

from sklearn.metrics import classification_report

print('metrics in train')
print(classification_report(y_train, train_pred))
print('metrics in test')
print(classification_report(y_test, test_pred))

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