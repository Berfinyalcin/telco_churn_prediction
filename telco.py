
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv(r"C:\Users\Telco-Customer-Churn.csv") # Data is read from the specified file path.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce') # The 'TotalCharges' column is converted to numerical values.

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0) # The 'Churn' column is converted to binary values: 1 for "Yes" and 0 for "No".

def check_df(dataframe): # A function to check basic information about the dataframe is defined.
    print("--------- head ----------")
    print(dataframe.head(10))
    print("-------- columns --------")
    print(dataframe.columns)
    print("---------- info ---------")
    print(dataframe.info())
    print("--------- shape ---------")
    print(dataframe.shape)
    print("----- missing value -----")
    print(dataframe.isnull().sum())

check_df(df)

def grab_col_names(dataframe, car_th = 20, cat_th = 10): #Detecting variable types.
    """
    Variable types.
     1. Categorical variable
     2. Numerical variable
     3. Categorical variable with numerical appearance
     4. Cardinal variable: A variable that has a categorical appearance, does not carry any information, and is very sparse.
    """
    #Categorical variables
    cat_cols = [col for col in df.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in df.columns if dataframe[col].dtype != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in df.columns if dataframe[col].dtype == "O" and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical variables.
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols:{len(cat_cols)}")
    print(f"num_cols:{len(num_cols)}")
    print(f"cat_but_car:{len(cat_but_car)}")
    print(f"num_but_cat:{len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



def cat_summary(dataframe, col_name, plot=False): #Categorical variable analysis
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    if plot:
        sns.countplot(x = dataframe[col_name], data=dataframe)
        plt.show
for col in cat_cols:
    cat_summary(df,col)

def target_summary_with_cat(dataframe, target, categorical_col): #Analysis of the categorical variable according to the target variable
    print("categorical colon")
    print(pd.DataFrame({"Target Mean": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}))
for col in cat_cols:
    target_summary_with_cat(df,"Churn", col)


def num_summary(dataframe, numeric_cols, plot = False): #Numeric variable analysis
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[numeric_cols].describe(quantiles).T)
    if plot:
        dataframe[numeric_cols].hist(bins=20)
        plt.title(numeric_cols)
        plt.xlabel(numeric_cols)
for col in num_cols:
    num_summary(df,col)

def target_summary_with_num(dataframe, target, numerical_col): #Analysis of the numeric variable according to the target variable
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end= "\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"Churn", col)

#++++++++++++++++++++++ CORRELATION ++++++++++++++++++++++++

df[num_cols].corr()
df[num_cols].columns

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# It is observed that 'TotalChargers' is highly correlated with monthly charges and tenure.
df.corrwith(df["Churn"]).sort_values(ascending=False)

# +++++++++++++++++++ MISSING VALUE ANALYSIS +++++++++++++++++++

# A function to create a missing values table is defined.
def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_cols

na_cols = missing_values_table(df, na_name=True)

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns): # A function to examine the relationship between missing values and the target variable is defined.
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(),1,0)
    na_flags = temp_df.loc[:,temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                            "Count":temp_df.groupby(col)[target].count()}), end="\n\n\n")
missing_vs_target(df,"TotalCharges", na_cols)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True) # Filling missing values with the median.


# +++++++++++++++++++ OUTLIER ANALYSIS +++++++++++++++++++++++

def outlier_thresolds(dataframe,col_name, q1=0.25, q3=0.75): #Calculating the lower limit and upper limit.
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 -1.5*iqr
    up_limit = quartile3 + 1.5*iqr
    print("col name:{},low limit:{}, up limit:{}".format(col_name, low_limit, up_limit))
    return low_limit, up_limit

for col in num_cols:
    outlier_thresolds(df, col)

def check_outlier(dataframe, col_name): #Checking for outliers.
    low_limit, up_limit = outlier_thresolds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        print("True")
        return True
    else:
        print("False")
        return False

for col in num_cols:
    check_outlier(df, col)  #It is observed that there are no outliers.

#++++++++++++++  FEATURE EXTRACTION +++++++++++++++

# Creating a categorical variable from the 'tenure' variable.
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Marking customers with 1 or 2 years of contract as 'Engaged'.
df["NEW_ENGADED"]= df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# Customers who do not have any support, backup, or protection services.
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Customers with a monthly contract and who are young.
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_ENGADED"] == 0 or x["SeniorCitizen"] == 0) else 0, axis=1)

# The total number of services a person has.
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Customers who have any streaming service.
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Whether the person makes automatic payments.
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# Average monthly payment.
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Increase in current price compared to average price.
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Fee per service.
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

#++++++++++++++++++++++ ENCODING ++++++++++++++++++++++++

# Separating the variables according to their types.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING

binary_col = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_col

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_col:
    label_encoder(df, col)

# ONE - HOT ENCODİNG
# Updating the cat_cols list
cat_cols =[col for col in cat_cols if col not in binary_col and col not in ["Churn", "NEW_TotalServices"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

for col in df.columns:
    if df[col].dtype == ('bool'):
        df[col] = df[col].astype("int64")
df.head()
df.columns

ss = StandardScaler() #Standardizing numerical columns using StandardScaler
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()
df.columns
df.isnull().sum()
#+++++++++++++++++++++++++ MODELING +++++++++++++++++++++++++++

y = df["Churn"] #target variable
X = df.drop(["Churn","customerID"], axis=1) #independent variables

models = [('LR', LogisticRegression(random_state=123)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=123)),
          ('RF', RandomForestClassifier(random_state=123)),
          ('SVM', SVC(gamma='auto', random_state=123)),
          ('XGB', XGBClassifier(random_state=123)),
          ("LightGBM", LGBMClassifier(random_state=123)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=123))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    print(f"+++++++++++++++ {name} +++++++++++++++")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 2)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 2)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 2)}")

#+++++++++++++++ Random Forest ++++++++++++++++++

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean() #0.80
cv_results["test_f1"].mean() #0.58
cv_results["test_roc_auc"].mean() #0.84

#++++++++++++++++++++ XGBoost ++++++++++++++++++++++++++++++

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean() #0.81
cv_results['test_f1'].mean() #0.59
cv_results['test_roc_auc'].mean() #0.85

#+++++++++++++++++++ LightGBM ++++++++++++++++++++++++++

lightgbm_model = LGBMClassifier(random_state=17)

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lightgbm_best_grid = GridSearchCV(lightgbm_model, lightgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

lightgbm_final = lightgbm_model.set_params(**lightgbm_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(lightgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean() #0.80
cv_results['test_f1'].mean() #0.58
cv_results['test_roc_auc'].mean() #0.84

#+++++++++++++++++++++ CatBoost ++++++++++++++++++++++++

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean() #0.80
cv_results['test_f1'].mean() #0.58
cv_results['test_roc_auc'].mean() #0.84

#+++++++++++++++++++++ FEATURE IMPORTANCE +++++++++++++++++++++++++
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
