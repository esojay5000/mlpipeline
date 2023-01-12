import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np


# load model
modelpath = 'RF_model_and_imputation_values.pkl'
global loaded_model
try:
  loaded_model, imputation_values = pickle.load(open(modelpath, 'rb'))
except:
  print('model failed to load!')

# data transformation functions
def conv_emp_length(x):
  """ Bin employment length feature into 1) unknown, 2) less than one year,
  3) greater than 10 years, 4) 1 to 4 years and 5) 5 to 9 years
  """
  if x == 'n/a':
    return 'unknown'
  elif x == 'NaN':
      return 'unknown'
  elif pd.isnull(x):
      return 'unknown'
  elif x ==  '< 1 year':
      return 'less-than-one-year'
  elif x == '10+ years':
      return 'ten-or-more-years'
  elif int(x.split()[0]) < 5:
      return 'one-to-four-years'
  elif int(x.split()[0]) < 10:
      return 'five-to-nine-years'
  else:
      raise RuntimeError('Unrecognized value: {}'.format(x))


# read-in the test_data
def load_test_data(testing_data_mle_path):

  test_data = pd.read_csv(testing_data_mle_path)

  return test_data


# pre-processs rawdata
def preprocessing(df_test):


  categorical_features = ['term', 'emp_length', 'home_ownership', 'purpose']
  num_features = ['loan_amnt', 'annual_inc', 'percent_bc_gt_75', 'bc_util', 'dti', 'inq_last_6mths', 'revol_util',
                'mths_since_recent_inq', 'total_bc_limit', 'mths_since_last_major_derog', 'tot_cur_bal']

  # apply same data transformation to test data
  df_test['term'] = df_test['term'].str.replace(' ','')  # loan term has two values so treating as categorical
  df_test['int_rate'] = df_test['int_rate'].str[:-1].astype(float)  # interest rate
  df_test['emp_length'] = df_test['emp_length'].apply(conv_emp_length)
  df_test['revol_util'] = df_test['revol_util'].str[:-1].astype(float)  # utilization on credit cards, etc.

  # add dummies
  df_test = pd.get_dummies(df_test, columns=categorical_features, drop_first=True)

  # impute missings
  for feature in num_features:
    df_test[feature].fillna(imputation_values[feature], inplace=True)


  return df_test

# predict on processed data
def prediction(df_test, model):
  cat_features = ['term_60months', 'emp_length_less-than-one-year', 'emp_length_one-to-four-years',
                'emp_length_ten-or-more-years', 'emp_length_unknown',
                'home_ownership_OWN', 'home_ownership_RENT',
                'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_home_improvement',
                'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
                'purpose_other', 'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation'
                ]
  num_features = ['loan_amnt', 'annual_inc', 'percent_bc_gt_75', 'bc_util', 'dti', 'inq_last_6mths', 'revol_util',
                'mths_since_recent_inq', 'total_bc_limit', 'mths_since_last_major_derog', 'tot_cur_bal']

  features = num_features + cat_features
  # score test data
  df_test['proba_score'] = model.predict_proba(df_test[features])[:,1]
  test_auroc = roc_auc_score(df_test['target'], df_test['proba_score'])
  return f"Test AUC is {np.round(test_auroc, 3)}"


def main():
  # main method
  testing_data_mle = 'testing_data_mle.csv'
  raw_data_test_data = load_test_data(testing_data_mle)
  df_test = preprocessing(raw_data_test_data)

  loaded_model, imputation_values = pickle.load(open('RF_model_and_imputation_values.pkl', 'rb'))
  score = prediction(df_test, loaded_model)
  print(score)


if __name__ == '__main__':
  main()