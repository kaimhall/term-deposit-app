# imports
#---------------------------------------------------------------------
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, request
from IPython.display import display
from joblib import dump, load
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# data wrangler functions
#---------------------------------------------------------------------
def load_data():
    df = pd.read_csv("bank-full.csv", skiprows=0, delimiter=";")
    df = df[["job", "marital", "default", "housing", "poutcome", "y"]]

    le = LabelEncoder()
    label = le.fit_transform(df["y"])
    df.drop("y", axis=1, inplace=True)
    df["y"] = label
    return df

def split_data():
    # get dummies variables
    df_train = pd.get_dummies(
        df, columns=["job", "marital", "default", "housing", "poutcome"]
    )

    X = df_train.iloc[:, 1:]
    y = df_train.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    return [X_train, y_train]

#---------------------------------------------------------------------

# flask app
app = Flask(__name__)

df = load_data()
X, y = split_data()
model = load("linear_svm.joblib")

#score = model.score(X, y)

# list for HTML dropdowns
jobs = sorted(list(df['job'].unique()))
marital = sorted(list(df['marital'].unique()))
default = sorted(list(df['default'].unique()))
housing = sorted(list(df['housing'].unique()))
poutcome = sorted(list(df['poutcome'].unique()))

print(jobs)

@app.route("/")
def index():
    # Main page
    return render_template(
      'index.html',  
      job = jobs, 
      marital = marital,
      default = default, 
      housing = housing,
      poutcome = poutcome
    )

@app.route('/result')

def result():
  variables = {
    'JOB':'job',
  }

  for i in variables:
    if variables[i] in request.args:
      variables[i] = request.args.get(variables[i], '', type= str)
    else:
      return f"Error: No {variables[i]} field provided. Please specify {variables[i]}."
  
  newdf = pd.DataFrame([variables])

  return render_template(
        'result.html', 
        job = variables['JOB'], 
        graphJSON=graphJSON
   )

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8000)
