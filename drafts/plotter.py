
#--imports-----------------------------
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

#--data loader-------------------------

def load_data():
  df = pd.read_csv("bank-full.csv", skiprows=0, delimiter=";")
  df = df[["job", "marital", "default", "housing", "poutcome", "y"]]
  
  le = LabelEncoder()
  label = le.fit_transform(df["y"])
  df.drop(["y"], axis=1, inplace=True)
  df["y"] = label

  return df

#--data preprocessing--------------------

def process(df):
  le = LabelEncoder()

  d = {
    "management": 1,
    "technician": 2,
    "entrepreneur": 3,
    "blue-collar": 4,
    "unknown": 5,
    "retired": 6,
    "admin.": 7,
    "services": 8,
    "self-employed": 9,
    "unemployed": 10,
    "housemaid": 11,
    "student": 12,
  }
  numeric_var = {
    "poutcome": {"success": 4, "failure": 3, "other": 2, "unknown": 1},
    "job": d,
    "marital": {"married": 1, "single": 2, "divorced": 3},
  }
  df = df.replace(numeric_var)

  housing = le.fit_transform(df["housing"])
  default = le.fit_transform(df["default"])

  df.drop(["housing", "default"], axis=1, inplace=True)
  df["housing"] = housing
  df["default"] = default

  return df[["job", "poutcome", "housing", "marital", "default"]]

#--plotter function--------------------
def plotly_hist(df):

  fig = px.histogram(df, x='job', y= 'y')
  fig.show()

#--main------------------------------------------------------------

def main():
  df = load_data()
  
  X=process(df)
  y=df['y']

  plotly_hist(df)

if __name__ == "__main__":
  main()

# %%
