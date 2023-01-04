# imports
#---------------------------------------------------------------------

import pandas as pd
from flask import Flask, render_template, request
from joblib import load
from sklearn.preprocessing import LabelEncoder

# plotly installs
import plotly
import plotly.graph_objects as go
import json

# data loader
#---------------------------------------------------------------------
def load_data():
    df = pd.read_csv("bank-full.csv", skiprows=0, delimiter=";")
    df = df[["job", "marital", "default", "housing", "poutcome", "y"]]
    
    le = LabelEncoder()
    label = le.fit_transform(df["y"])
    df.drop(["y"], axis=1, inplace=True)
    df["y"] = label

    return df

# data preprocessing
#---------------------------------------------------------------------
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

# --plotter function--------------------
def plotly_hist(df, jobtype, poutcometype):

  plot_series1 = df["y"]
  plot_series2 = df[(df.poutcome == poutcometype) & (df.job == jobtype)]["y"].astype(int)

  fig = go.Figure()

  fig.add_trace(
    go.Histogram(
      x=plot_series1,
      name="all data",
      nbinsx=2,
      histfunc="sum",
      histnorm="probability density",
    ),
  )
  fig.add_trace(
    go.Histogram(
      x=plot_series2,
      name="selection",
      nbinsx=2,
      histfunc="sum",
      histnorm="probability density",
    ),
  )

  fig.update_layout(
    bargap=0.2,
    title_text=f"Probability distibution",  # title of plot
    xaxis_title_text="makes deposit",  # xaxis label
    yaxis_title_text="probability density",  # yaxis label
    bargroupgap=0.1,
    template="seaborn",
  )

  fig.update_xaxes(ticktext=["No", "Yes"], tickvals=[0, 1])
  fig.update_traces(opacity=0.9)
  
  return fig

# flask app
app = Flask(__name__)

df = load_data()
model = load("rbf_svm.joblib")

#score = model.score(X, y)

# list for HTML dropdowns
jobs = sorted(list(df['job'].unique()))
marital = sorted(list(df['marital'].unique()))
default = sorted(list(df['default'].unique()))
housing = sorted(list(df['housing'].unique()))
poutcome = sorted(list(df['poutcome'].unique()))

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
    'job':'job',
    'marital': 'marital',
    'default': 'default',
    'housing': 'housing',
    'poutcome': 'poutcome',
  }

  for i in variables:
    if variables[i] in request.args:
      variables[i] = request.args.get(variables[i], '', type= str)
    else:
      return f"Error: No {variables[i]} field provided. Please specify {variables[i]}."
  
  df_new = pd.DataFrame([variables])

  df_new_train = process(df_new)
  prediction = model.predict(df_new_train)
  pred_fnl = (
      "will not make a term deposit"
      if prediction[0] == 0
      else "will make a term deposit"
  )

  prob_plot = plotly_hist(df, jobtype = variables['job'], poutcometype=variables['poutcome'])
  graphJSON = json.dumps(prob_plot, cls=plotly.utils.PlotlyJSONEncoder)
  
  #print(graphJSON)

  return render_template(
    'result.html', 
    job = variables['job'],
    marital = variables['marital'],
    default = variables['default'],
    housing = variables['housing'],
    poutcome = variables['poutcome'],
    pred_fnl = pred_fnl,
    graphJSON= graphJSON
   )

if __name__ == "__main__":
  app.run(debug=True)
  #flask --app app.py --debug run
