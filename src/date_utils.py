import datetime
import yfinance as yf
import pandas as pd
import matplotlib as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import json

BASE_DATE = pd.Timestamp(2000,1,1)
DATE_FORMAT = '%Y-%m-%d'

def date_to_int(date):
    return np.array((date-BASE_DATE)).astype('timedelta64[D]').astype(int)

def string_to_date(string_date):
    return datetime.datetime.strptime(string_date, DATE_FORMAT)