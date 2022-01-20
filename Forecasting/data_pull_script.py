#/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
import pandas as pd
import os
from datetime import datetime

now = datetime.now()

file_name = 'covid_data.csv'
if os.path.exists(file_name):
    os.remove(file_name)
    df = pd.read_csv(r"https://covid.ourworldindata.org/data/owid-covid-data.csv")
    df.to_csv(file_name)
    print('script sucessfully deleted and updated file at: ', now)
else:
    df = pd.read_csv(r"https://covid.ourworldindata.org/data/owid-covid-data.csv")
    df.to_csv(file_name)
    print('script sucessfully downloaded file at: ', now)

