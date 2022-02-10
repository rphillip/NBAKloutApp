import streamlit as st
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import inspect
import psycopg2
import pandas as pd

st.title('NBAKlout')

#DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

#@st.cache
#def load_data(nrows):
#    data = pd.read_csv(DATA_URL, nrows=nrows)
#    lowercase = lambda x: str(x).lower()
#    data.rename(lowercase, axis='columns', inplace=True)
#    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#    return data

#Postgres connect
data_load_state = st.text('Loading data...')
instance = st.secrets["instance"]
dbname = st.secrets["dbname"]
username = st.secrets["username"]
passwd = st.secrets["passwd"]
hostname = st.secrets["hostname"]
portnum = st.secrets["portnum"]
engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(username,passwd,hostname,portnum,dbname))
session = scoped_session(sessionmaker(bind=engine))
inspector = inspect(engine)
conn = engine.connect()
query = 'select * from public."Salary";'
df = pd.read_sql(query,conn)
st.dataframe(df)
data_load_state.text("Done! (using st.cache)")
#data_load_state = st.text('Loading data...')
#data = load_data(10000)
#data_load_state.text("Done! (using st.cache)")

#if st.checkbox('Show raw data'):
#    st.subheader('Raw data')
#    st.write(data)

#st.subheader('Number of pickups by hour')
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#st.bar_chart(hist_values)

# Some number in the range 0-23
#hour_to_filter = st.slider('hour', 0, 23, 17)
#filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

#st.subheader('Map of all pickups at %s:00' % hour_to_filter)
#st.map(filtered_data)
