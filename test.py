import streamlit as st
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import inspect
import psycopg2
import pandas as pd
import requests
import streamlit.components.v1 as components

class Tweet(object):
    def __init__(self, s, embed_str=False):
        if not embed_str:
            # Use Twitter's oEmbed API
            # https://dev.twitter.com/web/embedded-tweets
            api = "https://publish.twitter.com/oembed?url={}".format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text

    def component(self):
        return components.html(self.text, height=600)

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
player = st.sidebar.text_input("Enter Player")
stat1 = st.sidebar.text_input("Enter Stat")
stat2 = st.sidebar.text_input("Enter Stat: Per Game default")
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

st.text("Top 5")
df['max_rank'] = df['2021-22'].rank(method='max')
st.dataframe(df.sort_values(by=['max_rank'], ascending=False).head(5))
st.text("Bottom 5")
st.dataframe(df.sort_values(by=['max_rank']).tail(5))

data_load_state.text("Stats loaded")

t = Tweet("https://twitter.com/NBA/status/1491541849617612806").component()
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
