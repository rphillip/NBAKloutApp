import streamlit as st
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import inspect
import psycopg2
import pandas as pd
import requests
import streamlit.components.v1 as components
import plotly.express as px

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

st.image("logo.png")


#DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache(allow_output_mutation=True)
def get_connection(path):
    """Put the connection in cache to reuse if path does not change."""
    engine = create_engine(path)
    return engine.connect()

@st.cache(allow_output_mutation=True, hash_funcs={Connection: id})
def get_data(conn):
    query = 'select * from demo;'
    df = pd.read_sql(query,conn)
    return df
#Postgres connect
instance = st.secrets["instance"]
dbname = st.secrets["dbname"]
username = st.secrets["username"]
passwd = st.secrets["passwd"]
hostname = st.secrets["hostname"]
portnum = st.secrets["portnum"]
path = "postgresql://{}:{}@{}:{}/{}".format(username,passwd,hostname,portnum,dbname)

playerlist = ["All"]
statlist = ["None"]
#load data
spin = st.spinner('Loading data...')
stats = pd.DataFrame()
with spin:
    engine = get_connection(path)
    df = get_data(engine)
    stats = df.copy().sort_values(by='name')
    playerlist.extend(stats.set_index('name').index.to_list())


#stats output
playername = st.sidebar.selectbox("Enter Player", playerlist)
stat1 = st.sidebar.text_input("Enter Stat")
stat2 = st.sidebar.text_input("Enter Stat: Per Game default")
stats = stats.set_index('name')
stats['rk'] = stats['followers per dollar'].rank(method='first',ascending=False)
stats['stat1_rk'] = stats['twit_followers_count'].rank(method='first',ascending=False)
stats['stat2_rk'] = stats['2021-22'].rank(method='first',ascending=False)
if playername != "All":
    #st.dataframe(stats)
    #statone = stats.sort_values
    c1,c2,c3 = st.columns(3)
    with c2:
        st.image("imgs/{}.jpg".format(stats.loc[playername]['br_name']))
        st.write(playername)

    d1,d2,d3 = st.columns(3)
    with d1:
        st.metric("Followers",stats.loc[playername]['twit_followers_count'])
    with d2:
        st.metric("Followers/$",stats.loc[playername]['followers per dollar'])
    with d3:
        st.metric("Salary",stats.loc[playername]['2021-22'])

    st.markdown("<h1 style='text-align: center; color: red;'>Similar Followers</h1>", unsafe_allow_html=True)

    s1 = ""
    s2 = ""
    if stats.loc[playername]['stat1_rk'] == 1:
        s1 = stats.index[stats['stat1_rk']==2][0]
        s2 = stats.index[stats['stat1_rk']==3][0]
    else:
        ix = stats.loc[playername]['stat1_rk']
        s1 = stats.index[stats['stat1_rk']==ix+1][0]
        s2 = stats.index[stats['stat1_rk']==ix-1][0]
    e1,e2 = st.columns(2)
    with e1:
        st.image("imgs/{}.jpg".format(stats.loc[s1]['br_name']))
        st.write(s1)
        st.write("Followers: {}".format(stats.loc[s1]['twit_followers_count']))
        st.write("Salary: {}".format(stats.loc[s1]['2021-22']))
        st.write("Followers/$: {}".format(stats.loc[s1]['followers per dollar']))
    with e2:
        st.image("imgs/{}.jpg".format(stats.loc[s2]['br_name']))
        st.write(s2)
        st.write("Followers: {}".format(stats.loc[s2]['twit_followers_count']))
        st.write("Salary: {}".format(stats.loc[s2]['2021-22']))
        st.write("Followers/$: {}".format(stats.loc[s2]['followers per dollar']))
    st.markdown("<h1 style='text-align: center; color: red;'>Similar Salary</h1>", unsafe_allow_html=True)
    t1 = ""
    t2 = ""
    if stats.loc[playername]['stat2_rk'] == 1:
        t1 = stats.index[stats['stat2_rk']==2][0]
        t2 = stats.index[stats['stat2_rk']==3][0]
    else:
        ix = stats.loc[playername]['stat2_rk']
        t1 = stats.index[stats['stat2_rk']==ix+1][0]
        t2 = stats.index[stats['stat2_rk']==ix-1][0]
    f1,f2 = st.columns(2)
    with f1:
        st.image("imgs/{}.jpg".format(stats.loc[t1]['br_name']))
        st.write(t1)
        st.write("Followers: {}".format(stats.loc[t1]['twit_followers_count']))
        st.write("Salary: {}".format(stats.loc[t1]['2021-22']))
        st.write("Followers/$: {}".format(stats.loc[t1]['followers per dollar']))
    with f2:
        st.image("imgs/{}.jpg".format(stats.loc[t2]['br_name']))
        st.write(t2)
        st.write("Followers: {}".format(stats.loc[t2]['twit_followers_count']))
        st.write("Salary: {}".format(stats.loc[t2]['2021-22']))
        st.write("Followers/$: {}".format(stats.loc[t2]['followers per dollar']))

else:
    #st.dataframe(stats)
    with st.spinner('Loading data...'):
        fig = (px.scatter_3d(df, x='2021-22', y='twit_followers_count',
             hover_name='name',z='followers per dollar',
             color = 'followers per dollar', color_continuous_scale ="rainbow"))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("<h1 style='text-align: center; color: red;'>Top 3</h1>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    s1 = stats.index[stats['rk']==1][0]
    s2 = stats.index[stats['rk']==2][0]
    s3 = stats.index[stats['rk']==3][0]
    with g1:
        st.image("imgs/{}.jpg".format(stats.loc[s1]['br_name']))
        st.write(s1)
        st.write("Followers: {}".format(stats.loc[s1]['twit_followers_count']))
        st.write("Salary: {}".format(stats.loc[s1]['2021-22']))
        st.write("Followers/$: {}".format(stats.loc[s1]['followers per dollar']))
    with g2:
        st.image("imgs/{}.jpg".format(stats.loc[s2]['br_name']))
        st.write(s2)
        st.write("Followers: {}".format(stats.loc[s2]['twit_followers_count']))
        st.write("Salary: {}".format(stats.loc[s2]['2021-22']))
        st.write("Followers/$: {}".format(stats.loc[s2]['followers per dollar']))
    with g3:
        st.image("imgs/{}.jpg".format(stats.loc[s3]['br_name']))
        st.write(s3)
        st.write("Followers: {}".format(stats.loc[s3]['twit_followers_count']))
        st.write("Salary: {}".format(stats.loc[s3]['2021-22']))
        st.write("Followers/$: {}".format(stats.loc[s3]['followers per dollar']))



#loaded

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
