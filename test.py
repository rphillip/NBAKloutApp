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
    query2 = 'select name from stats;'
    query3 = 'select * from defs;'
    #query4 = 'select * from stats;'
    df = pd.read_sql(query,conn)
    df2 = pd.read_sql(query2,conn)
    df3 = pd.read_sql(query3,conn)
    #df4 = pd.read_sql(query4,conn)
    return (df, df2, df3)

def get_table(conn, s1, s2, op):
    if s1 == 1:
        query = 'select name, br_name, {} from stats;'.format(s1)
        return pd.read_sql(query,conn)
    elif s2 == 1:
        query = 'select name, br_name, {} from stats;'.format(s2)
        return pd.read_sql(query,conn)
    else:
        query = 'select name, br_name, {}, {} from stats;'.format(s1,s2)
        df = pd.read_sql(query,conn)
        s3 = ""
        if op == 1:
            s3 = s1 + '*' + s2
            df[s3] = df[s1] * df[s2]
        elif op == 2:
            s3 = s1 + '/' + s2
            df[s3] = df[s1] / df[s2]
            df.loc[np.isinf(df[s3]), s3] = np.nan

        df['rk'] = df[s3].rank(method='first',ascending=False)
        df[s1+'_rk'] = df[s1].rank(method='first',ascending=False)
        df[s2+'_rk'] = df[s3].rank(method='first',ascending=False)
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
statdf = pd.DataFrame()
#load data
spin = st.spinner('Loading data...')
stats = pd.DataFrame()
with spin:
    engine = get_connection(path)
    df = get_data(engine)
    stats = df[0].copy().sort_values(by='name')
    playerlist.extend(df[1]['name'].values.tolist())
    statlist.extend(df[2]['desc'].values.tolist())
    statdf = df[2].set_index('desc')

op = 0
playername = st.sidebar.selectbox("Enter Player", playerlist)
stat1 = st.sidebar.selectbox("Stat One", statlist, index=1)
operation = st.sidebar.radio("Operation", ('multiply by','divide by'),index = 1)
if operation == 'multiply by':
    op = 1
elif operation == 'divide by':
    op = 2
stat2 = st.sidebar.selectbox("Stat Two", statlist, index = statlist.index('Dollars: Salary'))

stats = pd.DataFrame()
sql1 =statdf.loc[stat1]['0']
sql2 =statdf.loc[stat2]['0']
if op == 2:
    sql3 = sql1 + '/' + sql2
    stat3 = stat1 +'/' + stat2
else:
    sql3 = sql1 + '*' + sql2
    stat3 = stat1 +'*' + stat2
with st.spinner('Loading data...'):
    if (stat1 == 'All') and (stat1 == 'All'):
        stats = stats
    elif stat1 == 'All':
        stats = get_table(engine,1,sql2,op)
    elif stat2 == 'All':
        stats = get_table(engine,sql1,1,op)
    else:
        stats = get_table(engine,sql1,sql2,op)


#stats['rk'] = stats['followers per dollar'].rank(method='first',ascending=False)
#stats['stat1_rk'] = stats['twit_followers_count'].rank(method='first',ascending=False)
#stats['stat2_rk'] = stats['2021-22'].rank(method='first',ascending=False)
if playername != "All":
    #st.dataframe(stats)
    #statone = stats.sort_values
    stats = stats.set_index('name')
    c1,c2,c3 = st.columns(3)
    with c2:
        st.image("imgs/{}.jpg".format(stats.loc[playername]['br_name']))
        st.write(playername)

    d1,d2,d3 = st.columns(3)
    with d1:
        st.metric(stat1,stats.loc[playername][sql1])
    with d2:
        st.metric(stat3,stats.loc[playername][sql3])
    with d3:
        st.metric(stat2,stats.loc[playername][sql2])

    st.markdown("<h1 style='text-align: center; color: red;'>Similar Followers</h1>", unsafe_allow_html=True)

    s1 = ""
    s2 = ""
    if stats.loc[playername][sql1+'_rk'] == 1:
        s1 = stats.index[stats[sql1+'_rk']==2][0]
        s2 = stats.index[stats[sql1+'_rk']==3][0]
    else:
        ix = stats.loc[playername][sql1+'_rk']
        s1 = stats.index[stats[sql1+'_rk']==ix+1][0]
        s2 = stats.index[stats[sql1+'_rk']==ix-1][0]
    e1,e2 = st.columns(2)
    with e1:
        st.image("imgs/{}.jpg".format(stats.loc[s1]['br_name']))
        st.write(s1)
        st.write("{}: {}".format(stat1,stats.loc[s1][sql1]))
        st.write("{}: {}".format(stat2,stats.loc[s1][sql2]))
        st.write("{}: {}".format(stat3,stats.loc[s1][sql3]))
    with e2:
        st.image("imgs/{}.jpg".format(stats.loc[s2]['br_name']))
        st.write(s2)
        st.write("{}: {}".format(stat1,stats.loc[s2][sql1]))
        st.write("{}: {}".format(stat2,stats.loc[s2][sql2]))
        st.write("{}: {}".format(stat3,stats.loc[s2][sql3]))
    st.markdown("<h1 style='text-align: center; color: red;'>Similar Salary</h1>", unsafe_allow_html=True)
    t1 = ""
    t2 = ""
    if stats.loc[playername][sql2+'_rk'] == 1:
        t1 = stats.index[stats[sql2+'_rk']==2][0]
        t2 = stats.index[stats[sql2+'_rk']==3][0]
    else:
        ix = stats.loc[playername][sql2+'_rk']
        t1 = stats.index[stats[sql2+'_rk']==ix+1][0]
        t2 = stats.index[stats[sql2+'_rk']==ix-1][0]
    f1,f2 = st.columns(2)
    with f1:
        st.image("imgs/{}.jpg".format(stats.loc[t1]['br_name']))
        st.write(s1)
        st.write("{}: {}".format(stat1,stats.loc[t1][sql1]))
        st.write("{}: {}".format(stat2,stats.loc[t1][sql2]))
        st.write("{}: {}".format(stat3,stats.loc[t1][sql3]))
    with f2:
        st.image("imgs/{}.jpg".format(stats.loc[t2]['br_name']))
        st.write(s2)
        st.write("{}: {}".format(stat1,stats.loc[t2][sql1]))
        st.write("{}: {}".format(stat2,stats.loc[t2][sql2]))
        st.write("{}: {}".format(stat3,stats.loc[t2][sql3]))

else:
    #st.dataframe(stats)
    with st.spinner('Loading data...'):
        fig = (px.scatter_3d(stats, x=sql1, y=sql2,
             hover_name='name',z=sql3,
             color = sql3, color_continuous_scale ="rainbow",
             labels={
                sql1: stat1,
                sql2: stat2,
                sql3: stat3
                },
             title=stat3))
        st.plotly_chart(fig,use_container_width=False)
    st.markdown("<h1 style='text-align: center; color: red;'>Top 3</h1>", unsafe_allow_html=True)
    stats = stats.set_index('name')
    g1, g2, g3 = st.columns(3)
    s1 = stats.index[stats['rk']==1][0]
    s2 = stats.index[stats['rk']==2][0]
    s3 = stats.index[stats['rk']==3][0]
    with g1:
        st.image("imgs/{}.jpg".format(stats.loc[s1]['br_name']))
        st.write(s1)
        st.write("{}: {}".format(stat1,stats.loc[s1][sql1]))
        st.write("{}: {}".format(stat2,stats.loc[s1][sql2]))
        st.write("{}: {}".format(stat3,stats.loc[s1][sql3]))
    with g2:
        st.image("imgs/{}.jpg".format(stats.loc[s2]['br_name']))
        st.write(s2)
        st.write("{}: {}".format(stat1,stats.loc[s2][sql1]))
        st.write("{}: {}".format(stat2,stats.loc[s2][sql2]))
        st.write("{}: {}".format(stat3,stats.loc[s2][sql3]))
    with g3:
        st.image("imgs/{}.jpg".format(stats.loc[s3]['br_name']))
        st.write(s3)
        st.write("{}: {}".format(stat1,stats.loc[s3][sql1]))
        st.write("{}: {}".format(stat2,stats.loc[s3][sql2]))
        st.write("{}: {}".format(stat3,stats.loc[s3][sql3]))



#loaded

t = Tweet("https://twitter.com/NBA/status/1491541849617612806").component()
st.dataframe(stats)

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
