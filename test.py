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
from millify import millify
import praw
import plotly.graph_objects as go
from sklearn.svm import SVR
import tweepy

class IDPrinter(tweepy.Stream):

    def on_status(self, status):
        print(status.id)


printer = IDPrinter(
  "Consumer Key here", "Consumer Secret here",
  "Access Token here", "Access Token Secret here"
)
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
        return components.html(self.text, height=800)
class Reddit(object):
    def __init__(self, s, embed_str=False):
        if not embed_str:
            #https://github.com/reddit-archive/reddit/wiki/oEmbed
            #api = "https://www.reddit.com/oembed?url={}".format(s)
            #api = f"https://www.redditmedia.com/{s}"
            api = "https://www.redditmedia.com/r/nba/comments/qz4n0i/spears_suns_say_frank_kaminsky_iii_has_been/?ref_source=embed&amp;ref=share&amp;embed=true"
            response = requests.get(api)
            self.text = response#.json()["html"]
        else:
            self.text = s
            #https://www.redditmedia.com/r/nba/comments/qz4n0i/spears_suns_say_frank_kaminsky_iii_has_been/?ref_source=embed&amp;ref=share&amp;embed=true
            #<iframe id="reddit-embed" src="https://www.redditmedia.com/r/nba/comments/qz4n0i/spears_suns_say_frank_kaminsky_iii_has_been/?ref_source=embed&amp;ref=share&amp;embed=true" sandbox="allow-scripts allow-same-origin allow-popups" style="border: none;" height="144" width="640" scrolling="no"></iframe>
    def _repr_html_(self):
        return self.text

    def component(self):
        return components.html(self.text, height=600)
class Player(object):
    def __init__(self, stat, values, name, br_name):
        self.values = values
        for value in
        self.stat = stat
        self.name = name
        self.br_name = br_name
    def pretty(self,n):
        if abs(n) < .01:
            return f'{n:.3g}'
        else:
            return millify(n,2)
    def show_selected_player(self):
        c1,c2,c3 = st.columns(3)
        with c2:
            st.image("imgs/{}.jpg".format(self.br_name))
            st.subheader(self.name)

        if len(stat) == 1:
            with c2:
                st.metric(self.stat[0], self.pretty(self.values[0]))

        else:
            d1,d2,d3 = st.columns(3)
            met = self.values[2]
            if self.values[2] is None:
                met = 0
            with d1:
                st.metric(stat[0],self.pretty(self.values[0]))
            with d2:
                st.metric(stat[2],self.pretty(met))
            with d3:
                st.metric(stat[1],self.pretty(self.values[1]))

    def show_similar_player(self, ot=None):
        other = ot
        st.image("imgs/{}.jpg".format(self.br_name))
        st.subheader(self.name)
        met = self.values[2]
        if self.values[2] is None:
            met = 0
        if other is None:
            if len(stat) == 1:
                st.metric(self.stat[0], self.pretty(self.values[0]))
            else:
                st.metric(self.stat[0], self.pretty(self.values[0]))
                st.metric(self.stat[1], self.pretty(self.values[1]))
                st.metric(self.stat[2], self.pretty(met))
        else:
            if len(stat) == 1:
                st.metric(self.stat[0], self.pretty(self.values[0]),self.pretty(float(self.values[0]-other[0])))
            else:
                st.metric(self.stat[0], self.pretty(self.values[0]),self.pretty(float(self.values[0]-other[0])))
                st.metric(self.stat[1], self.pretty(self.values[1]),self.pretty(float(self.values[1]-other[1])))
                st.metric(self.stat[2], self.pretty(met),self.pretty(float(met-other[2])))


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache(allow_output_mutation=True)
def get_connection(path):
    """Put the connection in cache to reuse if path does not change."""
    engine = create_engine(path)
    return engine.connect()

@st.cache(allow_output_mutation=True, hash_funcs={Connection: id})
def get_data(conn):
    query2 = 'select name from stats;'
    query3 = 'select * from defs;'
    #query4 = 'select * from stats;'
    df2 = pd.read_sql(query2,conn)
    df3 = pd.read_sql(query3,conn)
    #df4 = pd.read_sql(query4,conn)
    return (df2, df3)

def get_table(conn, op, s1, s2=None):
    if s2 == None:
        query = f'select name, br_name, {s1} from stats;'
        df = pd.read_sql(query,conn)
        df['rk'] = df[s1].rank(method='first',ascending=False)
        return df
    else:
        query = f'select name, br_name, {s1}, {s2} from stats;'
        df = pd.read_sql(query,conn)
        s3 = ""
        if op == 1:
            s3 = f'{s1}*{s2}'
            df[s3] = df[s1] * df[s2]
        elif op == 2:
            s3 = f'{s1}/{s2}'
            df[s3] = df[s1] / df[s2]
            df.loc[np.isinf(df[s3]), s3] = np.nan

        df['rk'] = df[s3].rank(method='first',ascending=False)
        df[s1+'_rk'] = df[s1].rank(method='first',ascending=False)
        df[s2+'_rk'] = df[s2].rank(method='first',ascending=False)
        return df
def pretty(n):
    if n < .01:
        return f'{n:.3g}'
    else:
        return millify(n,3)
def get_values(df,name,sql):
    v=[]
    v.append(df.loc[name][sql[0]])
    if len(sql) > 1:
        v.append(df.loc[name][sql[1]])
        v.append(df.loc[name][sql[2]])
    return v
def get_similar_players(df,name,s, isone=None):
    onlyone = isone
    out = []
    search = f"{s}_rk"
    if onlyone is True:
        search = f"rk"
    if df.loc[name][search] == 1:
        out.append(df.index[df[search]==2][0])
        out.append(df.index[df[search]==3][0])
    elif df.loc[name][search] == 502:
        out.append(df.index[df[search]==500][0])
        out.append(df.index[df[search]==501][0])
    else:
        ix = df.loc[name][search]
        out.append(df.index[df[search]==ix+1][0])
        out.append(df.index[df[search]==ix-1][0])
    return out
def linr3d(df,st):
    mesh_size = .02
    margin = 0
    X = df[[st[0], st[1]]]
    y = df[st[2]]
    model = SVR(C=1.)
    model.fit(X, y)
    x_min, x_max = X[st[0]].min() - margin, X[st[0]].max() + margin
    y_min, y_max = X[st[1]].min() - margin, X[st[1]].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    return (xrange, yrange, pred)
#Postgres connect
instance = st.secrets["instance"]
dbname = st.secrets["dbname"]
username = st.secrets["username"]
passwd = st.secrets["passwd"]
hostname = st.secrets["hostname"]
portnum = st.secrets["portnum"]
path = "postgresql://{}:{}@{}:{}/{}".format(username,passwd,hostname,portnum,dbname)
stream = (tweepy.Stream(
  st.secrets['twitter_id'], st.secrets['twitter_secret'],
  st.secrets['auth_id'], st.secrets['auth_secret']
))
reddit1 =(praw.Reddit(
    client_id=st.secrets["reddit_id"],
    client_secret=st.secrets["reddit_secret"],
    user_agent=("NBAKlout v1.0 by /u/r_phill https://nbaklout.icu")))

playerlist = ["All"]
statlist = ["None"]
statdf = pd.DataFrame()
#load data
spin = st.spinner('Loading data...')
with spin:
    st.image("logo.png")
    engine = get_connection(path)
    df = get_data(engine)
    playerlist.extend(df[0]['name'].values.tolist())
    statlist.extend(df[1]['desc'].values.tolist())
    statdf = df[1].set_index('desc')


op = 0
playername = st.sidebar.selectbox("Enter Player", playerlist)
stat1 = st.sidebar.selectbox("Stat One", statlist, index=1)
operation = st.sidebar.radio("Operation", ('multiply by','divide by'),index = 1)
if operation == 'multiply by':
    op = 1
elif operation == 'divide by':
    op = 2
stat2 = st.sidebar.selectbox("Stat Two", statlist, index = statlist.index('Dollars: Salary'))
st.sidebar.write("Stats from 2021-22 Season")

stats = pd.DataFrame()
sql = []
values = []
stat = []
with st.spinner('Loading data...'):
    if (stat1 == 'None') and (stat2 == 'None'):
        stats = None
        stat = None
    elif stat1 == 'None':
        sql.append(statdf.loc[stat2]['0'])
        stat.append(stat2)
        stats = get_table(engine,op,sql[0])
    elif stat2 == 'None':
        sql.append(statdf.loc[stat1]['0'])
        stat.append(stat1)
        stats = get_table(engine,op,sql[0])
    else:
        sql.append(statdf.loc[stat1]['0'])
        sql.append(statdf.loc[stat2]['0'])
        stat.append(stat1)
        stat.append(stat2)
        stats = get_table(engine,op,sql[0],sql[1])
        if op == 2:
            sql.append(f'{sql[0]}/{sql[1]}')
            stat.append(f'{stat1}/{stat2}')
        else:
            sql.append(f'{sql[0]}*{sql[1]}')
            stat.append(f'{stat1}*{stat2}')


if playername != "All" and stat is not None:
    #st.dataframe(stats)
    #statone = stats.sort_values
    stats = stats.set_index('name')
    val = get_values(stats,playername,sql)
    p = Player(stat,val,playername,stats.loc[playername]['br_name'])
    p.show_selected_player()

    st.markdown(f"<h1 style='text-align: center; color: #4285F4;'>Similar {stat[0]}</h1>", unsafe_allow_html=True)
    onlyone = False
    if len(stat) == 1:
        onlyone=True
    sim1= get_similar_players(stats,playername,sql[0], onlyone)
    e1,e2 = st.columns(2)
    with e1:
        val = get_values(stats,sim1[0],sql)
        p1= Player(stat,val,sim1[0],stats.loc[sim1[0]]['br_name'])
        p1.show_similar_player(p.values)
    with e2:
        val = get_values(stats,sim1[1],sql)
        p2= Player(stat,val,sim1[1],stats.loc[sim1[1]]['br_name'])
        p2.show_similar_player(p.values)
    if len(stat)>1:
        st.markdown(f"<h1 style='text-align: center; color: #4285F4;'>Similar {stat[1]}</h1>", unsafe_allow_html=True)
        sim2= get_similar_players(stats,playername,sql[1])
        f1,f2 = st.columns(2)
        with f1:
            val = get_values(stats,sim2[0],sql)
            p3= Player(stat,val,sim2[0],stats.loc[sim2[0]]['br_name'])
            p3.show_similar_player(p.values)
        with f2:
            val = get_values(stats,sim2[1],sql)
            p4= Player(stat,val,sim2[1],stats.loc[sim2[1]]['br_name'])
            p4.show_similar_player(p.values)
    subreddit = reddit1.subreddit("nba")
    #rc = subreddit.search(playername, limit=1)
    #for i in rc:
    #    st.write(i.permalink)
    #red = Reddit("https://www.reddit.com/r/nba/comments/stbide/comment/hx2rqx2/?utm_source=share&utm_medium=web2x&context=3").component()

elif stats is not None:
    #st.dataframe(stats)
    f1,f2,f3 = st.columns(3)
    #f1,f2,f3,f4 = st.columns(4)
    if len(stat) > 1 :
        with f3:
            lz = st.checkbox('log z axis', value=True)
        with f2:
            ly = st.checkbox('log y axis', value=True)
    with f1:
        lx = st.checkbox('log x axis', value=True)
    #with f4:
    #    lr = st.checkbox('linear regression', value=False)
    with st.spinner('Loading data...'):
        if len(stat) > 1 :
            fig = (px.scatter_3d(stats, x=sql[0], y=sql[1],
                 hover_name='name', z=sql[2], log_z=lz,
                 log_y=ly, log_x=lx,
                 color = sql[2], color_continuous_scale ="rainbow",
                 labels={
                    sql[0]: stat[0],
                    sql[1]: stat[1],
                    sql[2]: stat[2]
                    },
                 title=stat[2]))
        #    if lr:
        #        mesh = linr3d(stats,sql)
        #        fig.update_traces(marker=dict(size=5))
        #        fig.add_traces(go.Surface(x=mesh[0], y=mesh[1], z=mesh[0], name='prediction surface'))
        else:
             stats = stats.sort_values(by=f"rk")
             fig = (px.histogram(stats, x='name', y=sql[0],
                color = sql[0],
                labels={
                    sql[0]: stat[0],
                    },
                title=stat[0]))
        st.plotly_chart(fig,use_container_width=False)

        st.markdown("<h1 style='text-align: center; color: red;'>Top 3</h1>", unsafe_allow_html=True)
        stats = stats.set_index('name')
        s1 = stats.index[stats['rk']==1][0]
        s2 = stats.index[stats['rk']==2][0]
        s3 = stats.index[stats['rk']==3][0]
        g1, g2, g3 = st.columns(3)
        with g1:
            val = get_values(stats,s1,sql)
            p1= Player(stat,val,s1,stats.loc[s1]['br_name'])
            p1.show_similar_player()
        with g2:
            val = get_values(stats,s2,sql)
            p2= Player(stat,val,s2,stats.loc[s2]['br_name'])
            p2.show_similar_player()
        with g3:
            val = get_values(stats,s3,sql)
            p3= Player(stat,val,s3,stats.loc[s3]['br_name'])
            p3.show_similar_player()
    #    X = stats[sql[0]]
    #    y = stats[sql[1]]
    #    model = sklearn.linear_model.LinearRegression()
    #    model.fit(X, y)
    #    y_pred = model.predict(X)
    #    coefs = model.coef_
    #    intercept = model.intercept_


#loaded

#subreddit = reddit1.subreddit("nba")
#red = Reddit("https://www.reddit.com/r/nba/comments/stbide/comment/hx2rqx2/?utm_source=share&utm_medium=web2x&context=3").component()
#for comment in subreddit.stream.comments(skip_existing=True):
#    red = Reddit(comment.permalink).component()
#t = Tweet("https://twitter.com/NBA/status/1491541849617612806").component()
#
csv = convert_df(stats)
st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name='nbaklout.csv',
     mime='text/csv',
 )
st.dataframe(stats)
