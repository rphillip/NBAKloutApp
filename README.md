![logo](https://github.com/rphillip/NBAKloutApp/blob/main/logo.png)
# [NBAKLOUT - http://nbaklout.icu](http://nbaklout.icu) 
A website that combines player in-game basketball stats with social media stats

# Business Case / Purpose
As an advertiser, how do you know you are getting good value for your endorsement?
Who is the ideal athlete to endorse?
Well, we want someone with a huge outreach who can blast our product to the world.
We also want someone who is cheap… If someone makes less than other athletes… maybe we can get away with offering them a smaller contract.

We can use NBAKLOUT to create new stats and metrics to evaluate advertising value.
NBAKLOUT shows how players are similarly ranked with these metrics and can help you evaluate or find value players.
For example:
 - Followers/Dollar - Use this to find a value player with valuable social media outreach
 - Dunks*Likes - Use this to see if a player has dunks that go viral
 - Likes per post * Corner 3 % - Find a sharpshooter for your "Always in your Corner" campaign

And for the big NBA fan, its also found to mess around with. (Users learned that PJ Tucker doesn't have twitter!)

## Features

- 3d/2d interactive graph that has all the players plotted
- Shows similar players for the stats/metrics you chose.
- Gives you the top 3 players for stats/metrics you chose
- Download the stats/metrics data for all the players into a csv file

## How to use
Goto the sidebar and select the player, first stat, times/divide, and/or second stat.
Its that easy! The website will automatically update when you change your selection.

## Framework
![framework](https://github.com/rphillip/NBAKloutApp/blob/main/framework.png)

1. We use Selenium and BeautifulSoup to scrape Basketball Reference, HoopsHype, Instagram. We use APIs for [NBA Stats](https://github.com/swar/nba_api) and [twitter](https://docs.tweepy.org/en/stable/#)
2. We upload that data to a SQL server on AWS Elastic Beanstalk which automatically creates a Postgres SQL Server.
3. We have an AWS EC2 that run Streamlit Framework to operate as the dashboard and pulls data from the Postgres server.

## Directory
- Jupyter Notebook - contains the notebook files that helped create the website (data, scraping, etc)
- Jupyter Notebook/bbr - contains downloaded html of basketball references which were then scraped
- Jupyter Notebook/bbr - contains downloaded html of player's instagrams which were then scraped. The htmls contained json data.
- Docker - contains docker container files to build website
- imgs - contains profile photos for each player

## Possible Future Upgrades
- Further automation with AWS services (Lambda, Glue, Event Bridge, s3)
- Adding a REST API
- Move to serverless model (AWS SAM)
- Adding different leagues and inactive players
