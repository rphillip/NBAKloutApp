![logo](https://github.com/rphillip/NBAKloutApp/blob/main/logo.png)
# [NBAKLOUT - http://nbaklout.icu](http://nbaklout.icu) 
A website that combines player in-game basketball stats with social media stats

# Business Case
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

1. We have scrapers Basketball Reference, HoopsHype, Instagram. We use APIs for NBA Stats and twitter
2. We upload that data to a SQL server on AWS Elastic Beanstalk which automatically creats and RDS.
3. We have an AWS EC2 that run Streamlit Framework to operate as the dashboard and pull data from the server

## Possible Future Upgrades
- Automated scraping with Lambda
- Automated ETL with Glue
- Adding a REST API
- Move to serverless model (AWS SAM)
- Adding different leagues and inactive players
