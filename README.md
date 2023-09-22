# NBA Player Performance
**Andreas Nikolaidis** 

_September 2023_

- [Introduction](#introduction)
- [Import & Clean Data](#import)
- [Analysis & Correlations](#analysis)
- [OLS & Approximate Value (AV)](#ols_av)

## [Introduction](#introduction)
In this project I utilize python to analyze data of NBA players in the league and compare their performance against their monetary value (salary) i.e. Which players are undervalued / overvalued. We take a look at some interesting statistics, correlations and then utilize scikit-learn to see if we can accurately predict players next year salaries based on their previous year's performance. (Note: Most contracts and salaries are negotiated with 2-5 years attached so we are looking to see if that future value holds)

## [Import & Clean Data](#import)
Start by importing all the necessary packages into Python:
```python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_style('whitegrid')
%matplotlib inline

# Import for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

#Regex
import re
```
Import Stats & Clean
```python
stats_23 = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2023_totals.html')
```
```python
#put names of columns into list to assign headers into DF
columns = stats_23[0].columns
#create all_player_data
df23 = pd.DataFrame(stats_23[0], columns = columns)
df23.head()
```
<img width="934" alt="df_head" src="https://github.com/atnikola/nba-player-performance/assets/38530617/a2648903-544f-4392-848a-426883fb31b0">

Clean up
```python
df23.drop('Rk',axis=1, inplace=True)
df23.rename(columns={'Tm':'Team', 'Pos':'Position'},inplace=True)
```
```python
#look for player
df23[df23['Player']=='LeBron James']
```
<img width="952" alt="lebron" src="https://github.com/atnikola/nba-player-performance/assets/38530617/b9bc6757-cb0a-48f3-bc39-b668c4b677ad">

Get Salary Data & Clean
```python
#get data for salaries
contracts = pd.read_html('https://www.basketball-reference.com/contracts/players.html') #import
contracts = contracts[0].droplevel(0, axis=1)
contracts.rename(columns={'Tm':'Team'},inplace=True)
contracts.drop('Rk',axis=1, inplace=True)
```

Merge Data
```python
all_player_data = pd.merge(df23, contracts, on=['Player','Team'], how='left')
```

Check Players
```python
all_player_data[all_player_data['Player']=='Jaylen Brown']
```
<img width="1303" alt="jaylen" src="https://github.com/atnikola/nba-player-performance/assets/38530617/86b068d3-c0a6-4eed-99c2-1e75e14b5ab1">

```python
#multiple rows for trader player
all_player_data[all_player_data['Player']=='Russell Westbrook']
```
<img width="1231" alt="westbrook" src="https://github.com/atnikola/nba-player-performance/assets/38530617/749afa20-9464-4b14-990e-900ef1b3df8b">

Data Cleaning
```python
#Remove these rows that have 'Tm' in there..
all_player_data = all_player_data[all_player_data['Team'].str.contains('Tm')==False]

# Remove $ sign and convert to int (for all years)
all_player_data['2023-24']=[int(re.sub(r'[^\d.]+', '', s)) if isinstance(s, str) else s for s in all_player_data['2023-24'].values]

#set player to index
all_player_data = all_player_data.set_index('Player')
all_player_data.index.name = None
all_player_data.head()

#change object to numeric for stats
all_player_data[['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', \
                 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']] \
                    = all_player_data[['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', \
                    '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].apply(pd.to_numeric)

#Add Points per Game as a stat
all_player_data['PPG'] = (all_player_data['PTS']/all_player_data['G']).round(1)
```
Next since there are so many different variables, write a function to make filtering easier:
```python
def top_n(all_player_data, category, n):
    return (all_player_data.sort_values(category, ascending=False)[['Team', category]].head(n))

top_n(all_player_data, 'Guaranteed', 10)
```
<img width="271" alt="highest guaranteed" src="https://github.com/atnikola/nba-player-performance/assets/38530617/eba65454-99e5-4ad5-aec1-a98027bff019">

Can check for any category now:
```python
top_n(all_player_data, 'PTS', 5)
```
<img width="262" alt="pts" src="https://github.com/atnikola/nba-player-performance/assets/38530617/772d72dd-110c-47a9-912d-95c40eb370f2">

## [Analysis & Correlations](#analysis)

Let's take a look at the salary distributions:
```python
sns.histplot(data=all_player_data,x='2023-24',bins=30,color='#002F87').set(title='2023-24 Salary Distribution');
plt.xlabel('Salary (M)')
plt.ylabel('Number of players')
plt.show()
```
![salary dist](https://github.com/atnikola/nba-player-performance/assets/38530617/5247f659-d253-47ff-8f07-c6d76860bf91)

Stat distributions
```python
#Distribution by stats:
f, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.despine(left=True)

# Histograms
sns.distplot(all_player_data['PTS'], color = "b", ax = axes[0, 0])
sns.distplot(all_player_data['AST'], color = "r", ax = axes[0, 1])
sns.distplot(all_player_data['STL'], color = "g", ax = axes[1, 0])
sns.distplot(all_player_data['TRB'], color = "m", ax = axes[1, 1])
```
![stat dist](https://github.com/atnikola/nba-player-performance/assets/38530617/96df4288-d148-4d38-b12d-ce1d65427759)

```python
# Set up figure
f, axes = plt.subplots(2, 2, figsize=(15, 10))

# Regressionplot
sns.regplot(x = all_player_data['PTS'], y = all_player_data['2023-24'], color="b", ax=axes[0, 0])
sns.regplot(x = all_player_data['AST'], y = all_player_data['2023-24'], color="r", ax=axes[0, 1])
sns.regplot(x = all_player_data['STL'], y = all_player_data['2023-24'], color="g", ax=axes[1, 0])
sns.regplot(x = all_player_data['TRB'], y = all_player_data['2023-24'], color="m", ax=axes[1, 1])

plt.show()
```
![stat corr](https://github.com/atnikola/nba-player-performance/assets/38530617/05840a4a-a06c-49a3-9b73-0cfef3cf212a)

Salary v Position
```python
# Relationship with Position
sns.boxplot(all_player_data, x = 'Position', y = '2023-24', order = ['PG', 'SG', 'SF', 'PF', 'C'])
plt.show()
```
![salary v position](https://github.com/atnikola/nba-player-performance/assets/38530617/ab6b9657-93a7-4d4f-b5e2-68b987028f23)

```python
PPG_byPosition = all_player_data.groupby("Position").mean()["PPG"]

df_plot = pd.DataFrame(columns=["Position","PPG","colors"]) #square brackets
x = PPG_byPosition.values
df_plot["Position"] = PPG_byPosition.index
df_plot['PPG'] = (x - x.mean())/x.std()
df_plot['colors'] = ['red' if x < 0 else 'green' for x in df_plot['PPG']]
df_plot.sort_values('PPG', inplace=True)

plt.figure(figsize=(14,14), dpi=80)
plt.hlines(y=df_plot.Position, xmin=0, xmax=df_plot.PPG)
for x, y, tex in zip(df_plot.PPG, df_plot.Position, df_plot.PPG):
    t = plt.text(
        x, y, round(tex, 2),
        horizontalalignment='right' if x < 0 else 'left',
        verticalalignment='center',
        fontdict={'color':'red' if x < 0 else 'green', 'size':15})

plt.yticks(df_plot.Position, df_plot.Position, fontsize=12)
plt.title('Diverging Text Bars of PPG by Position', fontdict={'size':20})
plt.xlim(-3, 3)
plt.show()
```
![diverging ppg](https://github.com/atnikola/nba-player-performance/assets/38530617/43d82459-8e9f-433b-a04c-72fc6360884d)

Let's take a look at the correlation heatmap - but first, let's get rid of any salaries beyond 2024 initially as anything beyond that year may be skewed.

```python
#Correlation matrix
sns.set(style = "white")
cor_matrix = all_player_data.loc[:, 'Age': '2023-24'].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cor_matrix, dtype = bool))

plt.figure(figsize = (10, 7))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
```
![heatmap corr](https://github.com/atnikola/nba-player-performance/assets/38530617/7116d165-71e2-4aae-8ed9-75dfcae68c57)

```python
corr_matrix.sort_values('2023-24', ascending=False)['2023-24'].head(15)
```
<img width="228" alt="Screenshot 2023-09-22 at 11 53 51" src="https://github.com/atnikola/nba-player-performance/assets/38530617/e7625a7f-5f88-4338-8418-22ec1774fd45">

## [OLS & Approximate Value (AV)](#ols_av)

```python
from sklearn.linear_model import LinearRegression
reg_df = all_player_data[['2023-24', 'PPG']].dropna()
```
Then, fit the model with the predictor, independent variable (X) = PTS and the dependent variable (Y) = salary. 
Side note: Set fit_intercept=False since players cannot have less than $0 salary or score less than 0 PTS:
```python
X = reg_df['PPG'].values.reshape(-1,1)
Y = reg_df['2023-24'].values.reshape(-1,1)
reg = LinearRegression(fit_intercept=False).fit(X,Y)
y_pred = reg.predict(X)
plt.figure(figsize=(12, 6))
plt.scatter(X, Y)
plt.plot(X, y_pred, color='red')
plt.xlabel("Points per game (2022-2023)", fontsize = 12)
plt.ylabel("Salary (2023)", fontsize = 12)
plt.title('Salary vs PTS - Linear Regression', fontsize=16);
```
![salary v ppg](https://github.com/atnikola/nba-player-performance/assets/38530617/5b210885-7954-4e85-9346-ab1d22bfba85)

Consistent with the positive correlation seen previously, a regression line with a positive slope is fitted to the data. This is a very linear relationship, so how can I find the best value for salary? I will use what's known as the APPROXIMATE VALUE.

```python
# Extract the slope of the line:
print(reg.coef_)
```
<img width="180" alt="slope" src="https://github.com/atnikola/nba-player-performance/assets/38530617/06312373-bf1a-4b32-b963-b5190152bed0">

Just by looking at the fit above, you can see that the residuals will be heteroskedastic, the standard deviations of a predicted variable (salary), monitored over different values of an independent variable (PPG), are non-constant. With heteroskedasticity, the tell-tale sign upon visual inspection of the residual errors is that they will tend to fan out over time, as depicted in the graph above.
 
There are also a small number of players with high career points per game, but low salaries in the bottom right corner of the plot which are skewing the regression line.

The value of the slope is ~1.1M, which means that on average, for each unit of increase in points per game, the predicted salary paid to a player increases by $1.1M.... Free throws FTW.

High ROI Players

```python
#First look at the top players in terms of PTS in descending order
reg_df.sort_values('PPG', ascending=False).head(10)
```
<img width="292" alt="pts order" src="https://github.com/atnikola/nba-player-performance/assets/38530617/5fd4e6aa-ae6c-440f-8163-8df1e75f0ad0">

Find average salary - multiply by 2 since average will be heavily skewed since most players are not payed as much as a select few superstars.
Find average points - can multiply by 1.5-2 since average will be heavily skewed since most players do not score as much as a select few superstars

```python
reg_df['2023-24'].mean()*2
reg_df['PPG'].mean()
```
APPROXIMATE VALUE (AV) 
Credits Formula = (Points)+(Rebounds)+(Assists)+(Steal)+(Blocks)-(Field Goals Missed)-(Free Throws Missed)-(Turnovers)

AV Formula = (Credits * 0.75 )/21

The formula does not subtract PF (personal fouls) - My guess is fouls could be neutral/positive depending on the situation in the game.

```python
#Calculate per-minute play by dividing the salary by the actual total minutes played by each player.
## **** Since the salaries are projected for 2023-24 (could not obtain 2022-23 data) the future value will be based on last season's performance. *** ##

#Credits Formula
all_player_data["Credits"] = all_player_data["PTS"]+all_player_data["TRB"]+all_player_data["AST"]+all_player_data["STL"]+ all_player_data["BLK"]-(all_player_data["FGA"]-all_player_data["FG"])-(all_player_data["FTA"]-all_player_data["FT"])-all_player_data["TOV"]

all_player_data["AV"]=(all_player_data["Credits"]**(0.75))/21

all_player_data["$/mp"] = (all_player_data["2023-24"]/all_player_data["MP"]).round(2)
```
```python
#Exploring the relationship between a player’s value and pay-per-minute, you can immediately see the skewness of the distribution. 
#Most players’ AV are distributed within the $0–$15k / minute bands, with outliers taking place on the right-end side showing several players having low AV and high $/mp. AKA "Low Performers"

all_player_data.plot(kind="scatter",x="$/mp",y="AV")
plt.show()
```
![$permin v av](https://github.com/atnikola/nba-player-performance/assets/38530617/e52e6088-f82d-4373-8977-362bfa895e4d)

```python
all_player_data['$/mp'].describe().apply(lambda x: format(x, 'f')) #apply function to remove scientific notation
```
<img width="205" alt="skew 75%" src="https://github.com/atnikola/nba-player-performance/assets/38530617/2327b967-665e-450e-963b-c81dbf13a081">

You can see how skewed the data is, as over 75% of the data points are less than or equal to ~13K, with the mean located at ~12K.

This indicates that there are several outliers - players whose per-minute salary is too high given the excessively low amount of games they have played in the season (Maybe injuries)?

As one example, use the 75th percentile ($13K per-minute). 

```python
#lazy way to plot vertical x axis - find mean of "$/min"
all_player_data['$/mp'][(all_player_data["$/mp"]<=13639) & (all_player_data["$/mp"] > 0)].mean()

#lazy way to plot vertical x axis - find mean of "AV"
all_player_data['AV'][(all_player_data["$/mp"]<=13639) & (all_player_data["$/mp"] > 0)].mean()
```

High-level player clustering

The overall correlation seems to be somewhat positive. Let’s now identify clusters of players within the high-end of performance and within the low-end of $ per-minute played.

Using mean values, split the above graph into quadrants to see each data point’s position relative to the mean value for both the x and y axis.

To do so, use the plt.axvline and plt.axhline methods to draw vertical and horizontal lines which identify the quadrants relative to the averages.

```python
all_player_data[(all_player_data["$/mp"]<=13639) & (all_player_data["$/mp"] > 0)].plot(kind="scatter",x="$/mp",y="AV", color='tab:blue')
plt.axvline(x=6145.783277076966,c="tab:red") #exact overkill 
plt.axhline(y=6.698552562778662,c="tab:red") #exact overkill 
plt.show()
```

The 'absolute' best value players are located in the upper left quadrant.

```python
#General highest AV value
all_player_data.sort_values('AV', ascending=False)['AV'].head(10)
```
<img width="281" alt="Screenshot 2023-09-22 at 12 02 05" src="https://github.com/atnikola/nba-player-performance/assets/38530617/d4d51d64-cfd7-4c77-8841-1d588039752e">

Create a separate dataframe filtered by only upper left quadrant players
```python
underrated = all_player_data[(all_player_data["$/mp"]<=6146) & (all_player_data["AV"]>= 6.69) & (all_player_data["MP"]> 0)]
underrated[['Age','Position','AV','$/mp']].sort_values('AV', ascending=False).head(15)
```
<img width="382" alt="Screenshot 2023-09-22 at 12 02 51" src="https://github.com/atnikola/nba-player-performance/assets/38530617/3f03c4c1-64af-49f0-ae41-eff6c4c1fee7">

I knew Anthony Edwards & Evan Mobley were underrated but... wow

Let's create 3 total clusters of players:
1. Underrated: High AV +  Low Cost Players
2. Superstars: High AV +  High Cost Players
3. Overrated :  Low AV +  High Cost Players

```python
superstars = all_player_data[(all_player_data["$/mp"]>=6146) & (all_player_data["AV"]>= 6.69)]
superstars[['Age','Position','AV','$/mp']].sort_values('AV', ascending=False).head(15)
```
<img width="417" alt="superstars" src="https://github.com/atnikola/nba-player-performance/assets/38530617/43ef22b4-f233-48ac-9b6f-85206627a6c2">

```python
overrated = all_player_data[(all_player_data["$/mp"]>=6146) & (all_player_data["AV"]< 6.69)]
overrated[['Age','Position','AV','$/mp']].sort_values('AV', ascending=True).head(15)
```
<img width="373" alt="overrated" src="https://github.com/atnikola/nba-player-performance/assets/38530617/c0b7384e-75f4-442e-a5b0-99ee2a38756e">



















