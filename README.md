# NBA Player Performance
**Andreas Nikolaidis** 

_September 2023_

- [Introduction](#introduction)
- [Import & Clean Data](#import_clean)
- [Correlations & Descriptive Statistics](#descriptive)
- [Principal Component Analysis (PCA)](#pca)
- [Cross Validation & Regression Analysis](#cv-ra)
- [Conclusion](#conclusion)

## [Introduction](#introduction)
In this project I utilize python to analyze data of NBA players in the league and compare their performance against their monetary value (salary) i.e. Which players are undervalued / overvalued. We take a look at some interesting statistics, correlations and then utilize scikit-learn to see if we can accurately predict players next year salaries based on their previous year's performance. (Note: Most contracts and salaries are negotiated with 2-5 years attached so we are looking to see if that future value holds)

## [Import & Clean Data](#import_clean)
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















