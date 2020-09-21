# EECS 731 Project 2 - Classification
#### Author: Jace Kline
### Project Goal
The goal of this project is to classify a character (player) in a Shakespeare play given the line text, play, and data about the act, scene, and line where the character spoke the line.

### Setup
First, we import required Python 3 packages and our helper functions.


```python
# General imports
import sys
sys.path.append('../src/')
from funcs import *

import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
```

Now we load the raw data file into a Pandas DataFrame object and print it to the screen.


```python
df_orig = pd.read_csv("../data/raw/Shakespeare_data.csv")
df_orig
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataline</th>
      <th>Play</th>
      <th>PlayerLinenumber</th>
      <th>ActSceneLine</th>
      <th>Player</th>
      <th>PlayerLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACT I</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SCENE I. London. The palace.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Henry IV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Enter KING HENRY, LORD JOHN OF LANCASTER, the ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.1</td>
      <td>KING HENRY IV</td>
      <td>So shaken as we are, so wan with care,</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.2</td>
      <td>KING HENRY IV</td>
      <td>Find we a time for frighted peace to pant,</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111391</th>
      <td>111392</td>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.180</td>
      <td>LEONTES</td>
      <td>Lead us from hence, where we may leisurely</td>
    </tr>
    <tr>
      <th>111392</th>
      <td>111393</td>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.181</td>
      <td>LEONTES</td>
      <td>Each one demand an answer to his part</td>
    </tr>
    <tr>
      <th>111393</th>
      <td>111394</td>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.182</td>
      <td>LEONTES</td>
      <td>Perform'd in this wide gap of time since first</td>
    </tr>
    <tr>
      <th>111394</th>
      <td>111395</td>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.183</td>
      <td>LEONTES</td>
      <td>We were dissever'd: hastily lead away.</td>
    </tr>
    <tr>
      <th>111395</th>
      <td>111396</td>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>LEONTES</td>
      <td>Exeunt</td>
    </tr>
  </tbody>
</table>
<p>111396 rows × 6 columns</p>
</div>



## Data Preparation and Cleaning
### Removing Unknown 'Player' Rows
Since we want to train/test against the 'Player' label, we shall remove all records where the Player attribute is NaN.


```python
df = df_orig.dropna(subset=['Player'])
```

Let's also clean up the indeces by resetting the index and removing the old 'index' and 'Dataline' columns.


```python
df.reset_index(inplace=True)
df.drop(columns=['index','Dataline'],axis='columns',inplace=True)
df
```

    /home/jacekline/dev/eecs-731/project2/conda-env/lib/python3.8/site-packages/pandas/core/frame.py:3990: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().drop(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Play</th>
      <th>PlayerLinenumber</th>
      <th>ActSceneLine</th>
      <th>Player</th>
      <th>PlayerLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.1</td>
      <td>KING HENRY IV</td>
      <td>So shaken as we are, so wan with care,</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.2</td>
      <td>KING HENRY IV</td>
      <td>Find we a time for frighted peace to pant,</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.3</td>
      <td>KING HENRY IV</td>
      <td>And breathe short-winded accents of new broils</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.4</td>
      <td>KING HENRY IV</td>
      <td>To be commenced in strands afar remote.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Henry IV</td>
      <td>1.0</td>
      <td>1.1.5</td>
      <td>KING HENRY IV</td>
      <td>No more the thirsty entrance of this soil</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111384</th>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.180</td>
      <td>LEONTES</td>
      <td>Lead us from hence, where we may leisurely</td>
    </tr>
    <tr>
      <th>111385</th>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.181</td>
      <td>LEONTES</td>
      <td>Each one demand an answer to his part</td>
    </tr>
    <tr>
      <th>111386</th>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.182</td>
      <td>LEONTES</td>
      <td>Perform'd in this wide gap of time since first</td>
    </tr>
    <tr>
      <th>111387</th>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>5.3.183</td>
      <td>LEONTES</td>
      <td>We were dissever'd: hastily lead away.</td>
    </tr>
    <tr>
      <th>111388</th>
      <td>A Winters Tale</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>LEONTES</td>
      <td>Exeunt</td>
    </tr>
  </tbody>
</table>
<p>111389 rows × 5 columns</p>
</div>



### Converting All Strings to Lowercase
We shall convert all strings to uppercase to avoid confusion and ambiguity in our text analysis.


```python
str_cols = list(df.dtypes[df.dtypes == 'object'].keys())
str_cols
```




    ['Play', 'ActSceneLine', 'Player', 'PlayerLine']




```python
for colname in str_cols:
    df[colname] = df[colname].apply(lambda x: str(x).lower())
df
```

    <ipython-input-7-4268c64132ae>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[colname] = df[colname].apply(lambda x: str(x).lower())





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Play</th>
      <th>PlayerLinenumber</th>
      <th>ActSceneLine</th>
      <th>Player</th>
      <th>PlayerLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>henry iv</td>
      <td>1.0</td>
      <td>1.1.1</td>
      <td>king henry iv</td>
      <td>so shaken as we are, so wan with care,</td>
    </tr>
    <tr>
      <th>1</th>
      <td>henry iv</td>
      <td>1.0</td>
      <td>1.1.2</td>
      <td>king henry iv</td>
      <td>find we a time for frighted peace to pant,</td>
    </tr>
    <tr>
      <th>2</th>
      <td>henry iv</td>
      <td>1.0</td>
      <td>1.1.3</td>
      <td>king henry iv</td>
      <td>and breathe short-winded accents of new broils</td>
    </tr>
    <tr>
      <th>3</th>
      <td>henry iv</td>
      <td>1.0</td>
      <td>1.1.4</td>
      <td>king henry iv</td>
      <td>to be commenced in strands afar remote.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>henry iv</td>
      <td>1.0</td>
      <td>1.1.5</td>
      <td>king henry iv</td>
      <td>no more the thirsty entrance of this soil</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111384</th>
      <td>a winters tale</td>
      <td>38.0</td>
      <td>5.3.180</td>
      <td>leontes</td>
      <td>lead us from hence, where we may leisurely</td>
    </tr>
    <tr>
      <th>111385</th>
      <td>a winters tale</td>
      <td>38.0</td>
      <td>5.3.181</td>
      <td>leontes</td>
      <td>each one demand an answer to his part</td>
    </tr>
    <tr>
      <th>111386</th>
      <td>a winters tale</td>
      <td>38.0</td>
      <td>5.3.182</td>
      <td>leontes</td>
      <td>perform'd in this wide gap of time since first</td>
    </tr>
    <tr>
      <th>111387</th>
      <td>a winters tale</td>
      <td>38.0</td>
      <td>5.3.183</td>
      <td>leontes</td>
      <td>we were dissever'd: hastily lead away.</td>
    </tr>
    <tr>
      <th>111388</th>
      <td>a winters tale</td>
      <td>38.0</td>
      <td>nan</td>
      <td>leontes</td>
      <td>exeunt</td>
    </tr>
  </tbody>
</table>
<p>111389 rows × 5 columns</p>
</div>



### Converting 'PlayerLinenumber' to an Integer value to conserve memory
Since the 'PlayerLinenumber' column is listing whole number values only, it is a good idea to replace this column's values with the integer equivalents.


```python
df['PlayerLinenumber'] = df['PlayerLinenumber'].apply(lambda x: int(x))#.astype(np.short)
df['PlayerLinenumber'] = df['PlayerLinenumber'].astype(np.short)
df
```

    <ipython-input-8-112af6ef6a87>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['PlayerLinenumber'] = df['PlayerLinenumber'].apply(lambda x: int(x))#.astype(np.short)
    <ipython-input-8-112af6ef6a87>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['PlayerLinenumber'] = df['PlayerLinenumber'].astype(np.short)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Play</th>
      <th>PlayerLinenumber</th>
      <th>ActSceneLine</th>
      <th>Player</th>
      <th>PlayerLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1.1.1</td>
      <td>king henry iv</td>
      <td>so shaken as we are, so wan with care,</td>
    </tr>
    <tr>
      <th>1</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1.1.2</td>
      <td>king henry iv</td>
      <td>find we a time for frighted peace to pant,</td>
    </tr>
    <tr>
      <th>2</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1.1.3</td>
      <td>king henry iv</td>
      <td>and breathe short-winded accents of new broils</td>
    </tr>
    <tr>
      <th>3</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1.1.4</td>
      <td>king henry iv</td>
      <td>to be commenced in strands afar remote.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1.1.5</td>
      <td>king henry iv</td>
      <td>no more the thirsty entrance of this soil</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111384</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5.3.180</td>
      <td>leontes</td>
      <td>lead us from hence, where we may leisurely</td>
    </tr>
    <tr>
      <th>111385</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5.3.181</td>
      <td>leontes</td>
      <td>each one demand an answer to his part</td>
    </tr>
    <tr>
      <th>111386</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5.3.182</td>
      <td>leontes</td>
      <td>perform'd in this wide gap of time since first</td>
    </tr>
    <tr>
      <th>111387</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5.3.183</td>
      <td>leontes</td>
      <td>we were dissever'd: hastily lead away.</td>
    </tr>
    <tr>
      <th>111388</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>nan</td>
      <td>leontes</td>
      <td>exeunt</td>
    </tr>
  </tbody>
</table>
<p>111389 rows × 5 columns</p>
</div>



## Understanding the Data
### Querying Available Plays and Players
First, we shall get the play names in our dataset.


```python
play_series = df['Play'].drop_duplicates()
play_names = list(play_series)
print("Number of plays: ", play_series.count())
```

    Number of plays:  36


Now, we run a query to get the available players.


```python
all_players = df['Player'].drop_duplicates()
players = list(all_players)
print("Number of players: ", all_players.count())
```

    Number of players:  922


## Feature Engineering
To make our dataset more useful for training our model, we must perform transformations to our data. For each current feature in our dataset, we must look for ways to extract more meaningful information.

### Coversion of 'ActSceneLine' values to numerical values
Currently, the 'ActSceneLine' column is an object type, and therefore will serve little purpose in our numerically-inclined model. Hence, we must map this feature to an equivalent numerical form. We shall achieve this by creating three new features: 'Act', 'Scene', and 'Line', where the data type for each is an integer value. See the example below.


```python
def actSceneLineConvert(asl): # string -> 3-tuple of integers
    regex = '([1-9]+)[.]([1-9]+)[.]([1-9]+)'
    m = re.search(regex, str(asl))
    if asl == asl and asl is not None and m.group(1):
        return (int(m.group(1)),int(m.group(2)),int(m.group(3)))
    else:
        return (0,0,0)

# Example
actSceneLineConvert('3.4.27')
```




    (3, 4, 27)



Now, let us use this function that we defined to create our new features.


```python
r,c = df.shape
newarr = np.zeros((r,3),dtype=np.short)

for i in range(0,r):
    try:
        newarr[i,0:3] = actSceneLineConvert(df.loc[i,'ActSceneLine'])
    except:
        pass
```


```python
df_add = pd.DataFrame(newarr,columns=['Act','Scene','Line'])
df = pd.concat([df,df_add],axis='columns').drop('ActSceneLine', axis='columns')
cols = df.columns.tolist()
cols_ = cols[0:2] + cols[-3:] + cols[3:4] + cols[2:3]
df = df[cols_]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Play</th>
      <th>PlayerLinenumber</th>
      <th>Act</th>
      <th>Scene</th>
      <th>Line</th>
      <th>PlayerLine</th>
      <th>Player</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>so shaken as we are, so wan with care,</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>find we a time for frighted peace to pant,</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>and breathe short-winded accents of new broils</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>to be commenced in strands afar remote.</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>4</th>
      <td>henry iv</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>no more the thirsty entrance of this soil</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111384</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>18</td>
      <td>lead us from hence, where we may leisurely</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111385</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>181</td>
      <td>each one demand an answer to his part</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111386</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>182</td>
      <td>perform'd in this wide gap of time since first</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111387</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>183</td>
      <td>we were dissever'd: hastily lead away.</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111388</th>
      <td>a winters tale</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>exeunt</td>
      <td>leontes</td>
    </tr>
  </tbody>
</table>
<p>111389 rows × 7 columns</p>
</div>



### Conversion of 'Play' to Numerical values
Like above, we want to maximize the amount of numerical data available to our model and hence we shall convert each play to a unique integer value for in our dataset. We will store this new data in a new column called 'PlayNum'.


```python
# Mapping of play name -> integer
zipper = zip(play_names,range(0,len(play_names)))
map_dict = dict(zipper)

# Apply the transformation over the 'Play' column
df['PlayNum'] = df['Play'].apply(lambda s: map_dict[str(s)])
df['PlayNum'] = df['PlayNum'].astype(np.short)

# Reorder the columns
cols = df.columns.tolist()
cols_ = ['Play','PlayNum'] + cols[1:-1]
df = df[cols_]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Play</th>
      <th>PlayNum</th>
      <th>PlayerLinenumber</th>
      <th>Act</th>
      <th>Scene</th>
      <th>Line</th>
      <th>PlayerLine</th>
      <th>Player</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>henry iv</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>so shaken as we are, so wan with care,</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>henry iv</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>find we a time for frighted peace to pant,</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>henry iv</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>and breathe short-winded accents of new broils</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>henry iv</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>to be commenced in strands afar remote.</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>4</th>
      <td>henry iv</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>no more the thirsty entrance of this soil</td>
      <td>king henry iv</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>111384</th>
      <td>a winters tale</td>
      <td>35</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>18</td>
      <td>lead us from hence, where we may leisurely</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111385</th>
      <td>a winters tale</td>
      <td>35</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>181</td>
      <td>each one demand an answer to his part</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111386</th>
      <td>a winters tale</td>
      <td>35</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>182</td>
      <td>perform'd in this wide gap of time since first</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111387</th>
      <td>a winters tale</td>
      <td>35</td>
      <td>38</td>
      <td>5</td>
      <td>3</td>
      <td>183</td>
      <td>we were dissever'd: hastily lead away.</td>
      <td>leontes</td>
    </tr>
    <tr>
      <th>111388</th>
      <td>a winters tale</td>
      <td>35</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>exeunt</td>
      <td>leontes</td>
    </tr>
  </tbody>
</table>
<p>111389 rows × 8 columns</p>
</div>



#### Let's save our dataset
We have made a lot of changes to the 'df' dataset and should therefore save the changes into a new file for redundancy and backup.


```python
df.to_csv('../data/transformed/shakespeare.csv')
```

## Model Ideas
Now that we have cleaned and transformed our dataset, we can consider ideas regarding our approach to choosing and training a predictive model.

### Key Considerations
Given that this dataset is over 100,000 rows, we must be cognizant of our memory usage and our model complexity. Given that our machine is not specifically built with data science in mind, we must operate within the memory and time constraints of our machine.

### Text-Based Approach
#### Overview
One property of our dataset as it stands is that the text field 'PlayerLine' holds the highest quantity of raw information in each row. By extracting textual patterns, word usage, and word frequency for each player, it shall be possible to predict the player based on these textual cues. The common approach in this type of endeavor is to employ a vectorizer that will do the following:
1. Tokenize all text to create new features based on words, bigrams, trigrams, etc.
2. Vectorize each row's text field into a vector representing the count or frequency of each token
After these steps, we shall use the vectorized data to train a model (e.g. Naive Bayes). Then, we shall evaluate the model with test data to determine the model's predictive success rate.

#### Splitting Dataset into Multiple Datasets (by 'Play')
A property of the current dataset is that it is quite large (over 100,000 rows). In the pursuit of a player classification technique over the text field, we shall split our dataset by play name into distinct datasets and form distinct models for each play. Since the 'Play' attribute shall be given to us in the final evaluation, we may use this information to conditionally choose which model to run on a given record to output the estimated 'Player'. By reducing our dataset into pieces and forming separate models, we increase the likelihood of choosing the correct player. Essentially, this approach is expressing that, first and foremost, the 'Play' attribute must be matched in the model. In addition to the predictive potential of this new approach, the smaller datasets allow for more intensive text analysis on each play due to more free memory on the system.

#### Testing this Approach
We shall employ the text-based approach on a subset of the data to determine its practicality and effectiveness potential. In particular, we shall test this on the data for "Henry IV".

##### Process
We will perform the following high-level steps to accomplish this vectorization process:
1. Define a vectorizer to be a ScikitLearn TF-IDF tokenizer class instance (Term frequency-inverse document frequency)
2. Tokenize our input text
    * We will specify the tokenizer to include single words as tokens
    * This will help to capture the context of the characters' language in a vectorized numerical form
3. Generate a new set of features to match the tokens found in all the text (from each row)
4. Run the vectorizer capability on each row to fill the corresponding normalized token frequencies


```python
# Import required SciKit Learn constructs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
```


```python
# Get only the 'Henry IV' play data
henryiv = df[df['Play'] == 'henry iv']

# instantiate our vectorizer
vectorizer = TfidfVectorizer()

# instantiate our model
model = MultinomialNB()

# Assign independent, dependent variables for model
X_henryiv_text = henryiv['PlayerLine']
y_henryiv = henryiv['Player']

def printPerformance(dec):
    print("Model performance: {}%".format(dec*100))

def tfidfModelEval(X_text,y,vectorizer,model):
    # Map the text column to a vectorized array
    X = vectorizer.fit_transform(X_text).toarray()

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

    # Train the model on the training data
    clf = model.fit(X_train,y_train)

    # Evaluate and return the performance of the trained model on the testing data
    return clf.score(X_test,y_test)

printPerformance(tfidfModelEval(X_henryiv_text,y_henryiv,vectorizer,model))
```

    Model performance: 27.250000000000004%


#### Results
As we can see from above, using a text-based vectorization approach only resulted in roughly 27% accuracy on the testing data. This low accuracy measure is an indicator that this paradigm is inherently flawed and the patterns that may be extracted from the text are quite sparse barring the use of more advanced methods. In addition to the low accuracy, it is simply not feasible to execute this type of classification model on the entire dataset due to memory constraints of our target machine. This leads us to pursue other options.

### Numbers-Exclusive Approach
#### Overview
Contary to the previous method, we shall use all of the numerical columns in our transformed dataset as features. These columns include 'PlayNum', 'PlayerLinenumber', 'Act', 'Scene', and 'Line'. My hypothesis is that, due to the conditional structure of the numerical data as it relates to the player output, we may use a Decision Tree model with success. In addition, since the memory overhead of this model is much lower, we may process the entire dataset in a single model.


```python
# Import the Tree-based models
from sklearn.tree import DecisionTreeClassifier
```


```python
feature_names = ['PlayNum','PlayerLinenumber','Act','Scene','Line']
X = df[feature_names].to_numpy()
y = df['Player']

def standardModelEval(X,y,model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = model.fit(X_train, y_train)
    return clf.score(X_test,y_test)
```


```python
model = DecisionTreeClassifier()
printPerformance(standardModelEval(X,y,model))
```

    Model performance: 74.78813559322035%


#### Results
As we can see above, we achieved roughly 75% accuracy on our tree-based model. This model was very fast and required very little memory compared to the text vectorization model. These properties enabled us to train and test our entire dataset at one time. Although this percentage isn't perfect, the high speed and low memory allow us to work with this model over our entire dataset.

### Conclusion
The text-based analysis and modeling proved to be difficult in this project due to the sheer amount of data and the lack of textual patterns available to classify characters. On the contrary, utilization of the numerical features of our dataset and the deployment of a decision tree model proved to be not only much more accurate, but also fast and memory efficient in the context of our entire dataset. Although the text model proved ineffective in our limited trial, the idea may still be put to good use in conjuction to a tree-based model. Due to time constraints, we were not able to explore the concept of model combination, but this concept may prove useful in the context of this problem.
