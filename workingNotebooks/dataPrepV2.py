import time
import pandas as pd
import numpy as np
import gc

# Load in original raw data and give it column names
cols = ['id', 'event', 'device', 'channel', 'code', 'size', 'data']
emotiv = pd.read_csv('../../fulldata/EP1.txt', delimiter='\t', names=cols)
emotiv.drop(['device', 'id'], inplace=True, axis=1)

# Break up full df into sub-dfs by channel
occ0 = emotiv[(emotiv['channel'] == 'O1')]
occ1 = emotiv[(emotiv['channel'] == 'O2')]
fefF3 = emotiv[(emotiv['channel'] == 'F3')]
fefF4 = emotiv[(emotiv['channel'] == 'F4')]
fefF7 = emotiv[(emotiv['channel'] == 'F7')]
fefF8 = emotiv[(emotiv['channel'] == 'F8')]
temT7 = emotiv[(emotiv['channel'] == 'T7')]
temT8 = emotiv[(emotiv['channel'] == 'T8')]
pfcAF3 = emotiv[(emotiv['channel'] == 'AF3')]
pfcAF4 = emotiv[(emotiv['channel'] == 'AF4')]
motFC5 = emotiv[(emotiv['channel'] == 'FC5')]
motFC6 = emotiv[(emotiv['channel'] == 'FC6')]
parP7 = emotiv[(emotiv['channel'] == 'P7')]
parP8 = emotiv[(emotiv['channel'] == 'P8')]

# Delete and garbage collect the full df so computer doesn't run out of RAM and freeze
del emotiv
gc.collect()

def dataProcessor(df):
    '''
Cleans data column by splitting it into smaller strings, converting those to float, cutting it down to length defined by shortest data vector, normalizing the indexes by resetting.

i: Dataframe for single channel
o: Processed dataframe, printouts of lengths before and after clipping for check, timestamp for each iteration
    '''
    col = df['data'].apply(lambda x: list(map(float, x.split(','))))
    print(type(col), type(col.iloc[0]), type(col.iloc[0][0]))

    for i in range(len(col)):
        l = []
        l.append(len(col.iloc[i]))

    print(min(l))

    for i in range(len(col)):
        col.iloc[i] = col.iloc[i][:257] # or 257?

    for i in range(len(col)):
        l = []
        l.append(len(col.iloc[i]))

    print(max(l))
    return col.reset_index(drop=True)

# Choose  which channels to include
dfs = [occ0, occ1, fefF3,
       fefF4, fefF7, fefF8, temT7,temT8,
        pfcAF3, pfcAF4, motFC5, motFC6,
        parP7, parP8]

# Init blank dataframe for processed channels to be added to
dfTensor = pd.DataFrame()

#  select columns by name by grabbing channel name value string from the 'channel' column
# then running dataProcessor on all the 
for x in dfs:
    name = x['channel'].iloc[0]
    dfTensor[name] = dataProcessor(x) 
    print(time.time())

# Add code column from any channel df
dfTensor['code'] = occ1['code'].reset_index(drop=True)
print(dfTensor.head())
# Delete original dfs with this ugly stack of dels, garbage collect to conserve RAM
del occ0
del occ1
del fefF3
del fefF4
del fefF7
del fefF8
del temT7
del temT8
del pfcAF3
del pfcAF4
del motFC5
del motFC6
del parP7
del parP8
gc.collect()

# Save resulting dataframe to csv
dfTensor.to_csv('../data/dfTensor.csv', sep=',')
