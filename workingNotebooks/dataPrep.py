
import time
import pandas as pd
import numpy as np

cols = ['id', 'event', 'device', 'channel', 'code', 'size', 'data']
emotiv = pd.read_csv('data/EP1.txt', delimiter='\t', names=cols)
emotiv.drop(['device', 'id', 'size'], inplace=True, axis=1)

occ0 = emotiv[emotiv['channel' == 'O1']]
occ1 = emotiv[emotiv['channel' == 'O2']]
fefF3 = emotiv[emotiv['channel' == 'F3']]
fefF4 = emotiv[emotiv['channel' == 'F4']]
fefF7 = emotiv[emotiv['channel' == 'F7']]
fefF8 = emotiv[emotiv['channel' == 'F8']]

del emotiv

occ0['data'] = occ0['data'].apply(lambda x: x.split(','))
occ0Exp = occ0.explode('data')

occ1['data'] = occ1['data'].apply(lambda x: x.split(','))
occ1Exp = occ1.explode('data')

#fefF3['data'] = fefF3['data'].apply(lambda x: x.split(','))
#fefF3Exp = fefF3.explode('data')

#fefF4['data'] = fefF4['data'].apply(lambda x: x.split(','))
#fefF4Exp = fefF4.explode('data')

fefF7['data'] = fefF7['data'].apply(lambda x: x.split(','))
fefF7Exp = fefF7.explode('data')

fefF8['data'] = fefF8['data'].apply(lambda x: x.split(','))
fefF8Exp = fefF8.explode('data')

baseline = pd.concat([occ0, occ1, fefF7, fefF8])
print(baseline.head())
#baseline.to_csv('baseline.csv', sep=',')
