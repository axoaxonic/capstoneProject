#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, FORCE, ESN


# train_states = reservoir.run(XTrain, reset=True)
# readout = readout.fit(train_states, yTrain, warmup=10)
# test_states = reservoir.run(XTest)
# yPred = readout.run(test_states)

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(XTrain, yTrain, warmup=10)
print(reservoir.is_initialized, readout.is_initialized, readout.fitted)

yPred = esn_model.run(XTest)

