import numpy as np  
import pandas as pd

f = 2       # signal frequency [Hz] 

fs = 100    # sample frequency [Hz]

t0, tf = 0,                    # initial and final time [s]  

t = np.linspace(start = 0, stop = tf, num = tf*fs, endpoint=True) # time array

u = np.sin(2*np.pi*f*t);                     

signal = pd.DataFrame(data=u , index=t, columns = ['u1'])

import seaborn as sns

sns.scatterplot(data=signal)