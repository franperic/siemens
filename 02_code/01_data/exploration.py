import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rssi = pd.read_csv("01_data/rssi.csv")
rssi.columns
rssi.head()

rssi.Track.value_counts()
rssi.groupby("AreaNumber").Track.unique()

rssi.describe().T

plt.plot(rssi.PositionNoLeap[0:20000], rssi.A2_RSSI[:20000])
plt.show()
