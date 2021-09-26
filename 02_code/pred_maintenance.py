import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

disruptions = pd.read_csv("01_data/disruptions.csv")
rssi = pd.read_csv("01_data/rssi_enh.csv")
rssi = rssi.merge(disruptions, on="DateTime", how="left")
rssi_raw = rssi.copy()

rssi.rename(columns={"Description_y": "Description_disruption"}, inplace=True)
rssi.rename(columns={"Description_x": "Description"}, inplace=True)
rssi["DateTime"] = pd.to_datetime(rssi.DateTime)

time_check = rssi.DateTime.diff()
time_check = time_check > pd.Timedelta("1 Second")
check = rssi.loc[time_check]
data_faults = check.Ride.unique()
rssi = rssi.loc[~rssi.Ride.isin(data_faults)]

rssi["A2_RSSI_adj"] = np.where(rssi.A2_RSSI.isin([0, 3]), np.nan, rssi.A2_RSSI)
rssi["A2_RSSI_adj"] = np.where(rssi.A2_RSSI_adj < 0.9, np.nan, rssi.A2_RSSI_adj)
rssi["A2_RSSI_adj"] = np.where(rssi.A2_RSSI_adj > 2.95, np.nan, rssi.A2_RSSI_adj)
rssi["A2_RSSI_adj"] = rssi.A2_RSSI_adj.fillna(method="ffill")
rssi["A2_RSSI_delta"] = rssi.A2_RSSI_adj.diff()
rssi["Date"] = pd.to_datetime(rssi.Date)


relevant = [
    "Zwangsbremse wurde aktiviert",
    "Keine Linienleitertelegramme empfangen",
    "Stoerung: Zwangsbremse wurde aktiviert",
    "Stoerung: Linienleitertelegramme wurden erwartet, jedoch auf beiden Antennen keine empfangen.",
]

rssi["TestFlag"] = np.where(
    rssi.Description_disruption.isin(relevant) & rssi.CurrentVelocity == 0, 1, 0
)


events = rssi.loc[(rssi.Description_disruption.isin(relevant)) & (rssi.TestFlag == 0)]
events.Ride.value_counts()


trips = [2588, 2590, 2592, 2594]
illust = rssi.loc[rssi.Ride.isin(trips)]
illust = illust.loc[illust.PositionNoLeap > 360000]

fig, ax = plt.subplots(4, 1)
for i in trips:
    viz = illust.loc[illust.Ride == i]
    emergency_break = illust.loc[illust.Description_disruption.isin(relevant)]
    ax[0].plot(viz.PositionNoLeap, viz.A2_RSSI_adj, alpha=0.6)
    ax[0].set_ylabel("RSSI")
    if emergency_break.shape[0] > 0:
        # position = emergency_break.PositionNoLeap.value
        ax[0].vlines(
            emergency_break.PositionNoLeap, ymin=0, ymax=3, color="red", alpha=0.2
        )
    ax[1].plot(viz.PositionNoLeap, viz.A2_RSSI_delta, alpha=0.6)
    ax[1].set_ylabel("RSSI_delta")
    ax[2].plot(viz.PositionNoLeap, viz.A2_ValidTel_delta, alpha=0.6)
    ax[2].set_ylabel("ValidTel")
    ax[3].plot(viz.PositionNoLeap, viz.A2_TotalTel_delta, alpha=0.6)
    ax[3].set_ylabel("TotalTel")
plt.show()

plt.plot(viz.DateTime, viz.PositionNoLeap)
plt.hlines(
    emergency_break.PositionNoLeap,
    xmin=emergency_break.DateTime.min(),
    xmax=emergency_break.DateTime.max(),
    color="red",
)
plt.show()

# PCA Check
rssi["section"] = pd.cut(rssi.PositionNoLeap, bins=10, labels=np.arange(10))
section = rssi.loc[rssi["section"] == 5]

X = section.loc[:, ["A2_RSSI_adj", "A2_ValidTel_delta"]]
pca = PCA(n_components=1)
pca.fit(X)
pca.explained_variance_
signal = pca.transform(X)

section["Signal"] = signal

plt.plot(section.DateTime, section.Signal)
plt.show()

section = section.loc[
    :, ["DateTime", "A2_RSSI_adj", "A2_ValidTel_delta", "A2_TotalTel_delta", "Signal"]
]
section.index = section.DateTime
section = section.resample("1d").median()
section = section.reset_index()

fig, ax = plt.subplots(3, 1)
ax[0].plot(section.DateTime, section.A2_RSSI_adj)
ax[1].plot(section.DateTime, section.A2_ValidTel_delta)
ax[2].plot(section.DateTime, section.A2_TotalTel_delta)
plt.show()

section_cleaned = section.dropna()
section_cleaned["A2_RSSI_first_q"] = (
    section_cleaned.rolling(window=14).quantile(0.1).A2_RSSI_adj
)
section_cleaned["A2_RSSI_third_q"] = (
    section_cleaned.rolling(window=14).quantile(0.9).A2_RSSI_adj
)

fig, ax = plt.subplots(3, 1)
ax[0].plot(section_cleaned.DateTime, section_cleaned.A2_RSSI_adj)
ax[0].plot(
    section_cleaned.DateTime,
    section_cleaned.A2_RSSI_first_q,
    color="red",
    linestyle="--",
)
ax[0].plot(
    section_cleaned.DateTime,
    section_cleaned.A2_RSSI_third_q,
    color="red",
    linestyle="--",
)
ax[1].plot(section_cleaned.DateTime, section_cleaned.A2_ValidTel_delta)
ax[2].plot(section_cleaned.DateTime, section_cleaned.A2_TotalTel_delta)
plt.show()

section_cleaned.to_csv("01_data/section_5_median.csv", index=False)

excerpt = pd.read_csv("01_data/rssi_excerpt.csv")
excerpt.Ride.unique()
excerpt.shape
excerpt = rssi.loc[
    rssi.Ride.isin([145, 146, 147, 148, 149, 150]),
    ["DateTime", "Description", "Latitude", "Longitude", "section"],
]
excerpt.to_csv("01_data/rssi_excerpt.csv", index=False)
