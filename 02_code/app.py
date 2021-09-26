import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta  # to add days or years
import pydeck as pdk
import pdb
from pydeck.types import String
import plotly.express as px

st.sidebar.image("01_data/bugoff.png")
view = st.sidebar.selectbox("Select a view:", ["None", "Map View", "Expert View"])

st.title(
    """
# Bug Off!
Welcome to our failure prediction algorithm that increases the stability of *Aargau Verkehr AG's* train networks.
​
By moving the below slider you can view historical data on the error degradation over time.
"""
)
if view == "Map View":
    #%% IMPORT DATA
    DATE_COLUMN = "DateTime"
    DATA_path = "01_data/rssi_excerpt.csv"

    data = pd.read_csv(DATA_path)
    data_anomalies = data.loc[data.Description == "Zwangsbremsung | Zugdaten"]

    # data_anomalies = data[~data['Description'].isnull()]

    data.drop("EventCode", inplace=True, axis=1)
    data.drop("Description", inplace=True, axis=1)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data_anomalies[DATE_COLUMN] = pd.to_datetime(data_anomalies[DATE_COLUMN])

    data_track = data
    #
    # data = data.iloc[1000:9000:3000,:] # data anomalies

    # data_event = np.where(data[])

    # Visualize data on website
    # st.dataframe(data.head())

    data = data_anomalies

    #%% SLIDERS
    # Range selector days
    cols, _ = st.columns((1, 2))
    format = "MM/DD/YY hh:mm:ss"
    start_date = dt.date(
        year=data[DATE_COLUMN].dt.year.iloc[0],
        month=data[DATE_COLUMN].dt.month.iloc[0],
        day=data[DATE_COLUMN].dt.day.iloc[0],
    )  # -relativedelta(years=3)  #Need the right date and  period
    end_date = dt.date(
        year=data[DATE_COLUMN].dt.year.iloc[len(data) - 1],
        month=data[DATE_COLUMN].dt.month.iloc[len(data) - 1],
        day=data[DATE_COLUMN].dt.day.iloc[len(data) - 1],
    )  # dt.datetime.now().date()-relativedelta(years=3)

    if start_date == end_date:
        end_date_new = dt.date(
            year=data[DATE_COLUMN].dt.year.iloc[len(data) - 1],
            month=data[DATE_COLUMN].dt.month.iloc[len(data) - 1],
            day=data[DATE_COLUMN].dt.day.iloc[len(data) - 1] + 1,
        )
        add_slider = cols.slider("Select a range of values", start_date, end_date_new)
    else:
        add_slider = cols.slider("Select a range of values", start_date, end_date)

    #%%

    data_slider = data[data["DateTime"].dt.date == add_slider]

    data_map = "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"

    layer1 = pdk.Layer("HeatmapLayer", data_map, get_position="[lng, lat]")

    # colors = np.matrix([[0,0,0], [0,255,255], [227,207,87], [238,59,59], [222,184,135], [127,255,0], [255,97,3], [191,62,255], [238,18,137], [0,201,87]])

    data_idx = data_track["section"].unique()

    lay1 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[0])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[0, 0, 0],
        get_line_color=[0, 0, 0],
    )
    lay2 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[1])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[255, 0, 0],
        get_line_color=[255, 0, 0],
    )
    lay3 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[2])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[238, 59, 59],
        get_line_color=[238, 59, 59],
    )
    lay4 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[3])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[0, 255, 255],
        get_line_color=[0, 255, 255],
    )
    lay5 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[4])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[238, 18, 137],
        get_line_color=[238, 18, 137],
    )
    lay6 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[5])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[0, 201, 87],
        get_line_color=[0, 201, 87],
    )
    lay7 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[6])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[127, 255, 0],
        get_line_color=[127, 255, 0],
    )
    lay8 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[7])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[222, 184, 135],
        get_line_color=[222, 184, 135],
    )
    lay9 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[8])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[191, 62, 255],
        get_line_color=[191, 62, 255],
    )
    lay10 = pdk.Layer(
        "ScatterplotLayer",
        data_track.iloc[np.where(data_track["section"] == data_idx[9])],
        pickable=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=1,
        get_fill_color=[227, 207, 87],
        get_line_color=[227, 207, 87],
    )

    #
    # layer2= []
    # for n in data_idx:
    #    color = list(np.random.choice(range(254), size=3))
    ##    print(color)
    #    lay = pdk.Layer("ScatterplotLayer", data_track.iloc[np.where(data_track['section']==data_idx[n])], pickable=False, filled=True, radius_scale=6, radius_min_pixels=1, radius_max_pixels=100, line_width_min_pixels=1, get_position=["Longitude", "Latitude"], get_radius=1)#, get_fill_color=color, get_line_color=color)
    #    layer2.append(lay)
    #
    layer3 = pdk.Layer(
        "ScatterplotLayer",
        data_slider,
        pickable=True,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=50,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius=50,
        get_fill_color=[255, 0, 0],
        get_line_color=[255, 0, 0],
    )

    # Set the viewport location
    midpoint_long = (data["Longitude"].min() + data["Longitude"].max()) / 2
    midpoint_lat = (data["Latitude"].min() + data["Latitude"].max()) / 2
    view_state = pdk.ViewState(
        longitude=midpoint_long,
        latitude=midpoint_lat,
        zoom=10,
        min_zoom=3,
        max_zoom=15,
        pitch=0,
        bearing=0,
    )

    # #Combined all of it and render a viewport
    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        layers=[
            layer1,
            lay1,
            lay2,
            lay3,
            lay4,
            lay5,
            lay6,
            lay7,
            lay8,
            lay9,
            lay10,
            layer3,
        ],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Latitude:</b> {Latitude} <br/> <b>Longitude:</b> {Longitude}",
            "style": {"color": "white"},
        },
    )
    st.pydeck_chart(r)
    # r = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", layers=[layer1,layer2,layer3], initial_view_state=view_state,

    #%% Histogram section in the end
    if st.checkbox("Show raw data"):
        st.subheader("Raw data")
        st.write(data)


if view == "Expert View":

    info = {
        "Section 4": "01_data/section_4_median.csv",
        "Section 5": "01_data/section_5_median.csv",
    }

    selection = st.sidebar.selectbox("Select the relevant section:", info.keys())

    data = pd.read_csv(info[selection]).dropna()
    st.dataframe(data.head())
    data["OutlierFlag"] = (data.A2_RSSI_adj < data.A2_RSSI_first_q) | (
        data.A2_RSSI_adj > data.A2_RSSI_third_q
    )

    outliers = data.loc[data.OutlierFlag == 1]

    st.title("Warning System")
    st.markdown(
        "We discretize the distance that the train travels into 10 sections. In the plot below the RSSI metric is plotted for a certain section on the time axis. We upsampled the data from seconds to daily data. The median is used for aggregation To mitigate the impact of outliers."
    )
    data["Label"] = "RSSI"
    fig = px.line(data, x="DateTime", y="A2_RSSI_adj", color="Label")
    fig = fig.add_scatter(
        x=data.DateTime,
        y=data.A2_RSSI_first_q,
        name="Lower Bound",
        marker=dict(color="LightSalmon"),
        mode="lines",
    )
    fig = fig.add_scatter(
        x=data.DateTime,
        y=data.A2_RSSI_third_q,
        name="Upper Bound",
        marker=dict(color="LightSalmon"),
        mode="lines",
    )
    fig = fig.add_scatter(
        x=outliers.DateTime,
        y=outliers.A2_RSSI_adj,
        mode="markers",
        name="Warnings",
        marker=dict(color="#eb1362"),
    )

    st.plotly_chart(fig)
