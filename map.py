import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.randn(1000, 1000) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(df)