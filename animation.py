import streamlit as st
import time
import numpy as np

st.title("Neuron Firing Simulation")

progress = st.empty()

for _ in range(100):
    val = np.random.rand()
    progress.progress(val)
    time.sleep(0.05)
