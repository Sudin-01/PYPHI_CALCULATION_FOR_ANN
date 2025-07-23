# load_model_ui.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Loaded IIT Results", layout="centered")
st.markdown("<h1 style='text-align: center; color: #34495e;'>Loaded Model Results</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# --- Fetch Session Info ---
model_name = st.session_state.get("model_name", "UnknownModel")
epoch = st.session_state.get("epoch", 10)

# --- File Path Assumptions ---
base_path = f"saved_data/{model_name}"
phi_file = os.path.join(base_path, "phi_values.npy")
tpm_file = os.path.join(base_path, "tpm.npy")
conn_file = os.path.join(base_path, "connectivity.npy")

# --- Load Data ---
try:
    phi_values = np.load(phi_file)
    tpm_matrix = np.load(tpm_file)
    connectivity_matrix = np.load(conn_file)
    final_phi = phi_values[-1]

    st.success(f"Loaded Φ = {final_phi:.4f}")

    # --- Plot Φ vs Epoch ---
    st.markdown("### Φ vs Epoch")
    epochs = np.arange(1, len(phi_values) + 1)
    fig_phi, ax_phi = plt.subplots()
    ax_phi.plot(epochs, phi_values, marker='o', color='#16a085')
    ax_phi.set_xlabel("Epoch")
    ax_phi.set_ylabel("Φ (Phi)")
    ax_phi.set_title("Φ over Training Epochs")
    ax_phi.grid(True)
    st.pyplot(fig_phi)

    # --- TPM Matrix ---
    st.markdown("### Transition Probability Matrix (TPM)")
    fig_tpm, ax_tpm = plt.subplots()
    cax1 = ax_tpm.matshow(tpm_matrix, cmap='viridis')
    fig_tpm.colorbar(cax1)
    ax_tpm.set_title("TPM Heatmap")
    st.pyplot(fig_tpm)

    # --- Connectivity Matrix ---
    st.markdown("### Connectivity Matrix")
    fig_conn, ax_conn = plt.subplots()
    cax2 = ax_conn.matshow(connectivity_matrix, cmap='cividis')
    fig_conn.colorbar(cax2)
    ax_conn.set_title("Connectivity Heatmap")
    st.pyplot(fig_conn)

    # --- CSV Download ---
    csv_data = "Epoch,Phi\n" + "\n".join([f"{i+1},{phi_values[i]:.4f}" for i in range(len(phi_values))])
    st.download_button(
        label="Download Φ Data as CSV",
        data=csv_data,
        file_name=f"{model_name}_phi_results.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Error loading data for model '{model_name}': {e}")
