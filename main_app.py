# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# from train_model import get_available_models, train_model
# from build_tpm import generate_tpm
# from compute_phi import calculate_phi

# # --- Page Config ---
# st.set_page_config(page_title="IIT Visualizer", layout="centered")

# # --- Main Title ---
# st.markdown("<h1 style='text-align: center; color: #2c3e50;'>IIT Visualizer UI</h1>", unsafe_allow_html=True)
# st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# # --- Mode Selection ---
# st.subheader("Select Mode")
# mode = st.radio("Choose operation mode", ["Train Network", "Load Network Model"], horizontal=True)

# # --- Model Architecture Selection ---
# st.subheader("Model Architecture")
# models = get_available_models()
# model_name = st.selectbox("Select a model architecture", models)

# # --- Epoch Slider ---
# st.subheader("Training Epochs")
# epoch = st.slider("Select the number of epochs", min_value=1, max_value=100, value=10)

# # --- Run Button ---
# if st.button("Run"):
#     with st.spinner("Running, please wait..."):
#         if mode == "Train Network":
#             train_model(model_name, epoch)  # Saves binarized_states.npy

#         # Generate TPM and connectivity matrix
#         tpm_matrix, connectivity_matrix = generate_tpm("binarized_states.npy")

#         # Calculate Φ
#         phi = calculate_phi(tpm_matrix)

#         # --- Display Result ---
#         st.success(f"Current Φ = {phi:.4f}")

#         # --- Φ vs Epoch Plot ---
#         st.markdown("### Φ vs Epoch")
#         epochs = np.arange(1, epoch + 1)
#         phi_values = np.linspace(0.65, phi, epoch)  # Placeholder simulated values

#         fig_phi, ax_phi = plt.subplots()
#         ax_phi.plot(epochs, phi_values, marker='o', color='#2980b9')
#         ax_phi.set_xlabel("Epoch")
#         ax_phi.set_ylabel("Φ (Phi)")
#         ax_phi.set_title("Φ over Training Epochs")
#         ax_phi.grid(True)
#         st.pyplot(fig_phi)

#         # --- TPM Heatmap ---
#         st.markdown("### Transition Probability Matrix (TPM)")
#         fig_tpm, ax_tpm = plt.subplots()
#         cax1 = ax_tpm.matshow(tpm_matrix, cmap='viridis')
#         fig_tpm.colorbar(cax1)
#         ax_tpm.set_title("TPM Heatmap")
#         st.pyplot(fig_tpm)

#         # --- Connectivity Matrix Heatmap ---
#         st.markdown("### Connectivity Matrix")
#         fig_conn, ax_conn = plt.subplots()
#         cax2 = ax_conn.matshow(connectivity_matrix, cmap='cividis')
#         fig_conn.colorbar(cax2)
#         ax_conn.set_title("Connectivity Heatmap")
#         st.pyplot(fig_conn)

#         # --- CSV Download ---
#         csv_data = "Epoch,Phi\n" + "\n".join([f"{i+1},{phi_values[i]:.4f}" for i in range(epoch)])
#         st.download_button(
#             label="Download Φ Data as CSV",
#             data=csv_data,
#             file_name="phi_results.csv",
#             mime="text/csv"
#         )
# main_app.py
import streamlit as st
from train_model import get_available_models, train_model
from build_tpm import generate_tpm
from compute_phi import calculate_phi

st.set_page_config(page_title="IIT Visualizer", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2c3e50;'>IIT Visualizer UI</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# Mode Selection
st.subheader("Select Mode")
mode = st.radio("Choose operation mode", ["Train Network", "Load Network Model"], horizontal=True)

# Model Architecture
st.subheader("Model Architecture")
models = get_available_models()
model_name = st.selectbox("Select a model architecture", models)

# Training Epoch
st.subheader("Training Epochs")
epoch = st.slider("Select the number of epochs", min_value=1, max_value=100, value=10)

# Run Button
if st.button("Run"):
    if mode == "Train Network":
        with st.spinner("Training model..."):
            train_model(model_name, epoch)
            from utils import run_analysis_and_display  # common display logic
            run_analysis_and_display("binarized_states.npy", epoch)
    elif mode == "Load Network Model":
        # Save model name and epoch as session state before redirect
        st.session_state.model_name = model_name
        st.session_state.epoch = epoch
        st.switch_page("pages/load_model_ui.py")
