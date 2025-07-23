# utils.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from build_tpm import generate_tpm
from compute_phi import calculate_phi

def run_analysis_and_display(state_file_path: str, epoch: int):
    # Generate TPM and connectivity matrix
    tpm_matrix, connectivity_matrix = generate_tpm(state_file_path)

    # Calculate Φ
    phi = calculate_phi(tpm_matrix)

    # --- Display Result ---
    st.success(f"Current Φ = {phi:.4f}")

    # --- Φ vs Epoch Plot (Simulated/Placeholder) ---
    st.markdown("### Φ vs Epoch")
    epochs = np.arange(1, epoch + 1)
    phi_values = np.linspace(0.65, phi, epoch)  # Simulated trend

    fig_phi, ax_phi = plt.subplots()
    ax_phi.plot(epochs, phi_values, marker='o', color='#2980b9')
    ax_phi.set_xlabel("Epoch")
    ax_phi.set_ylabel("Φ (Phi)")
    ax_phi.set_title("Φ over Training Epochs")
    ax_phi.grid(True)
    st.pyplot(fig_phi)

    # --- TPM Heatmap ---
    st.markdown("### Transition Probability Matrix (TPM)")
    fig_tpm, ax_tpm = plt.subplots()
    cax1 = ax_tpm.matshow(tpm_matrix, cmap='viridis')
    fig_tpm.colorbar(cax1)
    ax_tpm.set_title("TPM Heatmap")
    st.pyplot(fig_tpm)

    # --- Connectivity Matrix Heatmap ---
    st.markdown("### Connectivity Matrix")
    fig_conn, ax_conn = plt.subplots()
    cax2 = ax_conn.matshow(connectivity_matrix, cmap='cividis')
    fig_conn.colorbar(cax2)
    ax_conn.set_title("Connectivity Heatmap")
    st.pyplot(fig_conn)

    # --- CSV Download ---
    csv_data = "Epoch,Phi\n" + "\n".join([f"{i+1},{phi_values[i]:.4f}" for i in range(epoch)])
    st.download_button(
        label="Download Φ Data as CSV",
        data=csv_data,
        file_name="phi_results.csv",
        mime="text/csv"
    )
