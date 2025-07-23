import numpy as np

def connectivity_matrix(model_name="MLP",tpm_matrix='tpm.npy'):
    if model_name != "MLP":
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")
     
    num_nodes = int(np.log2(tpm_matrix.shape[0]))
    num_neurons = num_nodes  # Assumes one node per neuron

    # Define a connectivity matrix (fully connected by default)
    connectivity = np.zeros((num_neurons, num_neurons), dtype=int)
    return connectivity
