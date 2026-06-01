import pandas as pd
import numpy as np
from nanoboost.scripts.utils.utils import unpickle


# load data from previous processing steps
rmse_df = unpickle("rmse_df.pkl")
accuracy_df = unpickle("accuracy_df.pkl")
smoothness_df = unpickle("smoothness_df.pkl")

# Normalise each individual metric
def normalize_dataframe(df, new_min=0, new_max=1):
    # Apply Min-Max normalization
    normalized_df = (df - df.min()) / (df.max() - df.min()) * (new_max - new_min) + new_min
    return normalized_df

rmse_df_norm = normalize_dataframe(rmse_df)
accuracy_df_norm = normalize_dataframe(accuracy_df)
smoothness_df_norm = normalize_dataframe(smoothness_df)

# accuracy loss
accuracy_loss_df_norm = 1 - accuracy_df_norm

# smoothness loss
smoothness_loss_df_norm = 1 - smoothness_df_norm






import torch

# df --> pytorch tensor
rmse_df_norm_torch = torch.tensor(rmse_df_norm.values, dtype=torch.float32)
accuracy_loss_df_norm_torch = torch.tensor(accuracy_loss_df_norm.values[:len(rmse_df_norm)], dtype=torch.float32)
smoothness_norm_torch = torch.tensor(smoothness_loss_df_norm.values, dtype=torch.float32)
target_value_torch = torch.tensor([0], dtype=torch.float32)

# initial params for alpha and beta (gamma = 1 - alpha - beta)
weights = torch.tensor([0.5, 0.5], requires_grad=True)
alpha = 0.01 # Learning rate
iterations = 2000  # Number of iterations

optimizer = torch.optim.SGD([weights], lr=alpha)

# define objective function: function we want to optimise
def objective_function(weights, rmse, loss_values, smoothness):
    # Extract weights
    alpha, beta = weights[0], weights[1]
    gamma = 1 - alpha - beta  # Derive gamma to ensure weights sum to 1

    # Calculate the weighted metric
    metric = alpha * loss_values + beta * rmse + gamma * smoothness
    mse = metric.mean()
    
    # Universal lower bound for all coefficients
    lower_bound = 0.1
    
    # Specific upper bounds
    upper_bound = 0.8

    # Apply penalties for weights out of bounds
    # Penalties for being below the lower bound
    lower_bounds_penalty = torch.clamp(lower_bound - alpha, min=0) + \
                           torch.clamp(lower_bound - beta, min=0) + \
                           torch.clamp(lower_bound - gamma, min=0)

    # Penalties for exceeding the upper bounds
    alpha_penalty = torch.clamp(alpha - upper_bound, min=0)
    beta_penalty = torch.clamp(beta - upper_bound, min=0)
    gamma_penalty = torch.clamp(gamma - upper_bound, min=0)

    # Penalty strength
    penalty_strength = 5
    penalty = penalty_strength * (lower_bounds_penalty + alpha_penalty + beta_penalty + gamma_penalty)

    return mse + penalty



# Optimization loop
for _ in range(iterations):
    optimizer.zero_grad()
    loss = objective_function(weights, rmse_df_norm_torch, accuracy_loss_df_norm_torch, smoothness_norm_torch)
    loss.backward()
    optimizer.step()

# gamma calculted from other weights as must total to 1
gamma = 1 - weights.detach().numpy()[0] - weights.detach().numpy()[1]
print("Optimal weights:", weights.detach().numpy(), gamma)