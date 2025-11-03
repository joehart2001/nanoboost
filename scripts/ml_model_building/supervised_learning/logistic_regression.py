from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from nanoboost.scripts.utils.utils import unpickle

# load data
X_train_02_bior33_NRNS_norm = unpickle("X_train_02_bior33_NRNS_norm.pkl")
y_train_02_bior33_NRNS_norm = unpickle("y_train_02_bior33_NRNS_norm.pkl")
best_model_XG_random_02_bior33_NRNS_norm = unpickle("best_model_XG_random_02_bior33_NRNS_norm.pkl")
feature_names = unpickle("feature_names_02_bior33_NRNS_norm.pkl")


# Best XGBoost model
xg_model = best_model_XG_random_02_bior33_NRNS_norm.named_steps['XG']

# Get feature importances and identify the top 2 features
feature_importances = xg_model.feature_importances_
top_features_indices = np.argsort(feature_importances)[-2:]  # Gets the indices of the top 2 features

# Extract the top 2 features and target variable
X_top_features = np.array(X_train_02_bior33_NRNS_norm)[:, top_features_indices]
y = y_train_02_bior33_NRNS_norm

# Scale features
scaler = MinMaxScaler()
X_top_features_scaled = scaler.fit_transform(X_top_features)

# Fit logistic regression with 2D features
X_with_intercept = sm.add_constant(X_top_features_scaled)  # Adds a constant term to the predictors
logit_model_2d = sm.Logit(y, X_with_intercept).fit()

# Visualization setup for 3D
x_min, x_max = X_top_features_scaled[:, 0].min() - 1, X_top_features_scaled[:, 0].max() + 1
y_min, y_max = X_top_features_scaled[:, 1].min() - 1, X_top_features_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_with_intercept = sm.add_constant(grid)

# Predict probabilities over the grid
probs = logit_model_2d.predict(grid_with_intercept).reshape(xx.shape)

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(xx, yy, probs, cmap='coolwarm', edgecolor='none', alpha=0.4)

# Scatter plot of actual points
ax.scatter(X_top_features_scaled[:, 0], X_top_features_scaled[:, 1], y, c=y, marker='o', edgecolor='k', cmap='coolwarm')

ax.set_xlabel(f'{feature_names[top_features_indices[0]]}', fontsize=16)
ax.set_ylabel(f'{feature_names[top_features_indices[1]]}', fontsize=16)
ax.set_zlabel('Probability', fontsize=16)
plt.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()