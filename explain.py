import pandas as pd
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from main import preprocess_data, build_dyan_fheg_pian

# Load data
train_df = pd.read_csv("Data/train.csv")
test_df = pd.read_csv("Data/test.csv")

# Split into features and target
X_test = test_df.drop(columns=["Target"]).values
y_test = test_df["Target"].values

X_test_tensor = tf.convert_to_tensor(X_test[:, :, None], dtype=tf.float32)

# Load model
model = build_dyan_fheg_pian(input_shape=(X_test.shape[1], 1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_test_tensor, y_test, epochs=3, verbose=0)  # Dummy fit for demo

# --- 1. Plotting Predictions vs Actuals ---
predictions = model.predict(X_test_tensor)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', linestyle='--')
plt.plot(predictions.flatten(), label='Predicted', alpha=0.7)
plt.title('Stock Price Forecast: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_vs_actual.png")
plt.show()

# --- 2. SHAP Explanation (KernelExplainer for simplicity) ---
print("Generating SHAP explanation (this may take a while)...")

explainer = shap.Explainer(model, X_test_tensor)
shap_values = explainer(X_test_tensor[:10])

shap.plots.bar(shap_values)
shap.plots.waterfall(shap_values[0])