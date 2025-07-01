import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau

# Load the dataset
def load_stock_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df

# Entropy-based feature weighting
def calculate_entropy_weights(df):
    norm_df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            col_sum = df[col].sum()
            if col_sum == 0:
                norm_df[col] = 0
            else:
                norm_df[col] = df[col] / col_sum
    entropy = -np.nansum(norm_df * np.log(norm_df + 1e-9), axis=0)
    weights = (1 - entropy / np.log(len(df)))  # Normalized weights
    return weights

# τ-Kendall correlation filtering
def kendall_filter(df, threshold=0.1):
    corr_matrix = df.corr(method='kendall')
    relevant_features = [col for col in corr_matrix.columns if any(abs(corr_matrix[col]) > threshold)]
    return df[relevant_features]

# Full preprocessing pipeline
def preprocess_data(filepath):
    df = load_stock_data(filepath)
    df_numeric = df.select_dtypes(include=[np.number])

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)

    # Entropy weighting
    weights = calculate_entropy_weights(scaled_df)
    weighted_df = scaled_df * weights

    # τ-Kendall correlation filter
    filtered_df = kendall_filter(weighted_df)

    return filtered_df, scaler

# Example usage
if __name__ == "__main__":
    filepath = "Data/msft_sample_data.csv"
    processed_df, fitted_scaler = preprocess_data(filepath)
    print("Preprocessed Data Shape:", processed_df.shape)
    print(processed_df.head())

import tensorflow as tf
from tensorflow.keras import layers, models

# Positional encoding layer (simplified Time2Vec substitute)
class TimeEmbedding(layers.Layer):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.embed_dim), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.embed_dim,), initializer="zeros", trainable=True)

    def call(self, x):
        return tf.math.sin(tf.tensordot(x, self.w, axes=1) + self.b)

# Transformer block (simplified MaxViT-style)
def transformer_block(embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = layers.Input(shape=(None, embed_dim))
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention = layers.Dropout(rate)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    ffn = layers.Dense(ff_dim, activation="relu")(attention)
    ffn = layers.Dense(embed_dim)(ffn)
    ffn = layers.Dropout(rate)(ffn)

    outputs = layers.LayerNormalization(epsilon=1e-6)(attention + ffn)
    return models.Model(inputs=inputs, outputs=outputs, name="transformer_block")

# Build MaxViT-style feature extractor
def build_feature_extractor(input_shape, embed_dim=32, num_heads=4, ff_dim=64):
    inputs = layers.Input(shape=input_shape)
    x = TimeEmbedding(embed_dim)(inputs)
    transformer = transformer_block(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(transformer)
    model = models.Model(inputs=inputs, outputs=x, name="feature_extractor")
    return model

# Example usage
if __name__ == "__main__":
    filepath = "Data/msft_sample_data.csv"
    processed_df, fitted_scaler = preprocess_data(filepath)
    data_tensor = tf.convert_to_tensor(processed_df.values[:, :, tf.newaxis], dtype=tf.float32)

    feature_extractor = build_feature_extractor(input_shape=(processed_df.shape[1], 1))
    features = feature_extractor(data_tensor)
    print("Extracted Feature Shape:", features.shape)

# --- FEINN MODULE (Simplified Physics-Informed Block) ---

class FEINN(tf.keras.layers.Layer):
    def __init__(self, units=64):
        super(FEINN, self).__init__()
        self.dense1 = layers.Dense(units, activation='tanh')
        self.dense2 = layers.Dense(units, activation='relu')

    def call(self, x):
        # Simulate market 'stress' and 'strain' using learnable transformations
        displacement = self.dense1(x)
        stress = self.dense2(displacement)
        return stress


# --- DHGANN MODULE (Simplified GAT Attention) ---

class DHGANN(tf.keras.layers.Layer):
    def __init__(self, units=64, heads=4):
        super(DHGANN, self).__init__()
        self.attn = layers.MultiHeadAttention(num_heads=heads, key_dim=units)
        self.norm = layers.LayerNormalization()
        self.dense = layers.Dense(units, activation='relu')

    def call(self, x):
        context = self.attn(x, x)
        x = self.norm(x + context)
        x = self.dense(x)
        return x


# --- PUFFERFISH OPTIMIZATION (Simplified Metaheuristic Tuning) ---

def pufferfish_optimizer(model, loss_fn, x_train, y_train, iterations=10):
    best_weights = model.get_weights()
    best_loss = float("inf")

    for i in range(iterations):
        noisy_weights = [w + tf.random.normal(tf.shape(w), stddev=0.01) for w in best_weights]
        model.set_weights(noisy_weights)

        with tf.GradientTape() as tape:
            predictions = model(x_train, training=True)
            loss = loss_fn(y_train, predictions)

        if loss < best_loss:
            best_loss = loss
            best_weights = noisy_weights

    model.set_weights(best_weights)
    return model


# --- Dyan-FHEG-PIAN Full Architecture ---

def build_dyan_fheg_pian(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = TimeEmbedding(embed_dim=32)(inputs)
    x = transformer_block(embed_dim=32, num_heads=4, ff_dim=64)(x)

    feinn = FEINN(units=64)(x)
    dhgann = DHGANN(units=64)(x)

    concat = layers.Concatenate()([feinn, dhgann])
    output = layers.Dense(1)(concat)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="Dyan-FHEG-PIAN")
    return model


# Example usage (continued)
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    filepath = "Data/msft_sample_data.csv"
    processed_df, fitted_scaler = preprocess_data(filepath)

    X = tf.convert_to_tensor(processed_df.values[:, :, tf.newaxis], dtype=tf.float32)
    y = tf.reduce_mean(X, axis=1)  # placeholder target variable for regression

    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)

    model = build_dyan_fheg_pian(input_shape=(X.shape[1], 1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Apply pufferfish optimization (10 iterations for demonstration)
    model = pufferfish_optimizer(model, tf.keras.losses.MeanSquaredError(),
                                  tf.convert_to_tensor(X_train, dtype=tf.float32),
                                  tf.convert_to_tensor(y_train, dtype=tf.float32))

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test)
    print("Final MSE Loss:", loss)
    print("Final MAE:", mae)