# Dyan-FHEG-PIAN

**Dynamic Finite Heterogeneous Graph-Pufferfish Integrated Attention Network** for real-time and long-term stock market forecasting using deep learning and graph-based models.

## 🧠 Model Architecture
- **FEINN**: Finite Element-Integrated Neural Network for physical behavior simulation
- **DHGANN**: Dynamic Heterogeneous Graph Attention Neural Network
- **POA**: Pufferfish Optimization Algorithm to fine-tune model parameters
- **MaxViT-inspired** Transformer block for spatiotemporal feature extraction

## 📁 Project Structure
```
Dyan-FHEG-PIAN/
├── Data/
│   ├── msft_sample_data.csv
│   ├── train.csv
│   └── test.csv
├── main.py                  # Full model pipeline
├── explain.py               # Visualization + SHAP explanation
├── requirements.txt         # Python dependencies
├── Dyan-FHEG-PIAN-Demo.ipynb # Sample Jupyter Notebook
├── README.md
└── .gitignore
```

## 🛠️ How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run model pipeline:
```bash
python main.py
```

3. Generate visual explanations:
```bash
python explain.py
```

4. Or interactively try:
```bash
jupyter notebook Dyan-FHEG-PIAN-Demo.ipynb
```

## 🔍 Features
- Real MSFT stock data (2020–2024)
- Transformer-based spatiotemporal learning
- SHAP-based model interpretability
- Modular design for future plug-and-play extensions

## 📊 Outputs
- Actual vs predicted stock plots
- SHAP bar/waterfall plots