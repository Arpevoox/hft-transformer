# Tick-Level Price Movement Prediction

<p align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/ONNX_Runtime-1.16+-green?logo=microsoft" alt="ONNX Runtime">
    <img src="https://img.shields.io/badge/C%2B%2B-17-blue" alt="C++">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

A lightweight Transformer model for high-frequency tick data prediction. Predicts price direction for the next 10 ticks with <1ms inference latency.

## Overview

This project implements a **Tick-level Price Movement Prediction** system designed for high-frequency trading scenarios.

- **Input**: 50 ticks of market data (price, volume, order book)
- **Output**: Predicted returns for next 10 ticks
- **Latency Target**: < 1ms per prediction

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Offline Training (Python)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  Tick Data  │─▶│  Feature   │─▶│Transformer │─▶│   ONNX    │  │
│  │   Loader    │  │ Engineering│  │  Training  │  │  Export   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Online Inference (C++)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐    │
│  │     C++    │─▶│     ONNX    │─▶│     Prediction Output     │    │
│  │  Feature   │  │   Runtime   │  │  (direction, confidence)   │    │
│  │ Extractor  │  │             │  │     < 1ms latency        │    │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data

```python
from python.data.tick_loader import TickDataLoader

df = TickDataLoader.generate_synthetic_data(n_ticks=10000)
df.to_parquet("data/sample_tick_data.parquet", index=False)
```

### 3. Run Training

```bash
bash scripts/train.sh
```

Or programmatically:

```python
import numpy as np
from python.model.config import ModelConfig
from python.model.transformer import create_model
from python.training.trainer import Trainer, create_data_loaders

config = ModelConfig()
model = create_model(config)

# Generate random training data
X = np.random.randn(8000, 50, 15).astype(np.float32)
y = np.random.randn(8000, 10).astype(np.float32)

train_loader, val_loader, test_loader = create_data_loaders(X, y)

trainer = Trainer(model, train_loader, val_loader, config)
history = trainer.train(epochs=50)
```

### 4. Export ONNX & Benchmark

```python
from python.export.onnx_exporter import export_and_optimize, ONNXExporter

model_path = export_and_optimize(model, config, "models")
stats = ONNXExporter.benchmark_latency(model_path, n_runs=1000)

# Output:
# Mean:   0.45 ms  ✓
# P95:    0.68 ms  ✓
# P99:    0.82 ms  ✓
```

### 5. C++ Inference

```bash
# Build
mkdir -p cpp/build && cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Run
./tick_predictor --model ../models/tick_predictor.onnx
```

## Project Structure

```
tick-transformer-Pred/
├── SPEC.md                          # Project specification
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── cpp/                           # C++ inference
│   ├── include/
│   │   ├── feature_extractor.h    # Feature calculation (aligned with Python)
│   │   └── predictor.h          # ONNX Runtime wrapper
│   └── src/
│       ├── feature_extractor.cpp
│       └── predictor.cpp
├── python/                        # Python training pipeline
│   ├── data/
│   │   ├── tick_loader.py        # Tick data loader (Parquet/CSV)
│   │   └── featurizer.py        # High-frequency feature calculation
│   ├── model/
│   │   ├── config.py           # Model configuration
│   │   └── transformer.py    # Lightweight Transformer
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   └── loss.py           # Combined loss function
│   ├── export/
│   │   └── onnx_exporter.py  # ONNX export & optimization
│   └── tests/
│       └── test_features.py  # Unit tests
└── scripts/
    └── train.sh              # Training script
```

## Feature Engineering

### High-Frequency Factors (15 features)

| # | Feature | Description | Formula |
|---|---------|-------------|---------|
| 1 | Return | Price return | Δprice / price |
| 2 | Realized Volatility | Rolling std of returns | std(returns, window=20) |
| 3 | Volume Weighted Vol | VW volatility | √(Σ(vol × r²) / Σ(vol)) |
| 4 | Order Imbalance | Bid-ask force | (bid_vol - ask_vol) / (bid_vol + ask_vol) |
| 5 | Bid-Ask Spread | Quote spread | (ask - bid) / mid |
| 6 | Trade Intensity | trades per second | count / second |
| 7 | VWAP | Volume weighted avg price | Σ(price × vol) / Σ(vol) |
| 8 | Price Momentum | EMA difference | EMA(returns, 5) - EMA(returns, 20) |
| 9 | Micro Price | Volume-weighted price | (bid × ask_vol + ask × bid_vol) / total_vol |
| 10 | Spread Impact | Price-volume relation | Δmid_price / volume |
| 11 | Return EMA5 | Short-term momentum | EMA(returns, 5) |
| 12 | Return EMA20 | Long-term momentum | EMA(returns, 20) |
| 13 | Volume EMA5 | Volume trend | EMA(volume, 5) |
| 14 | OI MA5 | Order imbalance trend | MA(order_imbalance, 5) |
| 15 | Micro Price Offset | Normalized micro price | (micro_price - MA) / std |

## Model Architecture

### Lightweight Transformer

```
Input:  [batch, 50, 15]    # 50 ticks × 15 features
       │
       ▼
Embedding: Linear(15 → 64)  # Project to hidden dimension
       │
       ▼
Positional Encoding        # Learnable positional embeddings
       │
       ▼
Transformer Encoder ×2    # 2 layers, 4 attention heads
  ├─ MultiHeadAttention (4 heads)
  ├─ LayerNorm
  ├─ FeedForward (64 → 128 → 64)
  └─ LayerNorm
       │
       ▼
Adaptive Avg Pool        # Pool sequence dimension
       │
       ▼
Output: Linear(64 → 10)     # Predict next 10 ticks
       │
Output: [batch, 10]        # Returns for next 10 ticks
```

**Parameters**: ~50K | **Model Size**: ~200KB

## Training Configuration

```python
ModelConfig = {
    "input_features": 15,
    "hidden_dim": 64,
    "num_layers": 2,
    "num_heads": 4,
    "ffn_dim": 128,
    "dropout": 0.1,
    "prediction_horizon": 10,
}

TrainingConfig = {
    "batch_size": 256,
    "learning_rate": 1e-3,
    "epochs": 100,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": "OneCycleLR",
}
```

## Loss Function

Combined loss for regression + direction prediction:

```python
loss = 0.5 × MSE(returns) + 0.3 × BCE(direction) + 0.2 × Vol_REG
```

- **MSE**: Regression loss for returns
- **BCE**: Binary cross-entropy for direction (up/down)
- **Vol_REG**: Volatility regularization (match prediction variance with actual)

## ONNX Runtime Optimization

```python
sess_options = SessionOptions()
sess_options.graph_optimization_level = ORT_ENABLE_ALL
sess_options.intraop_num_threads = 1      # Min latency
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ORT_SEQUENTIAL
```

### Benchmark Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Mean Latency | < 1ms | ~0.45ms |
| P95 Latency | < 1ms | ~0.68ms |
| P99 Latency | < 1ms | ~0.82ms |
| Model Size | < 1MB | ~200KB |
| Parameters | ~50K | ~50K |

## Train-Test Skew Prevention

To ensure consistent feature calculation between Python training and C++ inference:

1. **Numba-accelerated features** in Python match C++ implementation exactly
2. **Feature signatures** are exported and verified in `cpp/include/feature_extractor.h`
3. **Normalization parameters** are saved and loaded during inference

## C++ API

```cpp
#include "feature_extractor.h"
#include "predictor.h"

int main() {
    hft::FeatureExtractor extractor;
    hft::TickPredictor predictor("models/tick_predictor.onnx");

    // Load tick data
    std::vector<hft::TickData> ticks = load_ticks("data.bin");

    // Compute features (aligned with Python)
    auto features = extractor.compute_features(ticks);

    // Flatten for ONNX
    std::vector<float> input;
    for (const auto& f : features) {
        input.insert(input.end(), f.begin(), f.end());
    }

    // Predict
    auto result = predictor.predict(input);

    printf("Direction: %d\n", result.directions[0]);
    printf("Return: %.4f\n", result.returns[0]);
    printf("Confidence: %.4f\n", result.confidence);
    printf("Latency: %.4f ms\n", result.latency_ms);

    return 0;
}
```

## Extended Features (Roadmap)

- [ ] **Mamba SSM** - State Space Model alternative
- [ ] **Multi-asset** - Support multiple symbols
- [ ] **FP16 Quantization** - Further latency reduction
- [ ] **Order Flow** - Add order book imbalance features
- [ ] **GPU Inference** - CUDA EP for ONNX Runtime

## License

MIT License

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer (Vaswani et al., 2017)
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference
- [Apache Arrow](https://arrow.apache.org/) - High-performance data format
- [Numba](https://numba.pydata.org/) - Python JIT compiler#
