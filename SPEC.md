# 量化机器学习：基于 Transformer 的短期价格走势预测

## 1. Project Overview

**项目名称**: Tick-Level Price Movement Prediction  
**项目类型**: 高频时序预测系统  
**核心价值**: 证明将前沿 AI 模型应用于高频时序数据的能力

### 目标
- 针对 Tick 级别数据（毫秒级），构建轻量级 Transformer 预测未来 10 个 Tick 的价格走势
- 单次推理延迟压低到 1ms 以内（ONNX Runtime 优化）
- 确保线下训练与线上 C++ 推理特征计算完全一致

---

## 2. Technical Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Online Inference (C++)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ C++ Feature │─▶│  ONNX       │─▶│  Prediction Output          │  │
│  │ Extractor   │  │  Runtime    │  │  (10 Tick Direction/Price)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ ONNX Export
                                  │
┌───────────────────────────────────────────────────────────────────┐
│                       Offline Training (Python)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │ Tick Data   │─▶│  Feature    │─▶│ Transformer│─▶│  ONNX     │  │
│  │ Loader     │  │  Engineering│  │  Training  │  │  Export   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| Layer | Technology |
|-------|-------------|
| Data Storage | Apache Arrow (Parquet) |
| Training | PyTorch 2.0+ |
| Model | Transformer (Custom Lightweight) |
| Optimization | ONNX Runtime, Quantization |
| Inference | C++ with ONNX Runtime |
| Feature Store | NumPy/Numba |

---

## 3. Feature Engineering

### 3.1 原始 Tick 数据字段

```python
TickData = {
    "timestamp": int,      # 纳秒时间戳
    "price": float,       # 最新价
    "volume": int,       # 成交量
    "bid_price": float,   # 买一价
    "ask_price": float,   # 卖一价
    "bid_volume": int,   # 买一量
    "ask_volume": int,    # 卖一量
    "direction": int,    # 方向: 1=up, -1=down, 0=no_change
}
```

### 3.2 高频因子计算

| 因子名称 | 计算逻辑 | 用途 |
|---------|----------|------|
| **Return** | `Δprice / price` | 价格变动率 |
| **Realized Volatility** | `rolling_std(returns, window=20)` | 时变波动率 |
| **Volume Weighted Vol** | `sqrt(sum(vol * price^2) / sum(vol))` | 成交量加权波动率 |
| **Order Imbalance** | `(bid_vol - ask_vol) / (bid_vol + ask_vol)` | 买卖力道 |
| **Bid-Ask Spread** | `(ask_price - bid_price) / mid_price` | 价差因子 |
| **Trade Intensity** | `count per second` | 交易强度 |
| **VWAP** | `sum(price * vol) / sum(vol)` | 成交量加权均价 |
| **Price Momentum** | `EMA(returns, span=5) - EMA(returns, span=20)` | 动量因子 |
| **MicroPrice** | `(bid_price * ask_vol + ask_price * bid_vol) / (bid_vol + ask_vol)` | 微观价格 |
| **Spread Impact** | `Δmid_price / volume` | 价量关系 |

### 3.3 特征窗口

- **Input Window**: 50 ticks (回看50个tick)
- **Prediction Horizon**: 10 ticks (预测未来10个tick)
- **Feature Normalization**: Rolling z-score (window=100)

---

## 4. Model Architecture

### 4.1 模型设计: Lightweight Transformer

```python
class LightweightTransformer(nn.Module):
    """
    轻量级 Transformer for Tick Prediction
    
    Design:
    - 2-layer Encoder (可切换为 Mamba)
    - 4-head Attention
    - Hidden: 64, FFN: 128
    - ~50K parameters (轻量级)
    """
    
    # Architecture:
    # Input: [batch, 50, 10] (50 ticks × 10 features)
    # 
    # ┌─────────────────────────────────────┐
    # │  Embedding Layer (Linear)            │
    # │  Input 10 → Hidden 64              │
    # └─────────────────────────────────────┘
    #                   ↓
    # ┌─────────────────────────────────────┐
    # │  Transformer Encoder ×2           │
    # │  - MultiHeadAttention (4 heads)   │
    # │  - FeedForward (64→128→64)         │
    # │  - LayerNorm                       │
    # └─────────────────────────────────────┘
    #                   ↓
    # ┌─────────────────────────────────────┐
    # │  Output Layer                      │
    # │  Hidden 64 → 10 (prediction)       │
    # └─────────────────────────────────────┘
    #                   ↓
    # Output: [batch, 10] (未来10 tick方向)
```

### 4.2 模型配置

```python
ModelConfig = {
    "input_features": 10,      # 特征维度
    "hidden_dim": 64,          # 隐藏层维度
    "num_layers": 2,           # Transformer层数
    "num_heads": 4,            # 注意力头数
    "ffn_dim": 128,            # FFN维度
    "dropout": 0.1,           # Dropout
    "prediction_horizon": 10,  # 预测步数
    "total_params": "~50K",    # 参数量
}
```

### 4.3 替代方案: Mamba State Space Model

```python
# 可选: 用 Mamba SSM 替代 Transformer
# 优势: O(n) 推理复杂度, 更适合长序列
# 实现: 保持接口兼容, 可快速切换
```

---

## 5. Training Pipeline

### 5.1 数据流程

```python
# Training Pipeline:
# 1. Load Tick Data (Arrow/Parquet)
# 2. Compute Features (Numba加速)
# 3. Normalize (Rolling z-score)
# 4. Create Sequences (50 ticks → 10 prediction)
# 5. Train/Val/Test Split (8:1:1)
# 6. Train with AdamW + CosineAnnealing
# 7. Export ONNX
```

### 5.2 损失函数

```python
# 方案1: MSE Loss (回归价格)
loss = MSE(predicted_return, actual_return)

# 方案2: BCE Loss (分类方向)
loss = BCE(predicted_direction, actual_direction)

# 推荐: 组合损失
loss = 0.7 * MSE + 0.3 * BCE
```

### 5.3 训练配置

```python
TrainingConfig = {
    "batch_size": 256,
    "learning_rate": 1e-3,
    "epochs": 100,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "warmup_steps": 1000,
    "gradient_clip": 1.0,
    "early_stopping_patience": 10,
}
```

---

## 6. Inference Optimization

### 6.1 ONNX Export

```python
# ONNX Export Config:
torch.onnx.export(
    model,
    dummy_input,
    "tick_predictor.onnx",
    input_names=["features"],
    output_names=["prediction"],
    dynamic_axes={
        "features": {0: "batch_size"},
        "prediction": {0: "batch_size"}
    },
    opset_version=17
)
```

### 6.2 ONNX Runtime 优化

```python
# Session Options for <1ms Inference:
sess_options = SessionOptions()
sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intraop_num_threads = 1  #  pinned to 1 core
sess_options.inter_op_num_threads = 1

# Graph Optimization:
# - Constant Folding
# - Node Fusion
# - Memory Planning
```

### 6.3 量化策略

```python
# 量化方案:
# 1. FP32 → INT8 (post-training quantization)
# 2. 动态量化 (dynamic quant)
# 目标: 延迟 < 1ms on CPU
```

---

## 7. C++ Inference Interface

### 7.1 C++ Feature Extractor

```cpp
// feature_extractor.h
struct TickData {
    int64_t timestamp;
    double price;
    int volume;
    double bid_price;
    double ask_price;
    int bid_volume;
    int ask_volume;
};

class FeatureExtractor {
public:
    // 计算所有高频因子
    std::vector<float> compute_features(
        const std::vector<TickData>& ticks
    );
    
private:
    // 特征计算（与Python完全一致）
    float realized_volatility(const std::vector<float>& returns);
    float order_imbalance(int bid_vol, int ask_vol);
    float micro_price(double bid, double ask, int bid_vol, int askVol);
    // ... 其他因子
};
```

### 7.2 C++ ONNX推理

```cpp
// predictor.h
#include <onnxruntime_cxx_api.h>

class TickPredictor {
public:
    TickPredictor(const std::string& model_path);
    
    // 推理接口
    std::vector<float> predict(const std::vector<float>& features);
    
private:
    Ort::Env env_;
    Ort::Session session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};
```

---

## 8. Train-Test Skew Prevention

### 8.1 原则

1. **特征计算逻辑必须完全一致**
2. **使用相同的归一化参数**
3. **数据格式统一 (Arrow)**

### 8.2 对齐方案

```python
# 方案1: 共享特征计算库
feature_extractor/
├── python/
│   └── feature计算.py (training用)
└── cpp/
    └── feature计算.cpp (inference用)
    
# 方案2: 特征预计算 + 存储
# 线下: 预先计算所有特征, 存入Parquet
# 线上: 直接读取特征, 只做归一化
```

---

## 9. Project Structure

```
量化机器学习：基于 Transformer 的短期价格走势预测 (Tick-level Pred)/
├── SPEC.md                          # 本规范文档
├── README.md                       # 使用文档
├── requirements.txt              # Python依赖
├── data/                        # 数据目录
│   ├── sample_tick_data.parquet  # 示例Tick数据
│   └── features.parquet        # 预计算特征
├── python/                     # Python代码
│   ├── data/                  # 数据处理
│   │   ├── tick_loader.py     # Tick数据加载
│   │   └── featurizer.py    # 特征工程
│   ├── model/                # 模型定义
│   │   ├── transformer.py   # Transformer模型
│   │   └── config.py       # 模型配置
│   ├── training/             # 训练代码
│   │   ├── trainer.py      # 训练器
│   │   └── loss.py        # 损失函数
│   ├── export/              # 导出工具
│   │   └── onnx_exporter.py # ONNX导出
│   └── tests/               # 单元测试
│       └── test_features.py # 特征测试
├── cpp/                      # C++代码
│   ├── include/             # 头文件
│   │   ├── feature_extractor.h
│   │   └── predictor.h
│   └── src/                # 实现
│       ├── feature_extractor.cpp
│       └── predictor.cpp
├── notebooks/               # Jupyter notebooks
│   └── 01_eda.ipynb       # 数据探索
└── scripts/               # 脚本
    └── train.sh            # 训练脚本
```

---

## 10. Evaluation Metrics

### 10.1 核心指标

| 指标 | 说明 | 目标 |
|-----|------|------|
| **Direction Accuracy** | 方向预测准确率 | > 55% |
| **MSE** | 均方误差 | < 0.001 |
| **Inference Latency** | 推理延迟 | < 1ms |
| **Model Size** | 模型大小 | < 1MB |

### 10.2 风险指标

| 指标 | 说明 |
|-----|------|
| **Max Drawdown** | 最大回撤 |
| **Sharpe Ratio** | 夏普比率 |
| **Hit Rate** | 命中率 |

---

## 11. Success Criteria

- [ ] 特征工程模块完整实现 (10+ 高频因子)
- [ ] Transformer 模型训练完成
- [ ] ONNX 导出成功
- [ ] C++ 推理接口完成
- [ ] 推理延迟 < 1ms
- [ ] 方向准确率 > 55%

---

## 12. References

- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- ["Mamba: Linear-time Sequence Modeling"](https://arxiv.org/abs/2403.XXXXX) - Mamba SSM
- [ONNX Runtime Optimization](https://onnxruntime.ai/docs/performance/) - 推理优化指南
- [Apache Arrow](https://arrow.apache.org/) - 高性能数据格式