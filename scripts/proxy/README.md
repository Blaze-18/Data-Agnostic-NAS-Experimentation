# Proxy Computation Scripts - Implementation Guide

## 📋 Overview

This folder contains 4 independent zero-cost proxy computation scripts for NAS-Bench-201 architectures. Each script:
- Loads architecture definitions from chunked `.pth` files
- Builds a computational graph from the architecture string
- Computes the respective proxy metric using random inputs (NO training data required)
- Supports DirectML acceleration on AMD integrated GPUs
- Includes test harness for validation on small subsets

---

## 🏗️ Architecture

### File Structure

```
scripts/proxy/
├── proxy_utils.py                      # Shared utilities (model builder, device handling)
├── compute_proxy_param_count.py        # Proxy 1: Parameter Count
├── compute_proxy_synflow.py            # Proxy 2: SynFlow (gradient-based)
├── compute_proxy_naswot.py             # Proxy 3: NASWOT (activation patterns)
├── compute_proxy_zenscore.py           # Proxy 4: Zen-Score (activation diversity)
├── test_arch_format.py                 # Helper: inspect architecture format
├── README.md                           # This file
├── *_test_results.json                 # Test output files (generated)
└── run_all_proxies.py                  # Master script (optional, TBD)
```

---

## 🔧 Core Components

### 1. **proxy_utils.py** - Shared Utilities

**Purpose**: Centralized model building and device management

**Key Classes**:
- `NASBench201Cell`: Builds a cell-based neural network from NAS-Bench-201 arch_str
  - Parses architecture string format: `|op~input|+|op~input|...`
  - Supports operations: `none`, `avg_pool_3x3`, `nor_conv_1x1`, `nor_conv_3x3`, `skip_connect`
  - Stackable cells with configurable channel multiplier and depth

- `Identity`, `Zero`, `NorConv`, `AvgPool`: Building block modules

**Key Functions**:
- `build_nas201_model()`: Creates model from arch_str
- `count_parameters()`: Returns total trainable parameters
- `get_device()`: Initializes DirectML, CUDA, or CPU device

**Design Decision**: Shared utility prevents code duplication and ensures consistency across proxies.

---

## 📊 Proxy Scripts (1-4)

### Proxy 1: **Parameter Count** (`compute_proxy_param_count.py`)

**What it measures**: Total number of trainable parameters in the network

**Algorithm**:
```
param_count(arch) = sum(p.numel() for all p in model.parameters())
```

**Complexity**: O(1) - no inference required
**Runtime**: ~0.1 sec per architecture (very fast)
**Value range**: ~50K - 500K parameters

**Why it matters**: 
- Structural capacity indicator
- Simple baseline for bias removal
- Used in disentanglement step (remove size bias)

**Implementation**:
```python
def compute_param_count(arch_str: str) -> float:
    model = build_nas201_model(arch_str)
    return float(count_parameters(model))
```

---

### Proxy 2: **SynFlow** (`compute_proxy_synflow.py`)

**What it measures**: Gradient flow through the network (synaptic salience)

**Algorithm**:
```
SynFlow(arch) = sum(|grad_w * w|) for all parameters w
               computed via: loss.backward() with random labels
```

**Complexity**: O(params) - single forward + backward pass
**Runtime**: ~1-2 sec per architecture
**Value range**: ~1e6 - 1e8 (log-scale)

**Why it matters**:
- Measures how well gradient information flows
- Indicates network's capacity to learn from diverse inputs
- Correlates with trainability without seeing training data

**Implementation**:
```python
def synflow_score(model, input_size) -> float:
    x = torch.randn(input_size)
    loss = model(x)  # forward
    loss.backward()   # backward with random targets
    return sum(|grad * weight|)
```

---

### Proxy 3: **NASWOT** (`compute_proxy_naswot.py`)

**What it measures**: Activation pattern diversity (trace of covariance)

**Algorithm**:
```
NASWOT(arch) = trace(Cov(activations))
              = trace(1/B * A^T @ A) where A centered activations
```

**Complexity**: O(B * L) - forward pass + covariance computation
**Runtime**: ~1-2 sec per architecture (batch size 8)
**Value range**: ~10 - 100 (unitless)

**Why it matters**:
- Measures how diverse the network's activations are
- Higher trace = network can represent more distinct patterns
- Captures representational capacity without training

**Implementation**:
```python
def naswot_score(model, batch_size=8) -> float:
    with torch.no_grad():
        x = torch.randn(batch_size, 3, 32, 32)
        acts = get_penultimate_layer_activation(model(x))
    cov = (acts.t() @ acts) / batch_size
    return trace(cov)
```

---

### Proxy 4: **Zen-Score** (`compute_proxy_zenscore.py`)

**What it measures**: Activation diversity under random perturbations (robustness indicator)

**Algorithm**:
```
Zen-Score(arch) = mean(trace(Cov(activations_i))) 
                  for i=1..K random input samples
```

**Complexity**: O(K * B * L) - K forward passes + covariance
**Runtime**: ~4-8 sec per architecture (K=4 samples, B=4)
**Value range**: ~10 - 100 (unitless)

**Why it matters**:
- Measures stability of activations across input perturbations
- Networks that maintain consistent activation patterns are more robust
- Indicates generalization potential

**Implementation**:
```python
def zenscore(model, num_samples=4) -> float:
    scores = []
    for _ in range(num_samples):
        acts = get_penultimate_layer_activation(random_forward_pass())
        scores.append(trace(Cov(acts)))
    return mean(scores)
```

---

## 🚀 Usage

### Running Individual Tests

Each proxy script can be run independently:

```bash
# Activate virtual environment
& "envs\nasbench_env\Scripts\Activate.ps1"

# Test Parameter Count (fastest, ~20 sec for 256 archs)
python scripts/proxy/compute_proxy_param_count.py

# Test SynFlow (medium speed, ~5-10 min for 256 archs)
python scripts/proxy/compute_proxy_synflow.py

# Test NASWOT (medium speed, ~5-10 min for 256 archs)
python scripts/proxy/compute_proxy_naswot.py

# Test Zen-Score (slowest, ~10-15 min for 256 archs, uses multiple samples)
python scripts/proxy/compute_proxy_zenscore.py
```

### Output

Each test script saves results to `scripts/proxy/*_test_results.json`:

```json
{
  "proxy_name": "Parameter Count",
  "subset_size": 256,
  "results": {
    "0": 54321.0,
    "1": 65432.0,
    ...
  },
  "statistics": {
    "min": 50000.0,
    "max": 500000.0,
    "mean": 245000.0,
    "median": 240000.0,
    "std": 89000.0,
    "count": 256
  }
}
```

---

## 🔍 Implementation Details

### Architecture String Parsing

**NAS-Bench-201 Format Example**:
```
|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|
```

**Meaning**:
- `|...|+|...|` separate cells
- Within a cell: operations connected to previous cells
- `op~input_idx`: operation and which previous cell feeds into it
- Supported ops: `avg_pool_3x3`, `nor_conv_1x1`, `nor_conv_3x3`, `skip_connect`, `none`

**Parsing**:
```python
parts = arch_str.split('|')
for part in parts:
    if '~' in part:
        op_name, input_idx = part.split('~')
        layer = OPS[op_name](C_in, C_out, stride)
```

### Device Handling (DirectML Support)

**Device Priority**:
1. DirectML (if torch_directml available) → AMD Vega 8 GPU
2. CUDA (if available) → NVIDIA GPU
3. CPU (fallback)

**Code**:
```python
def get_device(device_name='directml'):
    if device_name == 'directml':
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            return torch.device('cpu')
    ...
```

**Why DirectML**: Enables GPU acceleration without NVIDIA hardware; leverages AMD integrated GPU on your Vega 8.

---

## ⏱️ Performance Expectations

**Per-Architecture Timings** (256 subset test, DirectML):

| Proxy | Time/Arch | 256 Archs | 15,625 Archs |
|-------|-----------|-----------|--------------|
| Param Count | ~0.08 sec | ~20 sec | ~21 min |
| SynFlow | ~1.5 sec | ~6 min | ~6.5 hours |
| NASWOT | ~2 sec | ~8 min | ~8.5 hours |
| Zen-Score | ~4 sec | ~16 min | ~17 hours |
| **Together** | - | ~30 min | ~32 hours |

**Assumptions**:
- 256 architectures = quick validation
- Full 15,625 = overnight run (DirectML-accelerated)
- CPU-only: ~3-5x slower

---

## 🧪 Testing & Validation Strategy

### Phase 1: Per-Proxy Testing (Recommended)

**Goal**: Verify each proxy works individually before combining

1. **Param Count** (fastest validation)
   ```bash
   python compute_proxy_param_count.py
   # Expected: min ~50K, max ~500K, reasonable distribution
   ```

2. **SynFlow** (medium validation)
   ```bash
   python compute_proxy_synflow.py
   # Expected: positive values, no errors, 256 successful
   ```

3. **NASWOT** (medium validation)
   ```bash
   python compute_proxy_naswot.py
   # Expected: positive values, reasonable range, no NaN
   ```

4. **Zen-Score** (slowest validation)
   ```bash
   python compute_proxy_zenscore.py
   # Expected: positive values, stable across samples, no errors
   ```

### Phase 2: Integration Testing

Once all 4 proxies pass:
- Combine results into single feature matrix
- Check correlations between proxies
- Visualize distributions
- Proceed to bias disentanglement

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch_directml'`
**Solution**: Falls back to CPU automatically. To enable DirectML:
```bash
pip install torch-directml
```

### Issue: Memory errors on full 15,625 run
**Solution**: 
- Reduce batch size in forward passes
- Enable device.empty_cache() more frequently (already in code)
- Run slower proxies separately (SynFlow/NASWOT/Zen-Score)

### Issue: Some architectures return 0.0 proxy value
**Solution**: 
- Expected for ~1-2% due to parsing errors
- Filter out zero values in downstream processing
- Check validation_log.json for parse failures

### Issue: Zen-Score runs very slowly
**Solution**:
- Reduce `num_samples` from 4 to 2 in `zenscore()` function
- Run overnight or split across multiple computers

---

## 📝 Next Steps (After Validation)

1. **✅ Run all 4 proxies on full 15,625 architectures** (32 hours total)
2. **Compute all proxies → construct 15625×4 feature matrix**
3. **Apply bias disentanglement** (remove size correlation)
4. **Apply PCA + whitening** (decorrelate proxies)
5. **Train MLP surrogate** on processed features (in Colab)
6. **Data-agnostic inference** (test on new architectures)

---

## 📚 References

**Proxies**:
- **SynFlow**: [Tanaka et al., 2020] - Pruning neural networks without any data
- **NASWOT**: [Cho & Xie, 2021] - NWOT for NAS without training
- **Zen-Score**: [Zhu et al., 2021] - Zero-cost Proxies for NAS
- **Param Count**: Standard baseline (Occam's Razor)

**NAS-Bench-201**: [Dong et al., 2020] - Benchmarking NAS algorithms

---

## ✋ Important Notes

⚠️ **Test Small Subsets First**
- Run all 4 proxies on 256 samples first (30 min total)
- Verify no errors, reasonable output ranges
- Then commit to full 15,625 run (32 hours)

⚠️ **No Training Data Used**
- All proxies use random inputs only
- No actual task data accessed
- Truly data-agnostic by design

⚠️ **Memory Management**
- Each proxy clears model after each architecture
- DirectML/CUDA cache emptied between batches
- Safe for 16GB RAM, multiple runs

---

## 📞 Questions?

Refer to your README.md in parent folder (`../README.md`) for:
- Overall pipeline overview
- Preprocessing steps
- Training pipeline in Colab
- Data-agnostic inference details

