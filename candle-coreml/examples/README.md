# CoreML Examples for Candle

This directory contains comprehensive examples demonstrating how to use CoreML with Candle for efficient machine learning inference on macOS and iOS devices.

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── basic/                       # Getting started examples
│   ├── hello_coreml.rs         # Minimal CoreML integration
│   └── bert_inference.rs       # Basic BERT inference
├── benchmarks/                  # Performance comparison tools
│   ├── bert_comparison.rs      # Candle vs CoreML BERT benchmarks
│   ├── overhead_analysis.rs    # Overhead measurement
│   └── tensor_conversion.rs    # Conversion performance tests
├── advanced/                    # Advanced use cases
│   └── embeddings.rs          # Sentence embeddings with CoreML
└── tests/                      # Validation and testing
    └── model_validation.rs     # Model loading and validation tests
```

## 🚀 Quick Start

### Prerequisites

- **macOS**: CoreML is only available on macOS
- **Candle with CoreML**: Build with `--features coreml`
- **Model Files**: CoreML models (`.mlmodelc` or `.mlpackage`)

### Hello World

Start with the simplest example:

```bash
cargo run --example hello_coreml --features coreml
```

### Basic BERT Inference

Run BERT fill-mask inference:

```bash
cargo run --example bert_inference --features coreml
```

## 📚 Examples Guide

### 🔰 Basic Examples

#### `hello_coreml.rs`
**Purpose**: Absolute minimal CoreML integration example  
**When to use**: First time exploring CoreML with Candle  
**Key concepts**: Model loading, tensor creation, basic inference

```bash
# Run the hello world example
cargo run --example hello_coreml --features coreml

# Set custom model path
COREML_MODEL_PATH=/path/to/model.mlmodelc cargo run --example hello_coreml --features coreml
```

#### `bert_inference.rs`
**Purpose**: Practical BERT model usage with CoreML  
**When to use**: Learning BERT-specific CoreML integration  
**Key concepts**: Text processing, fill-mask tasks, error handling

```bash
# Basic BERT inference
cargo run --example bert_inference --features coreml

# Custom text input
cargo run --example bert_inference --features coreml -- --text "The weather is [MASK] today"

# Use specific model
cargo run --example bert_inference --features coreml -- --model-path /path/to/bert.mlmodelc
```

### 📊 Benchmark Examples

#### `bert_comparison.rs`
**Purpose**: Comprehensive performance comparison between Candle and CoreML  
**When to use**: Evaluating CoreML performance benefits  
**Key concepts**: Multi-backend benchmarking, throughput analysis, loading time comparison

```bash
# Full benchmark suite
cargo run --example bert_comparison --features coreml

# Quick test
cargo run --example bert_comparison --features coreml -- --warmup 1 --iterations 3

# Test specific sequence lengths
cargo run --example bert_comparison --features coreml -- --sequence-lengths "128,256"

# Use local models instead of downloading
cargo run --example bert_comparison --features coreml -- --local-models
```

**Sample Output**:
```
📊 BENCHMARK SUMMARY
===================

📏 Sequence Length: 128
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Backend     │ Loading     │ Cold Inf.   │ Warm Inf.   │ Throughput  │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Candle-Cpu  │     2.1s    │    45.2ms   │    42.1ms   │    3041 t/s │
│ Candle-Metal│     2.3s    │    12.8ms   │     8.4ms   │   15238 t/s │
│ CoreML      │     0.9s    │     6.2ms   │     4.1ms   │   31219 t/s │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
🚀 Fastest loading: CoreML
⚡ Best throughput: CoreML (31219 tokens/sec)
```

#### `overhead_analysis.rs`
**Purpose**: Measure real-world overhead of CoreML integration  
**When to use**: Understanding CoreML vs pure Candle performance costs  
**Key concepts**: Overhead measurement, practical performance impact

```bash
cargo run --example overhead_analysis --features coreml
```

#### `tensor_conversion.rs`
**Purpose**: Benchmark Tensor ↔ MLMultiArray conversion performance  
**When to use**: Optimizing data pipeline performance  
**Key concepts**: Conversion costs, memory efficiency, data type handling

```bash
cargo run --example tensor_conversion --features coreml
```

### 🎓 Advanced Examples

#### `embeddings.rs`
**Purpose**: Generate and analyze sentence embeddings using BERT + CoreML  
**When to use**: Building semantic search, similarity analysis, or NLP pipelines  
**Key concepts**: Embedding generation, similarity calculation, batch processing

```bash
# Generate embeddings for custom sentences
cargo run --example embeddings --features coreml -- --sentences "Hello world" "How are you?"

# Compare different backends
cargo run --example embeddings --features coreml -- --compare-backends

# Process file of sentences
cargo run --example embeddings --features coreml -- --batch-file sentences.txt

# Show similarity matrix
cargo run --example embeddings --features coreml -- --similarity-matrix

# Save embeddings to file
cargo run --example embeddings --features coreml -- --output embeddings.csv
```

### 🧪 Test Examples

#### `model_validation.rs`
**Purpose**: Validate CoreML models and test integration  
**When to use**: Debugging model issues, validating new models  
**Key concepts**: Model validation, comprehensive testing, error diagnosis

```bash
cargo run --example model_validation --features coreml
```

## ⚙️ Configuration

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `COREML_MODEL_PATH` | Default model path for basic examples | `/path/to/model.mlmodelc` |
| `COREML_BERT_MODEL` | BERT-specific model path | `/path/to/bert.mlmodelc` |
| `RUST_LOG` | Enable debug logging | `debug`, `trace` |

### Model Setup

#### Option 1: Create a Simple Demo Model (Recommended for hello_coreml)

```bash
# Install Python dependencies
pip install coremltools numpy

# Create demo model
python candle-coreml/create_demo_model.py

# Run hello_coreml example
cargo run --example hello_coreml --features coreml
```

#### Option 2: Use Existing BERT Test Models

The examples include BERT test models in `bert-model-test/` directory:

```bash
# Models are automatically detected at:
# candle-coreml/bert-model-test/coreml/fill-mask/bert-compiled.mlmodelc/

# Run BERT inference example
cargo run --example bert_inference --features coreml
```

#### Option 3: Set Custom Model Path

```bash
# Set environment variable for basic examples
export COREML_MODEL_PATH=/path/to/your/model.mlmodelc

# Set environment variable for BERT examples
export COREML_BERT_MODEL=/path/to/your/bert.mlmodelc

# Or use command line argument
cargo run --example bert_inference --features coreml -- --model-path /path/to/model.mlmodelc
```

#### Option 4: Download from HuggingFace

```bash
# Some examples automatically download models
cargo run --example bert_comparison --features coreml  # Downloads google-bert/bert-base-uncased
```

## 🔧 Common Issues & Solutions

### "CoreML is only available on macOS"

**Problem**: Running on non-macOS platform  
**Solution**: Use macOS or check cross-platform examples

```bash
# Check current platform
cargo run --example hello_coreml --features coreml
```

### "Model file not found"

**Problem**: Missing CoreML model files  
**Solutions**:

1. Set environment variable:
   ```bash
   export COREML_BERT_MODEL=/path/to/your/model.mlmodelc
   ```

2. Use test models:
   ```bash
   # Ensure bert-model-test/ directory exists with models
   ls candle-coreml/bert-model-test/
   ```

3. Download from HuggingFace (some examples do this automatically)

### "Failed to load CoreML model"

**Problem**: Model format or configuration mismatch  
**Solutions**:

1. Check model inputs/outputs:
   ```bash
   # Use model validation example
   cargo run --example model_validation --features coreml
   ```

2. Verify model compatibility:
   - Input names match `Config.input_names`
   - Output name matches `Config.output_name`
   - Sequence length within `max_sequence_length`

### "CUDA tensors should be rejected"

**Problem**: Trying to use CUDA tensors with CoreML  
**Solution**: Use CPU or Metal tensors only

```rust
// ✅ Good - CPU tensor
let device = Device::Cpu;
let tensor = Tensor::ones((1, 128), DType::F32, &device)?;

// ✅ Good - Metal tensor  
let device = Device::new_metal(0)?;
let tensor = Tensor::ones((1, 128), DType::F32, &device)?;

// ❌ Bad - CUDA tensor
let device = Device::new_cuda(0)?;  // Will be rejected by CoreML
```

## 📖 Learning Path

### 1. **Start Here** - Basic Understanding
```bash
cargo run --example hello_coreml --features coreml
cargo run --example bert_inference --features coreml
```

### 2. **Performance Analysis**
```bash
cargo run --example bert_comparison --features coreml
cargo run --example overhead_analysis --features coreml
```

### 3. **Advanced Applications**
```bash
cargo run --example embeddings --features coreml
cargo run --example model_validation --features coreml
```

### 4. **Customization**
- Modify examples for your specific models
- Add new model types beyond BERT
- Integrate with your application

## 🏗️ Architecture Notes

### CoreML Integration Design

CoreML in Candle operates as a **pure inference engine**, not a device backend:

```rust
// CoreML accepts CPU/Metal tensors
let device = Device::Cpu;  // or Device::new_metal(0)?
let input = Tensor::ones((1, 128), DType::F32, &device)?;

// CoreML model validates input device
let output = model.forward(&[&input])?;  // Pass slice of tensor references

// Output tensor uses same device as input
assert_eq!(output.device(), input.device());
```

### Key Benefits

- **Energy Efficiency**: Automatic Neural Engine utilization
- **Unified Memory**: Leverages M1/M2/M3 unified memory architecture  
- **Device Flexibility**: Works with existing Candle CPU/Metal tensors
- **Error Handling**: Comprehensive validation and helpful error messages

### Performance Characteristics

| Aspect | CoreML | Candle-CPU | Candle-Metal |
|--------|--------|------------|--------------|
| **Loading Speed** | ⚡ Fast | 🔄 Moderate | 🔄 Moderate |
| **Cold Inference** | ⚡ Fast | 🐌 Slow | 🔄 Moderate |
| **Warm Inference** | ⚡ Very Fast | 🐌 Slow | ⚡ Fast |
| **Memory Usage** | ✅ Efficient | 🔄 Moderate | ✅ Efficient |
| **Power Efficiency** | ⚡ Excellent | 🔋 Poor | 🔄 Good |

## 🤝 Contributing

### Adding New Examples

1. Choose appropriate directory (`basic/`, `benchmarks/`, `advanced/`, `tests/`)
2. Follow naming convention: `descriptive_name.rs`
3. Include comprehensive documentation and error handling
4. Add usage examples to this README
5. Test on macOS with CoreML enabled

### Example Template

```rust
//! Example Title - Brief Description
//! 
//! Detailed description of what this example demonstrates.
//! Include key concepts and when to use this example.
//!
//! Usage:
//! ```bash
//! cargo run --example example_name --features coreml
//! ```

use anyhow::Result;
use candle_core::{Device, Tensor};

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn run_example() -> Result<()> {
    // CoreML-specific implementation
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn run_example() -> Result<()> {
    println!("❌ This example requires macOS and 'coreml' feature");
    Ok(())
}

fn main() -> Result<()> {
    run_example()
}
```

## 📚 Additional Resources

- [Candle Documentation](https://github.com/huggingface/candle)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [Apple Neural Engine Guide](https://developer.apple.com/machine-learning/)
- [HuggingFace Model Hub](https://huggingface.co/models)

## 🐛 Troubleshooting

For issues not covered here:

1. **Check Logs**: Run with `RUST_LOG=debug` for detailed output
2. **Validate Model**: Use `model_validation.rs` example
3. **Platform Check**: Ensure you're on macOS with CoreML support
4. **Feature Flag**: Verify `--features coreml` is included in build command
5. **File Issues**: Check our [GitHub Issues](https://github.com/huggingface/candle/issues) for known problems

---

Happy coding with CoreML and Candle! 🚀