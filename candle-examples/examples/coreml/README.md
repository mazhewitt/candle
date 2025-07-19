# candle-coreml

Candle implementation for running CoreML models on macOS.

## Text Generation Example

This example uses CoreML models for text generation, specifically designed for Apple silicon optimization through Core ML.

```bash
$ cargo run --example coreml --features coreml --release -- "The capital of France is"
...
The capital of France is Paris.
```

## Model Download and Setup

Due to CoreML models being directory structures (.mlpackage), they need to be downloaded manually using the HuggingFace CLI:

```bash
# Download the default model
huggingface-cli download corenet-community/coreml-OpenELM-450M-Instruct \
  --local-dir ./coreml-OpenELM-450M-Instruct

# Compile the .mlpackage to .mlmodelc for faster loading
xcrun coremlc compile ./coreml-OpenELM-450M-Instruct/OpenELM-450M-Instruct-128-float32.mlpackage/Data/com.apple.CoreML/model.mlmodel ./coreml-OpenELM-450M-Instruct/OpenELM-450M-Instruct-128-float32.mlmodelc

# Then run the example
cargo run --example coreml --features coreml -- "The capital of France is"
```

The tokenizer will be automatically downloaded from the original PyTorch model repository.

Default model: `corenet-community/coreml-OpenELM-450M-Instruct`

## Using custom models

To use a different model, specify the `model-id`:

```bash
$ cargo run --example coreml --features coreml --release -- --model-id "your-org/your-coreml-model" "Your prompt here"
```

You can also specify local files:

```bash
# Using local CoreML model and tokenizer
$ cargo run --example coreml --features coreml --release -- --model-file "./path/to/model.mlmodelc" --tokenizer-file "./path/to/tokenizer.json" "Your prompt"

# Using local model with downloaded tokenizer
$ cargo run --example coreml --features coreml --release -- --model-file "./model.mlpackage" "Your prompt"
```

## Creating CoreML Models

To create your own CoreML models for use with this example:

1. Start with a PyTorch model (e.g., from HuggingFace)
2. Convert to CoreML using Apple's `coremltools`:
   ```python
   import coremltools as ct
   # Conversion code here
   ```
3. Save as `.mlpackage` or compile to `.mlmodelc`

## Requirements

- macOS (CoreML is only available on Apple platforms)
- Xcode command line tools (for CoreML runtime)
- `coreml` feature enabled

## Model Format

This example expects:
- `.mlpackage` or `.mlmodelc` files (CoreML models)
- `tokenizer.json` files compatible with the HuggingFace tokenizers library

CoreML supports both `.mlpackage` (newer format) and `.mlmodelc` (compiled format). The default model uses `.mlpackage`.

To compile a `.mlmodel` to `.mlmodelc`, use Apple's `coremlc` tool:
```bash
coremlc compile model.mlmodel model.mlmodelc
```