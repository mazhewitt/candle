# FLAN-T5 CoreML Example

This example demonstrates how to run FLAN-T5 inference using CoreML on macOS with the encoder-decoder architecture.

FLAN-T5 (Fine-tuned Language-Agnostic Network T5) is an enhanced version of T5 that has been fine-tuned on a diverse set of tasks using instruction tuning. It excels at following task instructions and can perform various NLP tasks like summarization, translation, question answering, and more.

## CoreML Models

This example uses separate CoreML models for the encoder and decoder:
- `flan_t5_base_encoder_quality.mlpackage` - Encodes the input text into hidden representations (FP32, high quality)
- `flan_t5_base_decoder_quality.mlpackage` - Generates output text conditioned on encoder states (FP32, high quality)

The models support up to 512 tokens sequence length and include both quality (FP32) and INT8 quantized versions. This example uses the quality versions for optimal accuracy.

## Setup

The example will automatically download the CoreML models from HuggingFace Hub and cache them locally. The models are stored in the default HuggingFace cache directory (`~/.cache/huggingface/hub/`).

If you want to pre-download the models, you can use:
```bash
# Download the entire repository
huggingface-cli download mazhewitt/flan-t5-base-coreml

# Or download individual models
huggingface-cli download mazhewitt/flan-t5-base-coreml flan_t5_base_encoder_quality.mlpackage
huggingface-cli download mazhewitt/flan-t5-base-coreml flan_t5_base_decoder_quality.mlpackage
```

**Alternative:** If you have the models locally, specify their paths using the command line options.

## Usage

Run the example:
```bash
# Basic usage with summarization task
cargo run --example flan-t5-coreml --features coreml -- "The quick brown fox jumps over the lazy dog. This is a simple sentence used for demonstration purposes."

# Specify a different task
cargo run --example flan-t5-coreml --features coreml -- --task-prefix "translate English to French:" "Hello, how are you today?"

# Question answering
cargo run --example flan-t5-coreml --features coreml -- --task-prefix "answer the question:" "What is the capital of France? Paris is the capital of France."

# Using local model files
cargo run --example flan-t5-coreml --features coreml -- \
  --encoder-model-file ./models/flan_t5_base_encoder_quality.mlpackage \
  --decoder-model-file ./models/flan_t5_base_decoder_quality.mlpackage \
  "Please summarize this text."
```

## Parameters

- `--task-prefix`: Task instruction for FLAN-T5 (default: "summarize:")
- `--sample-len`: Maximum number of tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-p`: Nucleus sampling probability cutoff
- `--repeat-penalty`: Penalty for repeating tokens (default: 1.1)
- `--encoder-model-file`: Path to encoder CoreML model
- `--decoder-model-file`: Path to decoder CoreML model
- `--tokenizer-file`: Path to tokenizer file

## Task Examples

FLAN-T5 can handle various instruction-following tasks:

**Summarization:**
```bash
--task-prefix "summarize:" "Long text to summarize..."
```

**Translation:**
```bash
--task-prefix "translate English to French:" "Hello world"
```

**Question Answering:**
```bash
--task-prefix "answer the question:" "Question: What is 2+2? Context: Basic arithmetic."
```

**Text Classification:**
```bash
--task-prefix "classify the sentiment:" "I love this movie!"
```

## Architecture

This example implements the complete encoder-decoder pipeline:

1. **Encoder**: Processes the input text with task prefix and generates hidden representations
2. **Decoder**: Auto-regressively generates output tokens conditioned on encoder states
3. **Tokenization**: Uses the T5 tokenizer with SentencePiece encoding

The CoreML models accept CPU tensors and handle the encoding/decoding internally, following the T5 architecture while optimized for Apple Silicon inference.

**Model Specifications:**
- **Max Sequence Length**: 512 tokens (both input and output)
- **Vocab Size**: 32,128 tokens
- **Input Shape**: `[1, 512]` for token IDs and attention masks
- **Encoder Output**: `[1, 512, 768]` hidden states  
- **Decoder Output**: `[1, 512, 32128]` logits over vocabulary

## Tokenizer

The example automatically downloads the correct tokenizer from the same repository as the CoreML models. The tokenizer has been specifically included to ensure compatibility with the converted models.

If you encounter any tokenizer issues, you can specify a custom tokenizer with `--tokenizer-file <path>`.

## Requirements

- macOS with Apple Silicon or Intel Mac
- Xcode command line tools
- CoreML feature enabled (`--features coreml`)
- Downloaded CoreML models from HuggingFace

## Performance

CoreML models are optimized for Apple hardware and should provide fast inference on macOS devices, especially on Apple Silicon Macs where the Neural Engine can be utilized. Typical performance is 10-15 tokens/s on Apple Silicon.

## Recent Updates

- **Extended sequence length**: Updated to support 512 tokens (previously 128)
- **Quality models**: Uses FP32 precision models for best accuracy
- **HuggingFace cache integration**: Follows standard Candle patterns for model discovery
- **Improved positioning logic**: Fixed decoder token generation for better output quality

## Example Outputs

```bash
# Input: "What is the capital of France?"
# Output: "paris" ✅

# Input: "What color is the sky?"  
# Output: "blue" ✅

# Input: "The quick brown fox jumps over the lazy dog..."
# Output: "The fox is jumping over the lazy dog." ✅
```

The example successfully demonstrates:
- ✅ Encoder-decoder pipeline with separate CoreML models
- ✅ Automatic model compilation (`.mlpackage` → `.mlmodelc`)
- ✅ Proper attention mask handling for 512-token sequences
- ✅ Token-by-token generation with correct positioning
- ✅ Configurable sampling parameters
- ✅ Multiple task types (summarization, Q&A, instruction following)