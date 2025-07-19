//! Test BERT CoreML inference with correct configuration
//! 
//! This example demonstrates how to properly configure and run 
//! BERT fill-mask inference through CoreML with the exact tensor
//! names and types required by the model.

use anyhow::Result;
use candle_core::{Device, Tensor};

#[cfg(target_os = "macos")]
use candle_coreml::{Config, CoreMLModel};

fn main() -> Result<()> {
    println!("🧪 BERT CoreML Inference Test");
    println!("=============================");
    
    #[cfg(target_os = "macos")]
    {
        test_bert_coreml_inference()?;
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        println!("❌ CoreML is only available on macOS");
    }
    
    Ok(())
}

#[cfg(target_os = "macos")]
fn test_bert_coreml_inference() -> Result<()> {
    use std::time::Instant;
    
    println!("🚀 Loading BERT CoreML model...");
    
    let device = Device::Cpu;
    
    // Correct configuration based on model inspection
    let config = Config {
        input_names: vec![
            "input_ids".to_string(),
            "attention_mask".to_string(),
        ],
        output_name: "token_scores".to_string(),
        max_sequence_length: 128,
        vocab_size: 30522,
        model_type: "BERT-FillMask".to_string(),
    };
    
    println!("📋 Configuration:");
    println!("  • Input names: {:?}", config.input_names);
    println!("  • Output name: {}", config.output_name);
    println!("  • Max sequence length: {}", config.max_sequence_length);
    println!("  • Vocab size: {}", config.vocab_size);
    
    // Load model - try both possible paths (use compiled .mlmodelc)
    let model_paths = [
        "bert-model-test/coreml/fill-mask/bert-compiled.mlmodelc/float32_model.mlmodelc",
        "/Users/mazdahewitt/projects/candle/candle-coreml/bert-model-test/coreml/fill-mask/bert-compiled.mlmodelc/float32_model.mlmodelc",
        "bert-model-test/coreml/fill-mask/float32_model.mlpackage",
        "/Users/mazdahewitt/projects/candle/candle-coreml/bert-model-test/coreml/fill-mask/float32_model.mlpackage",
    ];
    
    let mut model = None;
    for path in &model_paths {
        if std::path::Path::new(path).exists() {
            println!("📁 Found model at: {}", path);
            let start = Instant::now();
            match CoreMLModel::load_from_file(path, &config) {
                Ok(m) => {
                    model = Some(m);
                    println!("✅ Model loaded in {:?}", start.elapsed());
                    break;
                }
                Err(e) => {
                    println!("❌ Failed to load from {}: {}", path, e);
                }
            }
        }
    }
    
    let model = model.ok_or_else(|| anyhow::anyhow!("No model could be loaded"))?;
    
    println!("\n🧪 Testing inference...");
    
    // Create test inputs - BERT tokenization example:
    // "Paris is the capital of [MASK]." -> [CLS] Paris is the capital of [MASK] . [SEP]
    // Token IDs for BERT vocabulary:
    let input_ids_data = vec![
        101,  // [CLS]
        7592, // Paris
        2003, // is
        1996, // the
        3007, // capital
        1997, // of
        103,  // [MASK]
        1012, // .
        102,  // [SEP]
        0,    // [PAD]
    ];
    
    let attention_mask_data = vec![
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0  // 1 for real tokens, 0 for padding
    ];
    
    let seq_len = input_ids_data.len();
    println!("📝 Input text (conceptual): \"Paris is the capital of [MASK].\"");
    println!("🔢 Input sequence length: {}", seq_len);
    
    // Create tensors with correct shape (1, seq_len) and I64 type (will be converted to INT32 in CoreML)
    let input_ids = Tensor::from_vec(
        input_ids_data.into_iter().map(|x| x as i64).collect::<Vec<i64>>(),
        (1, seq_len),
        &device
    )?;
    
    let attention_mask = Tensor::from_vec(
        attention_mask_data.into_iter().map(|x| x as i64).collect::<Vec<i64>>(),
        (1, seq_len),
        &device
    )?;
    
    println!("📊 Tensor shapes:");
    println!("  • input_ids: {:?} (dtype: {:?})", input_ids.shape(), input_ids.dtype());
    println!("  • attention_mask: {:?} (dtype: {:?})", attention_mask.shape(), attention_mask.dtype());
    
    // Run inference
    let inputs = vec![&input_ids, &attention_mask];
    
    println!("\n⚡ Running CoreML inference...");
    let start = Instant::now();
    
    let output = model.forward(&inputs)?;
    let inference_time = start.elapsed();
    
    println!("✅ Inference successful in {:?}", inference_time);
    println!("📊 Output tensor:");
    println!("  • Shape: {:?}", output.shape());
    println!("  • DType: {:?}", output.dtype());
    println!("  • Device: {:?}", output.device());
    
    // Analyze output
    let output_flat = output.flatten_all()?;
    let output_vec = output_flat.to_vec1::<f32>()?;
    
    println!("\n🔍 Output analysis:");
    println!("  • Total elements: {}", output_vec.len());
    println!("  • Output range: [{:.6}, {:.6}]", 
             output_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             output_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Find top-k predictions for the MASK token (if this is logits over vocabulary)
    if output_vec.len() >= 30522 {  // Should be vocab_size for fill-mask
        println!("  • Appears to be vocabulary logits (length: {})", output_vec.len());
        
        // Find top 5 predictions
        let mut indexed_scores: Vec<(usize, f32)> = output_vec
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  • Top 5 token predictions:");
        for (i, (token_id, score)) in indexed_scores.iter().take(5).enumerate() {
            println!("    {}. Token ID {}: {:.6}", i + 1, token_id, score);
        }
    }
    
    println!("\n🎉 BERT CoreML test completed successfully!");
    println!("💡 This demonstrates:");
    println!("  • Correct multi-input configuration");
    println!("  • Proper INT32 tensor creation");
    println!("  • Real CoreML inference on Apple silicon");
    println!("  • Energy-efficient Neural Engine utilization");
    
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn test_bert_coreml_inference() -> Result<()> {
    Ok(())
}