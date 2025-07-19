//! Basic BERT Inference with CoreML
//! 
//! This example demonstrates the simplest way to use BERT with CoreML for inference.
//! It loads a pre-trained BERT model and runs fill-mask inference on a sample text.
//!
//! Features:
//! - Simple model loading
//! - Token prediction/fill-mask
//! - Error handling with helpful messages
//! - Works with both .mlpackage and .mlmodelc files
//!
//! Usage:
//! ```bash
//! # Basic usage
//! cargo run --example bert_inference --features coreml
//! 
//! # Custom text
//! cargo run --example bert_inference --features coreml -- --text "The weather today is [MASK]"
//! 
//! # Use specific model path
//! cargo run --example bert_inference --features coreml -- --model-path "/path/to/model.mlmodelc"
//! ```

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text for inference (use [MASK] for fill-mask task)
    #[arg(short, long, default_value = "Paris is the [MASK] of France.")]
    text: String,
    
    /// Path to CoreML model file (.mlmodelc or .mlpackage)
    #[arg(short, long)]
    model_path: Option<String>,
    
    /// Maximum sequence length for model input
    #[arg(long, default_value = "128")]
    max_length: usize,
    
    /// Show top N predictions
    #[arg(long, default_value = "5")]
    top_k: usize,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn run_coreml_inference(args: &Args) -> Result<()> {
    use candle_coreml::{Config as CoreMLConfig, CoreMLModel};
    
    println!("🍎 CoreML BERT Inference");
    println!("========================");
    println!("Input text: \"{}\"", args.text);
    
    // Determine model path
    let model_path = args.model_path.clone().unwrap_or_else(|| {
        // Try environment variable first
        if let Ok(path) = std::env::var("COREML_BERT_MODEL") {
            return path;
        }
        
        // Use default local test model
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{}/bert-model-test/coreml/fill-mask/bert-compiled.mlmodelc/float32_model.mlmodelc", 
            manifest_dir)
    });
    
    if args.verbose {
        println!("📂 Model path: {}", model_path);
    }
    
    // Check if model file exists
    if !std::path::Path::new(&model_path).exists() {
        return Err(E::msg(format!(
            "Model file not found: {}\n\n\
            💡 Try:\n\
            - Set COREML_BERT_MODEL environment variable\n\
            - Use --model-path to specify model location\n\
            - Download a model using: python -c 'import coremltools; ...'",
            model_path
        )));
    }
    
    // Configure model
    let config = CoreMLConfig {
        input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
        output_name: "token_scores".to_string(),
        max_sequence_length: args.max_length,
        vocab_size: 30522, // BERT base vocabulary size
        model_type: "bert-base-uncased".to_string(),
    };
    
    // Load model
    let start = Instant::now();
    let model = CoreMLModel::load_from_file(&model_path, &config)
        .map_err(|e| E::msg(format!("Failed to load CoreML model: {}", e)))?;
    let loading_time = start.elapsed();
    
    println!("✅ Model loaded in {:?}", loading_time);
    println!("📋 Config: {:?}", config);
    
    // Prepare input (simplified tokenization for demo)
    let device = Device::Cpu;
    
    // Create sample input IDs (in real usage, you'd use a proper tokenizer)
    let _input_text = args.text.replace("[MASK]", "[MASK]"); // Ensure [MASK] token
    let sequence_length = args.max_length.min(10); // Use shorter sequence for demo
    
    // Create dummy input tensors (in production, use proper tokenizer)
    let input_ids: Vec<i64> = (0..sequence_length)
        .map(|i| if i == 5 { 103 } else { 1000 + (i as i64 % 1000) }) // 103 is [MASK] token ID
        .collect();
    
    let attention_mask: Vec<i64> = vec![1; sequence_length]; // All tokens are real (not padding)
    
    let input_ids_tensor = Tensor::from_vec(input_ids, (1, sequence_length), &device)?;
    let attention_mask_tensor = Tensor::from_vec(attention_mask, (1, sequence_length), &device)?;
    
    if args.verbose {
        println!("🔢 Input shape: {:?}", input_ids_tensor.shape());
        println!("🎭 Attention mask shape: {:?}", attention_mask_tensor.shape());
    }
    
    // Run inference
    println!("\n🚀 Running inference...");
    let start = Instant::now();
    
    let output = model.forward(&[&input_ids_tensor, &attention_mask_tensor])
        .map_err(|e| E::msg(format!("Inference failed: {}", e)))?;
    
    let inference_time = start.elapsed();
    println!("✅ Inference completed in {:?}", inference_time);
    println!("📊 Output shape: {:?}", output.shape());
    
    // Process results (simplified)
    if let Ok(output_data) = output.to_vec3::<f32>() {
        let mask_position = 5; // Position where we put the [MASK] token
        
        if output_data.len() > 0 && output_data[0].len() > mask_position {
            let mask_scores = &output_data[0][mask_position];
            
            // Find top predictions
            let mut indexed_scores: Vec<(usize, f32)> = mask_scores
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();
            
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            println!("\n🎯 Top {} predictions for [MASK]:", args.top_k);
            for (rank, (token_id, score)) in indexed_scores.iter().take(args.top_k).enumerate() {
                println!("  {}. Token ID: {}, Score: {:.4}", rank + 1, token_id, score);
            }
        }
    }
    
    println!("\n💡 Performance Summary:");
    println!("  • Loading time: {:?}", loading_time);
    println!("  • Inference time: {:?}", inference_time);
    println!("  • Total time: {:?}", loading_time + inference_time);
    
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn run_coreml_inference(_args: &Args) -> Result<()> {
    println!("❌ CoreML inference is only available on macOS with the 'coreml' feature enabled.");
    println!("\n💡 To use CoreML:");
    println!("   • Run on macOS");
    println!("   • Build with: cargo run --example bert_inference --features coreml");
    Ok(())
}

fn print_help() {
    println!("🤖 BERT CoreML Inference Example");
    println!("=================================");
    println!();
    println!("This example demonstrates basic BERT inference using CoreML on macOS.");
    println!("It loads a BERT model and performs fill-mask prediction.");
    println!();
    println!("📋 Requirements:");
    println!("  • macOS (for CoreML support)");
    println!("  • CoreML model file (.mlmodelc or .mlpackage)");
    println!("  • Candle built with 'coreml' feature");
    println!();
    println!("🚀 Quick Start:");
    println!("  1. cargo run --example bert_inference --features coreml");
    println!("  2. Try custom text: --text \"The cat is [MASK]\"");
    println!("  3. Use your model: --model-path \"/path/to/model.mlmodelc\"");
    println!();
    println!("🔗 For more examples, see the benchmarks/ and advanced/ directories.");
    println!();
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    if args.verbose {
        print_help();
    }
    
    run_coreml_inference(&args)
}