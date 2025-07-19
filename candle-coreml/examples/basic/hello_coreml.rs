//! Hello CoreML - Minimal CoreML Integration Example
//! 
//! This is the simplest possible example showing how to:
//! 1. Load a CoreML model
//! 2. Create input tensors
//! 3. Run inference
//! 4. Handle results
//!
//! Perfect starting point for understanding CoreML integration with Candle.
//!
//! Usage:
//! ```bash
//! cargo run --example hello_coreml --features coreml
//! ```

use anyhow::Result;
use candle_core::{Device, Tensor};

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn hello_coreml() -> Result<()> {
    use candle_coreml::{Config, CoreMLModel};
    
    println!("👋 Hello CoreML!");
    println!("================");
    
    // Step 1: Configure the model (will be adjusted based on detected model type)
    println!("📋 Setting up configuration...");
    
    // Step 2: Try to load a model and detect its type
    println!("🔍 Looking for CoreML model...");
    
    // Try environment variable first, then fallback to existing BERT model
    let model_path = std::env::var("COREML_MODEL_PATH").unwrap_or_else(|_| {
        // Use existing BERT test model as demo
        let bert_path = format!("{}/bert-model-test/coreml/fill-mask/bert-compiled.mlmodelc/float32_model.mlmodelc", 
            env!("CARGO_MANIFEST_DIR"));
        
        if std::path::Path::new(&bert_path).exists() {
            bert_path
        } else {
            // Fallback to models directory
            format!("{}/models/demo.mlmodelc", env!("CARGO_MANIFEST_DIR"))
        }
    });
    
    // Detect model type and create appropriate config
    let is_bert_model = model_path.contains("bert") || model_path.contains("BERT");
    let config = if is_bert_model {
        println!("🤖 Detected BERT model, using BERT configuration...");
        Config {
            input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
            output_name: "token_scores".to_string(),
            max_sequence_length: 10,
            vocab_size: 30522, // BERT vocabulary size
            model_type: "bert".to_string(),
        }
    } else {
        println!("🎯 Using simple demo configuration...");
        Config {
            input_names: vec!["input".to_string()],
            output_name: "output".to_string(),
            max_sequence_length: 10,
            vocab_size: 1000,
            model_type: "demo".to_string(),
        }
    };
    
    println!("📋 Config: input_names={:?}, output_name={}", config.input_names, config.output_name);
    
    match CoreMLModel::load_from_file(&model_path, &config) {
        Ok(model) => {
            println!("✅ CoreML model loaded from: {}", model_path);
            
            // Step 3: Create input tensor(s) based on model type
            println!("🔢 Creating input tensor(s)...");
            let device = Device::Cpu;
            
            let (input_tensors, input_description) = if is_bert_model {
                // Create BERT-style inputs: input_ids and attention_mask
                let seq_len = 5;
                let input_ids_data: Vec<i64> = (0..seq_len).map(|i| 1000 + i as i64).collect(); // Token IDs
                let attention_mask_data: Vec<i64> = vec![1; seq_len]; // All tokens are real (not padding)
                
                let input_ids = Tensor::from_vec(input_ids_data, (1, seq_len), &device)?;
                let attention_mask = Tensor::from_vec(attention_mask_data, (1, seq_len), &device)?;
                
                println!("   Input IDs shape: {:?}", input_ids.shape());
                println!("   Attention mask shape: {:?}", attention_mask.shape());
                
                (vec![input_ids, attention_mask], "BERT inputs (input_ids + attention_mask)")
            } else {
                // Create simple demo input
                let input_data = vec![1i64, 2, 3]; // Simple 3-element input for demo model
                let input_tensor = Tensor::from_vec(input_data, (3,), &device)?;
                
                println!("   Input shape: {:?}", input_tensor.shape());
                
                (vec![input_tensor], "Simple demo input")
            };
            
            println!("   Input device: {:?}", device);
            println!("   Input type: {}", input_description);
            
            // Step 4: Run inference
            println!("🚀 Running inference...");
            let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
            match model.forward(&input_refs) {
                Ok(output) => {
                    println!("✅ Inference successful!");
                    println!("   Output shape: {:?}", output.shape());
                    println!("   Output device: {:?}", output.device());
                    
                    // Show sample output values based on model type
                    if is_bert_model {
                        if let Ok(values) = output.to_vec3::<f32>() {
                            println!("   BERT output dimensions: [batch, sequence, vocab]");
                            if let Some(batch) = values.get(0) {
                                if let Some(first_token_logits) = batch.get(0) {
                                    let sample_logits: Vec<f32> = first_token_logits.iter().take(5).cloned().collect();
                                    println!("   First token logits (sample): {:?}", sample_logits);
                                    
                                    // Find top prediction for first token
                                    if let Some((max_idx, max_val)) = first_token_logits.iter()
                                        .enumerate()
                                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
                                        println!("   Top prediction: token_id={}, score={:.4}", max_idx, max_val);
                                    }
                                }
                            }
                        }
                    } else {
                        // Simple demo model output
                        if let Ok(values) = output.to_vec1::<f32>() {
                            println!("   Demo output values: {:?}", values);
                            println!("   Expected (input * 2 + 1): [3.0, 5.0, 7.0]");
                        } else if let Ok(values) = output.to_vec2::<f32>() {
                            println!("   Demo output values: {:?}", 
                                values.get(0).map(|row| &row[0..row.len().min(5)]));
                        }
                    }
                },
                Err(e) => {
                    println!("⚠️  Inference failed: {}", e);
                    println!("   This is normal if the model format doesn't match our config");
                }
            }
        },
        Err(e) => {
            println!("❌ Could not load CoreML model: {}", e);
            println!();
            println!("💡 To get a CoreML model for testing:");
            println!();
            println!("🎯 Option 1 - Use existing BERT model:");
            println!("   The examples include a working BERT model for testing.");
            println!("   Try: cargo run --example bert_inference --features coreml");
            println!();
            println!("🎯 Option 2 - Set custom model:");
            println!("   export COREML_MODEL_PATH=/path/to/your/model.mlmodelc");
            println!("   cargo run --example hello_coreml --features coreml");
            println!();
            println!("🎯 Option 3 - Create a simple model:");
            println!("   # Python script to create demo model");
            println!("   import coremltools as ct");
            println!("   import numpy as np");
            println!("   ");
            println!("   # Simple linear function");
            println!("   def demo_function(x):");
            println!("       return x * 2.0 + 1.0");
            println!("   ");
            println!("   # Convert to CoreML");
            println!("   input_features = [ct.TensorType(shape=(1, 5), name='input')]");
            println!("   model = ct.convert(demo_function, inputs=input_features)");
            println!("   model.save('demo.mlmodelc')");
            println!();
            println!("📚 More examples in: candle-coreml/examples/README.md");
        }
    }
    
    // Step 5: Show what CoreML can do
    println!();
    println!("🍎 CoreML Integration Features:");
    println!("   ✓ Device validation (accepts CPU/Metal, rejects CUDA)");
    println!("   ✓ Automatic tensor conversion (Candle ↔ MLMultiArray)");
    println!("   ✓ Multiple input support (pass slice of tensors)");
    println!("   ✓ Error handling with helpful messages");
    println!("   ✓ Apple Neural Engine utilization");
    println!("   ✓ Memory-efficient unified memory usage");
    
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn hello_coreml() -> Result<()> {
    println!("👋 Hello CoreML! (Platform Info)");
    println!("=================================");
    println!();
    println!("❌ CoreML is only available on macOS");
    println!();
    println!("📋 Current platform:");
    println!("   OS: {}", std::env::consts::OS);
    println!("   Architecture: {}", std::env::consts::ARCH);
    println!();
    println!("💡 To use CoreML:");
    println!("   • Run on macOS");
    println!("   • Enable the 'coreml' feature");
    println!("   • Build with: cargo run --example hello_coreml --features coreml");
    println!();
    println!("🔗 CoreML provides:");
    println!("   • Neural Engine acceleration on Apple Silicon");
    println!("   • Unified memory architecture benefits");
    println!("   • Power-efficient inference");
    println!("   • Integration with Apple's ML frameworks");
    
    Ok(())
}

fn main() -> Result<()> {
    hello_coreml()
}