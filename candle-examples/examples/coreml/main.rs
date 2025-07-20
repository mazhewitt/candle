
use anyhow::{Error as E, Result};
use candle::Device;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Use the CPU backend.
    #[arg(long)]
    cpu: bool,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10)]
    sample_len: usize,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    prompt: String,
}

fn get_local_or_remote_file(filename: &str, api: &hf_hub::api::sync::ApiRepo) -> Result<PathBuf> {
    let local_filename = PathBuf::from(filename);
    if local_filename.exists() {
        Ok(local_filename)
    } else {
        Ok(api.get(filename)?)
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let start = std::time::Instant::now();

    let device = Device::Cpu;

    let default_model = "corenet-community/coreml-OpenELM-450M-Instruct".to_string();
    let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, "main".to_string()),
    };

    let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
    let api = Api::new()?;
    let api = api.repo(repo);

    let model_filename = match &args.model_file {
        Some(filename) => PathBuf::from(filename),
        None => {
            // Try to find available CoreML models in the repository
            // Common patterns for CoreML models
            let model_patterns = [
                "OpenELM-450M-Instruct-128-float32.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                "model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                "model.mlmodel",
            ];
            
            let mut found_model_file = None;
            let mut mlpackage_name = None;
            
            println!("Searching for CoreML model files...");
            
            for pattern in &model_patterns {
                if let Ok(model_file) = api.get(pattern) {
                    // Extract the .mlpackage name from the path
                    let path_components: Vec<&str> = pattern.split('/').collect();
                    if let Some(package_component) = path_components.first() {
                        if package_component.ends_with(".mlpackage") {
                            mlpackage_name = Some(package_component.trim_end_matches(".mlpackage"));
                        }
                    }
                    
                    // Also download associated files for .mlpackage models
                    if pattern.contains(".mlpackage") {
                        let base_path = pattern.replace("/Data/com.apple.CoreML/model.mlmodel", "");
                        let _ = api.get(&format!("{}/Data/com.apple.CoreML/weights/weight.bin", base_path));
                        let _ = api.get(&format!("{}/Manifest.json", base_path));
                    }
                    
                    found_model_file = Some(model_file);
                    break;
                }
            }
            
            let model_file = found_model_file.ok_or_else(|| {
                anyhow::anyhow!("No CoreML model found in repository {}", model_id)
            })?;
            
            // If this is an .mlmodel file, we need to compile it
            if model_file.extension().and_then(|s| s.to_str()) == Some("mlmodel") {
                // Determine the output name for compilation
                let model_name = mlpackage_name.unwrap_or("model");
                let compiled_model_name = format!("{}.mlmodelc", model_name);
                
                // The model file is in a .mlpackage structure, find the .mlpackage directory
                let mlpackage_dir = if model_file.to_string_lossy().contains(".mlpackage") {
                    model_file.parent().unwrap().parent().unwrap().parent().unwrap()
                } else {
                    model_file.parent().unwrap()
                };
                
                let cache_dir = mlpackage_dir.parent().unwrap();
                let compiled_model_path = cache_dir.join("compiled_models").join(&compiled_model_name);
                
                if !compiled_model_path.exists() {
                    println!("Compiling CoreML model (this may take a moment)...");
                    std::fs::create_dir_all(compiled_model_path.parent().unwrap())?;
                    
                    // Use system coremlc to compile the .mlpackage directory or .mlmodel file
                    let source_path = if mlpackage_dir.extension().and_then(|s| s.to_str()) == Some("mlpackage") {
                        mlpackage_dir
                    } else {
                        &model_file
                    };
                    
                    let output = std::process::Command::new("xcrun")
                        .args(&[
                            "coremlc", 
                            "compile", 
                            &source_path.to_string_lossy(),
                            &compiled_model_path.to_string_lossy()
                        ])
                        .output()
                        .map_err(|e| anyhow::anyhow!("Failed to run coremlc: {}. Make sure Xcode command line tools are installed.", e))?;
                    
                    if !output.status.success() {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        return Err(anyhow::anyhow!("CoreML compilation failed: {}", stderr));
                    }
                    
                    println!("CoreML model compiled successfully");
                }
                
                // Return path to the actual compiled model directory (may be nested)
                if compiled_model_path.join(&compiled_model_name).exists() {
                    compiled_model_path.join(&compiled_model_name)
                } else {
                    compiled_model_path
                }
            } else {
                // Already a compiled model
                model_file
            }
        }
    };

    let tokenizer_filename = match &args.tokenizer_file {
        Some(filename) => get_local_or_remote_file(filename, &api)?,
        None => {
            // Try multiple tokenizer sources following Candle patterns
            let tokenizer_patterns = [
                ("tokenizer.json", &api),
                ("tokenizer_config.json", &api),
            ];
            
            let mut found_tokenizer = None;
            
            // Try the main repository first
            for (pattern, repo_api) in &tokenizer_patterns {
                if let Ok(file) = repo_api.get(pattern) {
                    found_tokenizer = Some(file);
                    break;
                }
            }
            
            // If not found, try common alternative repositories
            if found_tokenizer.is_none() {
                println!("Tokenizer not found in primary repo, trying alternative sources...");
                
                // Extract base model name for alternative repo lookup
                let base_model_name = model_id
                    .strip_prefix("corenet-community/coreml-")
                    .unwrap_or(&model_id)
                    .to_string();
                
                let alternative_repos = [
                    format!("mlx-community/{}", base_model_name),
                    format!("microsoft/{}", base_model_name),
                    format!("{}", base_model_name), // Try the base name directly
                ];
                
                for alt_repo in &alternative_repos {
                    let api_instance = Api::new()?;
                    let tokenizer_repo = api_instance.repo(Repo::with_revision(
                        alt_repo.clone(),
                        RepoType::Model, 
                        "main".to_string()
                    ));
                    
                    if let Ok(file) = get_local_or_remote_file("tokenizer.json", &tokenizer_repo) {
                        println!("Found tokenizer in repository: {}", alt_repo);
                        found_tokenizer = Some(file);
                        break;
                    }
                }
            }
            
            found_tokenizer.ok_or_else(|| {
                anyhow::anyhow!("No tokenizer found for model {}. Try specifying --tokenizer-file explicitly.", model_id)
            })?
        },
    };

    println!("Retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    {
        // Create CoreML configuration
        let config = candle_coreml::Config {
            input_names: vec!["input_ids".to_string()],
            output_name: "logits".to_string(),
            max_sequence_length: 128,
            vocab_size: 32000,
            model_type: "OpenELM-450M-Instruct".to_string(),
        };
        
        let model = candle_coreml::CoreMLModel::load_from_file(&model_filename, &config)?;
        println!("Loaded the model in {:?}", start.elapsed());

        // T5-style text generation with LogitsProcessor
        print!("{}", args.prompt);
        std::io::stdout().flush()?;

        // Tokenize the prompt
        let tokens = tokenizer
            .encode(args.prompt.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // Initialize generation with prompt tokens
        let mut output_token_ids = tokens.clone();
        
        // Setup temperature and LogitsProcessor (following T5 pattern)
        let temperature = if args.temperature <= 0. {
            None
        } else {
            Some(args.temperature)
        };
        let mut logits_processor = LogitsProcessor::new(args.seed, temperature, args.top_p);
        let start = std::time::Instant::now();

        // Generation loop (T5-style)
        for _index in 0..args.sample_len {
            if output_token_ids.len() > config.max_sequence_length {
                break;
            }

            // Create input tensor with proper f32 conversion for CoreML
            let max_seq_len = config.max_sequence_length;
            let pad_token_id = 0u32;
            
            // Pad/truncate to fixed sequence length
            let mut padded_tokens = vec![pad_token_id; max_seq_len];
            let actual_len = output_token_ids.len().min(max_seq_len);
            for (i, &token) in output_token_ids[..actual_len].iter().enumerate() {
                padded_tokens[i] = token;
            }
            
            // Convert to f32 tensor for CoreML
            let input_ids = candle::Tensor::from_vec(padded_tokens, (1, max_seq_len), &device)?
                .to_dtype(candle::DType::F32)?;
            
            // Run inference
            let logits = model.forward(&[&input_ids])?.squeeze(0)?;
            
            // Get logits for next token position
            let next_token_logits = logits.get(output_token_ids.len().min(max_seq_len) - 1)?;
            
            // Apply repeat penalty (following T5 pattern)
            let next_token_logits = if args.repeat_penalty == 1. {
                next_token_logits
            } else {
                let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &next_token_logits,
                    args.repeat_penalty,
                    &output_token_ids[start_at..],
                )?
            };

            // Sample next token using LogitsProcessor
            let next_token_id = logits_processor.sample(&next_token_logits)?;
            
            // Check for EOS or pad token (simple stopping condition)
            if next_token_id == 0 {
                break;
            }
            
            output_token_ids.push(next_token_id);
            
            // Decode and print token (T5 pattern)
            if let Some(text) = tokenizer.id_to_token(next_token_id) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                print!("{text}");
                std::io::stdout().flush()?;
            }
        }
        
        let dt = start.elapsed();
        println!(
            "\n{} tokens generated ({:.2} token/s)",
            output_token_ids.len() - tokens.len(),
            (output_token_ids.len() - tokens.len()) as f64 / dt.as_secs_f64(),
        );
    }
    
    #[cfg(not(all(target_os = "macos", feature = "coreml")))]
    {
        println!("CoreML is only supported on macOS with the coreml feature enabled.");
        println!("Prompt: {}", args.prompt);
        println!("This would run CoreML inference if compiled on macOS with --features coreml");
    }

    Ok(())
}