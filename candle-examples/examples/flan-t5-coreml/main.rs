use anyhow::Result;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

#[cfg(all(target_os = "macos", feature = "coreml"))]
use anyhow::Error as E;
#[cfg(all(target_os = "macos", feature = "coreml"))]
use candle::{Device, IndexOp, Tensor};
#[cfg(all(target_os = "macos", feature = "coreml"))]
use candle_transformers::generation::LogitsProcessor;
#[cfg(all(target_os = "macos", feature = "coreml"))]
use candle_transformers::models::deepseek2::TopKLastDimOp;
#[cfg(all(target_os = "macos", feature = "coreml"))]
use std::io::Write;
#[cfg(all(target_os = "macos", feature = "coreml"))]
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long, default_value = "mazhewitt/flan-t5-base-coreml")]
    model_id: String,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    encoder_model_file: Option<String>,

    #[arg(long)]
    decoder_model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Use the CPU backend.
    #[arg(long)]
    cpu: bool,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 50)]
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

    /// Task prefix for FLAN-T5 (e.g. "translate English to French:")
    #[arg(long, default_value = "summarize:")]
    task_prefix: String,

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

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn compile_coreml_model(model_path: &PathBuf, model_name: &str) -> Result<PathBuf> {
    use std::process::Command;

    // Check if the model is already a compiled .mlmodelc
    if model_path.extension().and_then(|s| s.to_str()) == Some("mlmodelc") {
        return Ok(model_path.clone());
    }

    // Check if this is an .mlpackage that needs compilation
    if model_path.extension().and_then(|s| s.to_str()) == Some("mlpackage") {
        let cache_dir = model_path.parent().unwrap();
        let compiled_model_name = format!("{}.mlmodelc", model_name);
        let compiled_model_path = cache_dir.join("compiled_models").join(&compiled_model_name);

        if !compiled_model_path.exists() {
            println!(
                "Compiling CoreML model {} (this may take a moment)...",
                model_name
            );
            std::fs::create_dir_all(compiled_model_path.parent().unwrap())?;

            let output = Command::new("xcrun")
                .args(&[
                    "coremlc",
                    "compile",
                    &model_path.to_string_lossy(),
                    &compiled_model_path.to_string_lossy()
                ])
                .output()
                .map_err(|e| anyhow::anyhow!("Failed to run coremlc: {}. Make sure Xcode command line tools are installed.", e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow::anyhow!(
                    "CoreML compilation failed for {}: {}",
                    model_name,
                    stderr
                ));
            }

            println!("CoreML model {} compiled successfully", model_name);
        }

        // Check if the compiled model is nested (coremlc sometimes creates nested structure)
        let nested_path = compiled_model_path.join(&compiled_model_name);
        if nested_path.exists() {
            Ok(nested_path)
        } else {
            Ok(compiled_model_path)
        }
    } else {
        // Assume it's already in the right format
        Ok(model_path.clone())
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

    let revision = args.revision.unwrap_or_else(|| "main".to_string());
    let repo = Repo::with_revision(args.model_id.clone(), RepoType::Model, revision);
    let api = Api::new()?;
    let api = api.repo(repo);

    #[cfg(all(target_os = "macos", feature = "coreml"))]
    let device = Device::Cpu;

    // Get encoder model
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    let encoder_model_filename = {
        let raw_path = match &args.encoder_model_file {
            Some(filename) => get_local_or_remote_file(filename, &api)?,
            None => {
                println!("Searching for FLAN-T5 encoder model...");
                api.get("flan_t5_base_encoder_quality.mlpackage")
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "Encoder model not found in repository {}. Error: {}\n\
                         You can download the models using:\n\
                         huggingface-cli download {} --local-dir .\n\
                         Or specify the model file with --encoder-model-file <path>",
                            args.model_id,
                            e,
                            args.model_id
                        )
                    })?
            }
        };
        compile_coreml_model(&raw_path, "flan_t5_base_encoder_quality")?
    };

    // Get decoder model
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    let decoder_model_filename = {
        let raw_path = match &args.decoder_model_file {
            Some(filename) => get_local_or_remote_file(filename, &api)?,
            None => {
                println!("Searching for FLAN-T5 decoder model...");
                api.get("flan_t5_base_decoder_quality.mlpackage")
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "Decoder model not found in repository {}. Error: {}\n\
                         You can download the models using:\n\
                         huggingface-cli download {} --local-dir .\n\
                         Or specify the model file with --decoder-model-file <path>",
                            args.model_id,
                            e,
                            args.model_id
                        )
                    })?
            }
        };
        compile_coreml_model(&raw_path, "flan_t5_base_decoder_quality")?
    };

    // Get tokenizer
    let tokenizer_filename = match &args.tokenizer_file {
        Some(filename) => get_local_or_remote_file(filename, &api)?,
        None => {
            println!("Looking for tokenizer in model repository...");
            // Try to find tokenizer in the same repository as the models first
            let tokenizer_patterns = ["tokenizer.json", "spiece.model", "tokenizer_config.json"];
            let mut found_tokenizer = None;

            for pattern in &tokenizer_patterns {
                if let Ok(file) = api.get(pattern) {
                    println!("Found tokenizer {} in model repository", pattern);
                    found_tokenizer = Some(file);
                    break;
                }
            }

            if found_tokenizer.is_none() {
                println!("Tokenizer not found in model repo, trying fallback repositories...");

                // Fallback to other FLAN-T5 repositories
                let flan_t5_repos = ["google/flan-t5-base", "google/flan-t5-small"];

                for repo_name in &flan_t5_repos {
                    let api_instance = Api::new()?;
                    let tokenizer_repo = api_instance.repo(Repo::with_revision(
                        repo_name.to_string(),
                        RepoType::Model,
                        "main".to_string(),
                    ));

                    for tokenizer_file in &tokenizer_patterns {
                        if let Ok(file) = get_local_or_remote_file(tokenizer_file, &tokenizer_repo)
                        {
                            println!(
                                "Found fallback tokenizer {} from {}",
                                tokenizer_file, repo_name
                            );
                            found_tokenizer = Some(file);
                            break;
                        }
                    }
                    if found_tokenizer.is_some() {
                        break;
                    }
                }
            }

            found_tokenizer.ok_or_else(|| {
                anyhow::anyhow!(
                    "No tokenizer found for model {}. Try specifying --tokenizer-file explicitly.",
                    args.model_id
                )
            })?
        }
    };

    println!("Retrieved the files in {:?}", start.elapsed());

    #[cfg(all(target_os = "macos", feature = "coreml"))]
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    #[cfg(all(target_os = "macos", feature = "coreml"))]
    let start = std::time::Instant::now();

    #[cfg(all(target_os = "macos", feature = "coreml"))]
    {
        // Create CoreML configuration for encoder
        // Based on HuggingFace model specs: input_ids & attention_mask -> hidden_states
        let encoder_config = candle_coreml::Config {
            input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
            output_name: "hidden_states".to_string(), // float32, shape: [1, 512, 768]
            max_sequence_length: 512,
            vocab_size: 32128, // FLAN-T5 vocab size
            model_type: "flan-t5-base-encoder".to_string(),
        };

        // Create CoreML configuration for decoder
        // Based on HuggingFace model specs: decoder_input_ids, encoder_hidden_states, decoder_attention_mask, encoder_attention_mask -> logits
        let decoder_config = candle_coreml::Config {
            input_names: vec![
                "decoder_input_ids".to_string(),
                "encoder_hidden_states".to_string(),
                "decoder_attention_mask".to_string(),
                "encoder_attention_mask".to_string(),
            ],
            output_name: "logits".to_string(), // float32, shape: [1, 512, 32128]
            max_sequence_length: 512,
            vocab_size: 32128, // FLAN-T5 vocab size
            model_type: "flan-t5-base-decoder".to_string(),
        };

        let encoder_model =
            candle_coreml::CoreMLModel::load_from_file(&encoder_model_filename, &encoder_config)?;
        let decoder_model =
            candle_coreml::CoreMLModel::load_from_file(&decoder_model_filename, &decoder_config)?;

        println!("Loaded encoder and decoder models in {:?}", start.elapsed());

        // Prepare the full prompt with task prefix
        let full_prompt = format!("{} {}", args.task_prefix, args.prompt);
        print!("Input: {}\nOutput: ", full_prompt);
        std::io::stdout().flush()?;

        // Tokenize the input prompt
        let input_tokens = tokenizer
            .encode(full_prompt.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        println!(
            "Input tokens: {:?}",
            &input_tokens[..input_tokens.len().min(10)]
        );

        // Pad/truncate input to max sequence length
        let max_input_len = encoder_config.max_sequence_length;
        let pad_token_id = 0u32;
        let mut padded_input_tokens = vec![pad_token_id; max_input_len];
        let actual_input_len = input_tokens.len().min(max_input_len);
        for (i, &token) in input_tokens[..actual_input_len].iter().enumerate() {
            padded_input_tokens[i] = token;
        }

        // Create attention mask (1 for real tokens, 0 for padding)
        let mut attention_mask = vec![0.0f32; max_input_len];
        for i in 0..actual_input_len {
            attention_mask[i] = 1.0;
        }

        // Convert to f32 tensors for CoreML encoder
        let input_ids = Tensor::from_vec(padded_input_tokens, (1, max_input_len), &device)?
            .to_dtype(candle::DType::F32)?;
        let attention_mask_tensor = Tensor::from_vec(attention_mask, (1, max_input_len), &device)?;

        // Run encoder
        let encoder_outputs = encoder_model.forward(&[&input_ids, &attention_mask_tensor])?;
        let encoder_hidden_states = &encoder_outputs;

        // Initialize decoder with start token
        // For T5, the decoder start token should be the pad token (0)
        // But let's also try with an empty sequence first
        let mut output_token_ids = vec![0u32];

        // Setup temperature and LogitsProcessor
        let temperature = if args.temperature <= 0. {
            None
        } else {
            Some(args.temperature)
        };
        let mut logits_processor = LogitsProcessor::new(args.seed, temperature, args.top_p);
        let start = std::time::Instant::now();

        // Generation loop
        for _index in 0..args.sample_len {
            let max_decoder_len = decoder_config.max_sequence_length;
            if output_token_ids.len() >= max_decoder_len {
                break;
            }

            // Pad decoder input tokens
            let mut padded_decoder_tokens = vec![pad_token_id; max_decoder_len];
            let actual_decoder_len = output_token_ids.len().min(max_decoder_len);
            for (i, &token) in output_token_ids[..actual_decoder_len].iter().enumerate() {
                padded_decoder_tokens[i] = token;
            }

            // Create decoder attention mask
            let mut decoder_attention_mask = vec![0.0f32; max_decoder_len];
            for i in 0..actual_decoder_len {
                decoder_attention_mask[i] = 1.0;
            }

            // Convert to f32 tensors for CoreML decoder
            let decoder_input_ids =
                Tensor::from_vec(padded_decoder_tokens, (1, max_decoder_len), &device)?
                    .to_dtype(candle::DType::F32)?;
            let decoder_attention_mask_tensor =
                Tensor::from_vec(decoder_attention_mask, (1, max_decoder_len), &device)?;

            // Run decoder with inputs in HuggingFace model order: decoder_input_ids, encoder_hidden_states, decoder_attention_mask, encoder_attention_mask
            let logits = decoder_model
                .forward(&[
                    &decoder_input_ids,
                    encoder_hidden_states,
                    &decoder_attention_mask_tensor,
                    &attention_mask_tensor,
                ])?
                .squeeze(0)?;

            // Get logits for the next token position
            // For T5, we want the logits from the last non-padding position
            println!("Logits shape: {:?}", logits.shape());
            let current_length = output_token_ids.len();
            println!("Output token length: {}", current_length);

            // Use the last position in the current sequence for next token prediction
            let next_position = (current_length - 1).min(max_decoder_len - 1);
            println!("Using position {} for generation", next_position);
            let next_token_logits = logits.i(next_position)?;

            // Apply repeat penalty
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

            // Debug: Show top 5 most likely tokens
            let top_k_output = next_token_logits.topk(5)?;
            println!("Top 5 tokens:");
            for i in 0..5 {
                let token_id = top_k_output.indices.get(i)?.to_scalar::<u32>()?;
                let prob = top_k_output.values.get(i)?.to_scalar::<f32>()?;
                let token_text = tokenizer.id_to_token(token_id).unwrap_or("UNK".to_string());
                println!("  {}: {} ({:.3}) -> {}", i + 1, token_id, prob, token_text);
            }

            // Sample next token using LogitsProcessor
            let next_token_id = logits_processor.sample(&next_token_logits)?;

            // Check for EOS token (T5 uses 1 for EOS) or pad token
            if next_token_id == 1 || next_token_id == 0 {
                break;
            }

            println!(
                "Selected token: {} -> {:?}",
                next_token_id,
                tokenizer.id_to_token(next_token_id)
            );

            output_token_ids.push(next_token_id);

            // Decode and print token
            if let Some(text) = tokenizer.id_to_token(next_token_id) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                print!("{text}");
                std::io::stdout().flush()?;
            }
        }

        let dt = start.elapsed();
        println!(
            "\n{} tokens generated ({:.2} token/s)",
            output_token_ids.len() - 1, // -1 because we started with decoder start token
            (output_token_ids.len() - 1) as f64 / dt.as_secs_f64(),
        );
    }

    #[cfg(not(all(target_os = "macos", feature = "coreml")))]
    {
        let _ = tokenizer_filename; // Suppress unused variable warning
        println!("CoreML is only supported on macOS with the coreml feature enabled.");
        println!("Task: {}", args.task_prefix);
        println!("Prompt: {}", args.prompt);
        println!(
            "This would run FLAN-T5 CoreML inference if compiled on macOS with --features coreml"
        );
    }

    Ok(())
}
