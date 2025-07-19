//! CoreML model wrapper for Candle
//!
//! This module provides a high-level interface for CoreML models that integrates
//! with Candle's tensor system.

use candle_core::{Device, Tensor, Error as CandleError};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Configuration for CoreML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Input tensor name (e.g., "input_ids")
    pub input_name: String,
    /// Output tensor name (e.g., "logits") 
    pub output_name: String,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model architecture name
    pub model_type: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_name: "input_ids".to_string(),
            output_name: "logits".to_string(), 
            max_sequence_length: 128,
            vocab_size: 32000,
            model_type: "coreml".to_string(),
        }
    }
}

#[cfg(target_os = "macos")]
use objc2::rc::{autoreleasepool, Retained};
#[cfg(target_os = "macos")]
use objc2_core_ml::{MLModel, MLMultiArray, MLDictionaryFeatureProvider, MLFeatureProvider};
#[cfg(target_os = "macos")]
use objc2_foundation::{NSString, NSURL};
#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2::AnyThread;
#[cfg(target_os = "macos")]
use block2::StackBlock;

/// CoreML model wrapper that provides Candle tensor integration
pub struct CoreMLModel {
    #[cfg(target_os = "macos")]
    inner: Retained<MLModel>,
    #[cfg(not(target_os = "macos"))]
    _phantom: std::marker::PhantomData<()>,
    config: Config,
}

impl std::fmt::Debug for CoreMLModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMLModel")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CoreMLModel {
    /// Load a CoreML model from a .mlmodelc directory with default configuration
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CandleError> {
        let config = Config::default();
        Self::load_from_file(path, &config)
    }

    /// Load a CoreML model from a .mlmodelc directory following standard Candle patterns
    /// 
    /// Note: Unlike other Candle models, CoreML models are pre-compiled and don't use VarBuilder.
    /// This method provides a Candle-compatible interface while loading from CoreML files.
    pub fn load_from_file<P: AsRef<Path>>(path: P, config: &Config) -> Result<Self, CandleError> {
        #[cfg(target_os = "macos")]
        {
            let path = path.as_ref();
            if !path.exists() {
                return Err(CandleError::Msg(format!(
                    "Model file not found: {}",
                    path.display()
                )));
            }

            autoreleasepool(|_| {
                let url = unsafe {
                    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
                };
                
                match unsafe { MLModel::modelWithContentsOfURL_error(&url) } {
                    Ok(model) => Ok(CoreMLModel { 
                        inner: model,
                        config: config.clone(),
                    }),
                    Err(err) => Err(CandleError::Msg(format!(
                        "Failed to load CoreML model: {:?}",
                        err
                    ))),
                }
            })
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (path, config);
            Err(CandleError::Msg(
                "CoreML is only available on macOS".to_string(),
            ))
        }
    }

    /// Run forward pass through the model
    /// 
    /// Accepts tensors from CPU or Metal devices, rejects CUDA tensors.
    /// Returns output tensor on the same device as the input.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, CandleError> {
        // Validate input device - accept CPU/Metal, reject CUDA
        match input.device() {
            Device::Cpu | Device::Metal(_) => {
                // Valid devices for CoreML
            }
            Device::Cuda(_) => {
                return Err(CandleError::Msg(
                    "CoreML models do not support CUDA tensors. Please move tensor to CPU or Metal device first.".to_string()
                ));
            }
        }

        #[cfg(target_os = "macos")]
        {
            self.forward_impl(input)
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            let _ = input;
            Err(CandleError::Msg(
                "CoreML is only available on macOS".to_string(),
            ))
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    #[cfg(target_os = "macos")]
    fn forward_impl(&self, input: &Tensor) -> Result<Tensor, CandleError> {
        autoreleasepool(|_| {
            // Convert Candle tensor to MLMultiArray
            let ml_array = self.tensor_to_mlmultiarray(input)?;
            
            // Create feature provider with configured input name
            let provider = self.create_feature_provider(&self.config.input_name, &ml_array)?;
            
            // Run prediction
            let prediction = self.run_prediction(&provider)?;
            
            // Extract output with configured output name
            let output_tensor = self.extract_output(&prediction, &self.config.output_name, input.device())?;
            
            Ok(output_tensor)
        })
    }

    #[cfg(target_os = "macos")]
    fn tensor_to_mlmultiarray(&self, tensor: &Tensor) -> Result<Retained<MLMultiArray>, CandleError> {
        use objc2_core_ml::MLMultiArrayDataType;
        use objc2_foundation::{NSArray, NSNumber};
        
        let contiguous_tensor = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()?
        };

        let element_count = tensor.elem_count();
        let dims = tensor.dims();
        let mut shape = Vec::with_capacity(dims.len());
        for &dim in dims {
            shape.push(NSNumber::new_usize(dim));
        }
        let shape_nsarray = NSArray::from_retained_slice(&shape);

        let multi_array_result = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &shape_nsarray,
                MLMultiArrayDataType::Float32,
            )
        };

        match multi_array_result {
            Ok(ml_array) => {
                use std::sync::atomic::{AtomicBool, Ordering};
                let copied = AtomicBool::new(false);

                let flattened_tensor = contiguous_tensor.flatten_all()?;
                let data_vec = flattened_tensor.to_vec1::<f32>()?;

                unsafe {
                    ml_array.getMutableBytesWithHandler(&StackBlock::new(
                        |ptr: std::ptr::NonNull<std::ffi::c_void>, len, _| {
                            let dst = ptr.as_ptr() as *mut f32;
                            let src = data_vec.as_ptr();
                            let copy_elements = element_count.min(len as usize / std::mem::size_of::<f32>());

                            if copy_elements > 0
                                && len as usize >= copy_elements * std::mem::size_of::<f32>()
                            {
                                std::ptr::copy_nonoverlapping(src, dst, copy_elements);
                                copied.store(true, Ordering::Relaxed);
                            }
                        },
                    ));
                }

                if copied.load(Ordering::Relaxed) {
                    Ok(ml_array)
                } else {
                    Err(CandleError::Msg("Failed to copy data to MLMultiArray".to_string()))
                }
            }
            Err(err) => Err(CandleError::Msg(format!(
                "Failed to create MLMultiArray: {:?}",
                err
            ))),
        }
    }

    #[cfg(target_os = "macos")]
    fn create_feature_provider(
        &self,
        input_name: &str,
        input_array: &MLMultiArray,
    ) -> Result<Retained<MLDictionaryFeatureProvider>, CandleError> {
        use objc2_core_ml::MLFeatureValue;
        use objc2_foundation::{NSDictionary, NSString};
        use objc2::runtime::AnyObject;

        autoreleasepool(|_| {
            let key = NSString::from_str(input_name);
            let value = unsafe { MLFeatureValue::featureValueWithMultiArray(input_array) };

            let dict: Retained<NSDictionary<NSString, AnyObject>> =
                NSDictionary::from_slices::<NSString>(&[&*key], &[&*value]);

            unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    dict.as_ref(),
                )
            }
            .map_err(|e| CandleError::Msg(format!("CoreML initWithDictionary_error: {:?}", e)))
        })
    }

    #[cfg(target_os = "macos")]
    fn run_prediction(
        &self,
        provider: &MLDictionaryFeatureProvider,
    ) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>, CandleError> {
        autoreleasepool(|_| unsafe {
            let protocol_provider = ProtocolObject::from_ref(provider);

            self.inner
                .predictionFromFeatures_error(protocol_provider)
                .map_err(|e| CandleError::Msg(format!("CoreML prediction error: {:?}", e)))
        })
    }

    #[cfg(target_os = "macos")]
    fn extract_output(
        &self,
        prediction: &ProtocolObject<dyn MLFeatureProvider>,
        output_name: &str,
        input_device: &Device,
    ) -> Result<Tensor, CandleError> {
        autoreleasepool(|_| unsafe {
            let name = NSString::from_str(output_name);
            let value = prediction
                .featureValueForName(&name)
                .ok_or_else(|| CandleError::Msg(format!("Output '{}' not found", output_name)))?;

            let marray = value.multiArrayValue().ok_or_else(|| {
                CandleError::Msg(format!("Output '{}' is not MLMultiArray", output_name))
            })?;

            let count = marray.count() as usize;
            let mut buf = vec![0.0f32; count];

            use std::cell::RefCell;
            let buf_cell = RefCell::new(&mut buf);

            marray.getBytesWithHandler(&StackBlock::new(
                |ptr: std::ptr::NonNull<std::ffi::c_void>, len: isize| {
                    let src = ptr.as_ptr() as *const f32;
                    let copy_elements = count.min(len as usize / std::mem::size_of::<f32>());
                    if copy_elements > 0 && len as usize >= copy_elements * std::mem::size_of::<f32>() {
                        if let Ok(mut buf_ref) = buf_cell.try_borrow_mut() {
                            std::ptr::copy_nonoverlapping(src, buf_ref.as_mut_ptr(), copy_elements);
                        }
                    }
                },
            ));

            // Get shape from MLMultiArray
            let shape_nsarray = marray.shape();
            let shape_count = shape_nsarray.count();
            let mut shape = Vec::with_capacity(shape_count);
            
            for i in 0..shape_count {
                let dim_number = shape_nsarray.objectAtIndex(i);
                let dim_value = dim_number.integerValue() as usize;
                shape.push(dim_value);
            }

            // Create tensor with the same device as input
            Tensor::from_vec(buf, shape, input_device)
                .map_err(|e| CandleError::Msg(format!("Failed to create output tensor: {}", e)))
        })
    }
}


/// Builder for CoreML models
/// 
/// This provides an interface for loading CoreML models with configuration
/// management and device selection.
pub struct CoreMLModelBuilder {
    config: Config,
    model_filename: PathBuf,
}

impl CoreMLModelBuilder {
    /// Create a new builder with the specified model path and config
    pub fn new<P: AsRef<Path>>(model_path: P, config: Config) -> Self {
        Self {
            config,
            model_filename: model_path.as_ref().to_path_buf(),
        }
    }
    
    /// Load a CoreML model from HuggingFace or local files
    pub fn load_from_hub(
        model_id: &str,
        model_filename: Option<&str>,
        config_filename: Option<&str>,
    ) -> Result<Self, CandleError> {
        use crate::get_local_or_remote_file;
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let api = Api::new().map_err(|e| CandleError::Msg(format!("Failed to create HF API: {}", e)))?;
        let repo = api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string()));

        // Load config
        let config_path = match config_filename {
            Some(filename) => get_local_or_remote_file(filename, &repo)
                .map_err(|e| CandleError::Msg(format!("Failed to get config file: {}", e)))?,
            None => get_local_or_remote_file("config.json", &repo)
                .map_err(|e| CandleError::Msg(format!("Failed to get config.json: {}", e)))?,
        };

        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| CandleError::Msg(format!("Failed to read config file: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| CandleError::Msg(format!("Failed to parse config: {}", e)))?;

        // Get model file
        let model_path = match model_filename {
            Some(filename) => get_local_or_remote_file(filename, &repo)
                .map_err(|e| CandleError::Msg(format!("Failed to get model file: {}", e)))?,
            None => {
                // Try common CoreML model filenames
                for filename in &["model.mlmodelc", "model.mlpackage"] {
                    if let Ok(path) = get_local_or_remote_file(filename, &repo) {
                        return Ok(Self::new(path, config));
                    }
                }
                return Err(CandleError::Msg("No CoreML model file found".to_string()));
            }
        };

        Ok(Self::new(model_path, config))
    }

    /// Build the CoreML model 
    pub fn build_model(&self) -> Result<CoreMLModel, CandleError> {
        CoreMLModel::load_from_file(&self.model_filename, &self.config)
    }

    /// Get the config
    pub fn config(&self) -> &Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    #[cfg(target_os = "macos")]
    fn test_model_creation() {
        // This test requires an actual .mlmodelc file
        // Skip if file doesn't exist
        let model_path = "models/test.mlmodelc";
        if !std::path::Path::new(model_path).exists() {
            return;
        }

        let config = Config::default();
        let device = Device::Cpu;
        
        let model = CoreMLModel::load_from_file(model_path, &config)
            .expect("Failed to load model");
        
        // Test config access
        assert_eq!(model.config().input_name, "input_ids");
        
        // Test with dummy input tensor on CPU device
        let input = Tensor::ones((1, 10), candle_core::DType::F32, &device)
            .expect("Failed to create input tensor");
        
        // This will fail without a real model but tests the interface
        let _result = model.forward(&input);
    }
}

