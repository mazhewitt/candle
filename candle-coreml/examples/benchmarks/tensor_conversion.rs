//! Tensor Conversion Overhead Benchmark
//! 
//! Measures the cost of converting between Candle tensors and CoreML MLMultiArray.
//! This helps developers understand the overhead of CoreML integration.

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::{Duration, Instant};

struct ConversionResult {
    tensor_size: String,
    elements: usize,
    tensor_to_ml_time: Duration,
    ml_to_tensor_time: Duration,
    round_trip_time: Duration,
    memory_mb: f64,
}

impl ConversionResult {
    fn print(&self) {
        println!("{:<12} {:<10} {:<12} {:<12} {:<12} {:<8}",
            self.tensor_size,
            format!("{:.1}M", self.elements as f64 / 1_000_000.0),
            format!("{:.2?}", self.tensor_to_ml_time),
            format!("{:.2?}", self.ml_to_tensor_time), 
            format!("{:.2?}", self.round_trip_time),
            format!("{:.1}", self.memory_mb)
        );
    }
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn benchmark_tensor_conversion(shape: &[usize], iterations: usize) -> Result<ConversionResult> {
    
    let device = Device::Cpu;
    let elements = shape.iter().product::<usize>();
    let memory_mb = (elements * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);
    
    // Create test tensor
    let tensor = Tensor::randn(0.0, 1.0, shape, &device)?;
    
    // Create a dummy model to access conversion functions
    // We'll access the conversion functions through the model's methods
    // This is a bit of a hack for benchmarking, but it works
    
    let mut tensor_to_ml_times = Vec::new();
    let mut ml_to_tensor_times = Vec::new();
    
    for _ in 0..iterations {
        // Benchmark Tensor → MLMultiArray conversion
        let start = Instant::now();
        let result = test_tensor_to_mlarray(&tensor);
        let tensor_to_ml_time = start.elapsed();
        
        match result {
            Ok(ml_array) => {
                tensor_to_ml_times.push(tensor_to_ml_time);
                
                // Benchmark MLMultiArray → Tensor conversion  
                let start = Instant::now();
                let _result = test_mlarray_to_tensor(&ml_array, &device);
                let ml_to_tensor_time = start.elapsed();
                ml_to_tensor_times.push(ml_to_tensor_time);
            }
            Err(_) => {
                // Skip this iteration if conversion failed
                continue;
            }
        }
    }
    
    if tensor_to_ml_times.is_empty() {
        return Err(anyhow::anyhow!("All conversion attempts failed"));
    }
    
    // Calculate average times
    let avg_tensor_to_ml = tensor_to_ml_times.iter().sum::<Duration>() / tensor_to_ml_times.len() as u32;
    let avg_ml_to_tensor = ml_to_tensor_times.iter().sum::<Duration>() / ml_to_tensor_times.len() as u32;
    let round_trip = avg_tensor_to_ml + avg_ml_to_tensor;
    
    Ok(ConversionResult {
        tensor_size: format!("{:?}", shape),
        elements,
        tensor_to_ml_time: avg_tensor_to_ml,
        ml_to_tensor_time: avg_ml_to_tensor,
        round_trip_time: round_trip,
        memory_mb,
    })
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn test_tensor_to_mlarray(tensor: &Tensor) -> Result<objc2::rc::Retained<objc2_core_ml::MLMultiArray>> {
    use objc2_core_ml::MLMultiArrayDataType;
    use objc2_foundation::{NSArray, NSNumber};
    use objc2::rc::autoreleasepool;
    use objc2::AnyThread;
    use block2::StackBlock;
    
    autoreleasepool(|_| {
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
            objc2_core_ml::MLMultiArray::initWithShape_dataType_error(
                objc2_core_ml::MLMultiArray::alloc(),
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
                    Err(anyhow::anyhow!("Failed to copy data to MLMultiArray"))
                }
            }
            Err(err) => Err(anyhow::anyhow!("Failed to create MLMultiArray: {:?}", err)),
        }
    })
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn test_mlarray_to_tensor(
    ml_array: &objc2_core_ml::MLMultiArray, 
    device: &Device
) -> Result<Tensor> {
    use objc2::rc::autoreleasepool;
    use block2::StackBlock;
    
    autoreleasepool(|_| {
        // Get shape information
        let shape_nsarray = unsafe { ml_array.shape() };
        let shape: Vec<usize> = (0..shape_nsarray.count())
            .map(|i| shape_nsarray.objectAtIndex(i).unsignedIntegerValue())
            .collect();
        
        let element_count: usize = shape.iter().product();
        
        // Extract data
        let mut data = vec![0.0f32; element_count];
        
        use std::cell::RefCell;
        let data_cell = RefCell::new(&mut data);
        
        unsafe {
            ml_array.getBytesWithHandler(&StackBlock::new(
                |ptr: std::ptr::NonNull<std::ffi::c_void>, len: isize| {
                    let src = ptr.as_ptr() as *const f32;
                    let copy_elements = element_count.min(len as usize / std::mem::size_of::<f32>());
                    
                    if copy_elements > 0 && len as usize >= copy_elements * std::mem::size_of::<f32>() {
                        if let Ok(mut data_ref) = data_cell.try_borrow_mut() {
                            std::ptr::copy_nonoverlapping(src, data_ref.as_mut_ptr(), copy_elements);
                        }
                    }
                },
            ));
        }
        
        // Create tensor
        let tensor = Tensor::from_vec(data, &shape[..], device)?;
        Ok(tensor)
    })
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn benchmark_tensor_conversion(_shape: &[usize], _iterations: usize) -> Result<ConversionResult> {
    Err(anyhow::anyhow!("CoreML benchmarks only available on macOS"))
}

fn benchmark_pure_candle_operations(shape: &[usize], iterations: usize) -> Result<Duration> {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0.0, 1.0, shape, &device)?;
    
    let mut times = Vec::new();
    
    for _ in 0..iterations {
        let start = Instant::now();
        
        // Simulate the same operations as conversion: clone, flatten, to_vec, from_vec
        let flattened = tensor.flatten_all()?;
        let data: Vec<f32> = flattened.to_vec1()?;
        let _reconstructed = Tensor::from_vec(data, shape, &device)?;
        
        times.push(start.elapsed());
    }
    
    Ok(times.iter().sum::<Duration>() / times.len() as u32)
}

fn main() -> Result<()> {
    println!("Tensor Conversion Overhead Benchmark");
    println!("===================================");
    println!("Measuring the cost of Candle ↔ CoreML tensor conversion\n");
    
    let test_shapes = vec![
        vec![64],           // 64 elements (256 bytes)
        vec![256, 256],     // 65K elements (256 KB) 
        vec![512, 512],     // 262K elements (1 MB)
        vec![1024, 1024],   // 1M elements (4 MB)
        vec![2048, 1024],   // 2M elements (8 MB) 
        vec![1, 3, 512, 512], // Image-like: 786K elements (3 MB)
        vec![32, 128, 768], // BERT-like: 3M elements (12 MB)
    ];
    
    let iterations = 10;
    
    println!("{:<12} {:<10} {:<12} {:<12} {:<12} {:<8}", 
        "Shape", "Elements", "→ MLArray", "← Tensor", "Round Trip", "Memory");
    println!("{:-<70}", "");
    
    let mut results = Vec::new();
    
    for shape in &test_shapes {
        match benchmark_tensor_conversion(shape, iterations) {
            Ok(result) => {
                result.print();
                results.push(result);
            }
            Err(e) => {
                println!("{:<12} {:<58}", format!("{:?}", shape), format!("❌ Failed: {}", e));
            }
        }
        
        // Compare with pure Candle operations
        if let Ok(pure_candle_time) = benchmark_pure_candle_operations(shape, iterations) {
            println!("{:<12} {:<10} {:<36} {:<8}", 
                "", 
                "",
                format!("Pure Candle equivalent: {:?}", pure_candle_time),
                ""
            );
        }
        println!();
    }
    
    // Analysis
    if !results.is_empty() {
        println!("=== ANALYSIS ===");
        
        let smallest = results.first().unwrap();
        let largest = results.last().unwrap();
        
        println!("Overhead scaling:");
        println!("  Small tensors ({:.1}MB): {:?} round-trip", 
            smallest.memory_mb, smallest.round_trip_time);
        println!("  Large tensors ({:.1}MB): {:?} round-trip", 
            largest.memory_mb, largest.round_trip_time);
        
        let overhead_per_mb = largest.round_trip_time.as_nanos() as f64 / largest.memory_mb;
        println!("  Overhead: ~{:.0}µs per MB", overhead_per_mb / 1000.0);
        
        println!("\nRecommendations:");
        println!("  • Use CoreML for large models where energy efficiency matters");
        println!("  • Consider conversion cost for small, frequent operations"); 
        println!("  • Batch operations when possible to amortize overhead");
    }
    
    Ok(())
}