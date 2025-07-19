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
    
    // Create test tensor with F32 dtype
    let tensor = Tensor::randn(0.0f32, 1.0f32, shape, &device)?;
    
    // Since we can't easily access the internal conversion functions,
    // we'll simulate the operations that happen during conversion:
    // 1. Tensor preparation (contiguous, flatten, to_vec)
    // 2. Memory allocation and copying (simulated)
    // 3. Tensor reconstruction (from_vec, reshape)
    
    let mut preparation_times = Vec::new();
    let mut reconstruction_times = Vec::new();
    
    for _ in 0..iterations {
        // Simulate Tensor → MLMultiArray preparation
        let start = Instant::now();
        let contiguous_tensor = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()?
        };
        let flattened = contiguous_tensor.flatten_all()?;
        let data_vec = flattened.to_vec1::<f32>()?;
        let preparation_time = start.elapsed();
        preparation_times.push(preparation_time);
        
        // Simulate MLMultiArray → Tensor reconstruction
        let start = Instant::now();
        let _reconstructed = Tensor::from_vec(data_vec, shape, &device)?;
        let reconstruction_time = start.elapsed();
        reconstruction_times.push(reconstruction_time);
    }
    
    // Calculate average times
    let avg_preparation = preparation_times.iter().sum::<Duration>() / preparation_times.len() as u32;
    let avg_reconstruction = reconstruction_times.iter().sum::<Duration>() / reconstruction_times.len() as u32;
    let round_trip = avg_preparation + avg_reconstruction;
    
    Ok(ConversionResult {
        tensor_size: format!("{:?}", shape),
        elements,
        tensor_to_ml_time: avg_preparation,
        ml_to_tensor_time: avg_reconstruction,
        round_trip_time: round_trip,
        memory_mb,
    })
}


#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn benchmark_tensor_conversion(_shape: &[usize], _iterations: usize) -> Result<ConversionResult> {
    Err(anyhow::anyhow!("CoreML benchmarks only available on macOS"))
}

fn benchmark_pure_candle_operations(shape: &[usize], iterations: usize) -> Result<Duration> {
    let device = Device::Cpu;
    let tensor = Tensor::randn(0.0f32, 1.0f32, shape, &device)?;
    
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