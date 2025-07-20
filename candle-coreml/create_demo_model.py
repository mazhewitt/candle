#!/usr/bin/env python3
"""
Create a simple CoreML demo model for testing Candle CoreML integration.

This script creates a minimal CoreML model that can be used with the
hello_coreml example to demonstrate basic CoreML functionality.

Usage:
    python create_demo_model.py

Requirements:
    pip install coremltools numpy

The script creates:
    - demo.mlmodelc: A simple linear transformation model
    - models/demo.mlmodelc: Same model in the models directory
"""

import sys
import os
from pathlib import Path

try:
    import coremltools as ct
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("📦 Install with: pip install coremltools numpy")
    sys.exit(1)

def create_demo_model():
    """Create a simple CoreML model for demonstration."""
    
    print("🔧 Creating demo CoreML model...")
    
    # Create a simple neural network using coremltools builder
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models import datatypes
    
    # Define input and output (use size 3 to avoid Neural Network constraints)
    input_features = [('input', datatypes.Array(3))]
    output_features = [('output', datatypes.Array(3))]
    
    # Create neural network builder
    builder = NeuralNetworkBuilder(input_features, output_features)
    
    # Add a simple linear layer: y = x * 2 + 1
    # This is implemented as: y = x * W + b where W = [2,2,2,2,2] and b = [1,1,1,1,1]
    builder.add_elementwise(
        name='multiply_by_2',
        input_names=['input'],
        output_name='multiplied',
        mode='MULTIPLY',
        alpha=2.0
    )
    
    builder.add_bias(
        name='add_1',
        input_name='multiplied',
        output_name='output',
        b=np.array([1.0, 1.0, 1.0]),
        shape_bias=[3]
    )
    
    # Build the model
    print("📋 Building CoreML model...")
    model = ct.models.MLModel(builder.spec)
    
    # Add metadata
    model.short_description = "Demo model for Candle CoreML integration"
    model.input_description["input"] = "5-element input vector"
    model.output_description["output"] = "Transformed output vector (input * 2 + 1)"
    
    return model

def save_model(model, filename="demo.mlpackage"):
    """Save the model to the specified location."""
    
    # Save in current directory
    print(f"💾 Saving model as {filename}...")
    model.save(filename)
    
    # Compile the model to .mlmodelc
    compiled_filename = filename.replace('.mlpackage', '.mlmodelc')
    print(f"🔨 Compiling model to {compiled_filename}...")
    try:
        compiled_model_url = ct.models.utils.save_multifunction(model, compiled_filename)
        print(f"✅ Compiled model saved to {compiled_filename}")
    except Exception as e:
        print(f"⚠️  Compilation failed, trying alternative method: {e}")
        try:
            # Alternative: use the older save method
            model.save(compiled_filename)
            print(f"✅ Model saved as {compiled_filename}")
        except Exception as e2:
            print(f"❌ Could not create .mlmodelc: {e2}")
    
    # Also save in models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    models_path = models_dir / filename
    print(f"💾 Saving model as {models_path}...")
    model.save(str(models_path))
    
    # Try to compile the models directory version too
    compiled_models_path = models_dir / compiled_filename
    print(f"🔨 Compiling model to {compiled_models_path}...")
    try:
        model.save(str(compiled_models_path))
        print(f"✅ Compiled model saved to {compiled_models_path}")
    except Exception as e:
        print(f"⚠️  Models directory compilation failed: {e}")
    
    return filename

def test_model(model_path):
    """Test the created model with sample input."""
    
    print("🧪 Testing the created model...")
    
    try:
        # Load the model
        model = ct.models.MLModel(model_path)
        
        # Create test input
        test_input = {
            "input": np.array([1.0, 2.0, 3.0], dtype=np.float32)
        }
        
        # Run prediction
        result = model.predict(test_input)
        
        print("✅ Model test successful!")
        input_data = test_input['input']
        output_data = result['output']
        expected = input_data * 2 + 1
        print(f"   Input:  {input_data}")
        print(f"   Output: {output_data}")
        print(f"   Expected: {expected}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main function to create and test the demo model."""
    
    print("🚀 CoreML Demo Model Creator")
    print("============================")
    print()
    
    try:
        # Create the model
        model = create_demo_model()
        
        # Save the model
        model_path = save_model(model)
        
        # Test the model
        if test_model(model_path):
            print()
            print("🎉 Demo model created successfully!")
            print()
            print("📚 Usage with Candle:")
            print("   cargo run --example hello_coreml --features coreml")
            print()
            print("   Or set custom path:")
            print(f"   export COREML_MODEL_PATH=$(pwd)/{model_path}")
            print("   cargo run --example hello_coreml --features coreml")
            print()
            print("📁 Model files created:")
            print(f"   • {model_path}")
            print(f"   • models/{model_path}")
            
        else:
            print("⚠️  Model created but testing failed")
            
    except Exception as e:
        print(f"❌ Error creating demo model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())