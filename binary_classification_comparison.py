#!/usr/bin/env python3
"""Binary Classification Comparison Test for GitHub Issue #93.

This test compares binary classification predictions between Python LightGBM
and Rust GBM implementations before and after the hessian calculation fix.
"""

import lightgbm as lgb
import numpy as np
import subprocess
import json
import sys
import os
import tempfile
import shutil
from pathlib import Path

def create_binary_classification_data():
    """Create binary classification test data."""
    # Create reproducible binary classification data
    np.random.seed(42)
    
    # Generate features
    n_samples = 100
    n_features = 3
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create labels with some signal
    linear_combination = features[:, 0] * 0.5 + features[:, 1] * -0.3 + features[:, 2] * 0.2
    probabilities = 1 / (1 + np.exp(-linear_combination))
    labels = (probabilities > 0.5).astype(np.float32)
    
    # Small test set for comparison
    test_features = np.array([
        [0.5, -0.3, 0.2],
        [-0.2, 0.8, -0.1], 
        [1.0, -1.0, 0.5],
        [-0.5, 0.2, -0.8],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    return features, labels, test_features

def train_python_lightgbm_binary(features, labels, test_features):
    """Train Python LightGBM binary classification model."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_iterations': 10,
        'learning_rate': 0.1,
        'num_leaves': 7,
        'min_data_in_leaf': 2,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': 42,
    }
    
    train_data = lgb.Dataset(features, label=labels)
    model = lgb.train(params, train_data)
    
    # Get predictions (probabilities)
    train_predictions = model.predict(features)
    test_predictions = model.predict(test_features)
    
    return {
        'train_predictions': train_predictions.tolist(),
        'test_predictions': test_predictions.tolist(),
        'params': params
    }

def create_rust_binary_test_program(features, labels, test_features):
    """Create a Rust program for binary classification testing."""
    rust_code = f'''//! Binary classification comparison test
use lightgbm_rust::*;
use ndarray::{{Array1, Array2}};
use serde_json;

fn main() -> Result<()> {{
    lightgbm_rust::init()?;
    
    // Training data
    let features = Array2::from_shape_vec(
        ({features.shape[0]}, {features.shape[1]}),
        vec!{features.flatten().tolist()}
    ).expect("Failed to create feature array");
    
    let labels = Array1::from_vec(vec!{labels.tolist()});
    
    // Test data  
    let test_features = Array2::from_shape_vec(
        ({test_features.shape[0]}, {test_features.shape[1]}),
        vec!{test_features.flatten().tolist()}
    ).expect("Failed to create test feature array");
    
    // Create dataset
    let dataset = Dataset::new(features.clone(), labels, None, None, None, None)?;
    
    // Configure for binary classification
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(7)
        .min_data_in_leaf(2)
        .lambda_l2(1.0)
        .build()?;
    
    // Train model
    let mut gbdt = GBDT::new(config, dataset)?;
    gbdt.train()?;
    
    // Get predictions
    let train_predictions = gbdt.predict(&features)?;
    let test_predictions = gbdt.predict(&test_features)?;
    
    let result = serde_json::json!({{
        "train_predictions": train_predictions,
        "test_predictions": test_predictions
    }});
    
    println!("{{}}", serde_json::to_string_pretty(&result).unwrap());
    
    Ok(())
}}'''
    
    return rust_code

def backup_objective_file():
    """Create a backup of the objective.rs file."""
    objective_path = Path("src/config/objective.rs")
    backup_path = Path("/tmp/objective_backup.rs")
    shutil.copy2(objective_path, backup_path)
    return backup_path

def restore_objective_file(backup_path):
    """Restore the objective.rs file from backup."""
    objective_path = Path("src/config/objective.rs")
    shutil.copy2(backup_path, objective_path)

def apply_issue_93_fix():
    """Apply the GitHub Issue #93 fix to objective.rs."""
    objective_path = Path("src/config/objective.rs")
    
    # Read the file
    with open(objective_path, 'r') as f:
        content = f.read()
    
    # Apply the fixes
    # Fix 1: Remove sigmoid from prediction calculation
    content = content.replace(
        "let prob = 1.0 / (1.0 + (-predictions[i] * self.config.sigmoid).exp());",
        "let prob = 1.0 / (1.0 + (-predictions[i]).exp());"
    )
    
    # Fix 2: Remove sigmoid^2 from hessian calculation  
    content = content.replace(
        "hessians[i] = prob * (1.0 - prob) * self.config.sigmoid * self.config.sigmoid;",
        "hessians[i] = prob * (1.0 - prob);"
    )
    
    # Write back
    with open(objective_path, 'w') as f:
        f.write(content)

def build_rust_program():
    """Build the Rust program."""
    try:
        result = subprocess.run(
            ["cargo", "build", "--release"],
            capture_output=True,
            text=True,
            cwd="."
        )
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Build error: {e}")
        return False

def run_rust_binary_test(rust_code):
    """Run the Rust binary classification test."""
    # Write the test program
    test_file = Path("/tmp/binary_test.rs")
    with open(test_file, 'w') as f:
        f.write(rust_code)
    
    # Try to build and run (this might fail if dependencies are missing)
    try:
        # Copy to src/bin for easier compilation
        bin_dir = Path("src/bin")
        bin_dir.mkdir(exist_ok=True)
        shutil.copy2(test_file, bin_dir / "binary_test.rs")
        
        # Build
        if not build_rust_program():
            return None
            
        # Run
        result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "binary_test"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Rust test failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error running Rust test: {e}")
        return None

def calculate_difference(predictions1, predictions2):
    """Calculate numerical difference between two prediction arrays."""
    p1 = np.array(predictions1)
    p2 = np.array(predictions2)
    
    # Calculate various difference metrics
    absolute_diff = np.abs(p1 - p2)
    max_diff = np.max(absolute_diff)
    mean_diff = np.mean(absolute_diff)
    rmse = np.sqrt(np.mean((p1 - p2) ** 2))
    
    return {
        'max_absolute_difference': float(max_diff),
        'mean_absolute_difference': float(mean_diff),
        'rmse': float(rmse),
        'all_differences': absolute_diff.tolist()
    }

def run_comparison_test():
    """Run the complete before/after comparison test."""
    print("🧪 Running Binary Classification Comparison Test for Issue #93")
    print("=" * 70)
    
    # Create test data
    print("📊 Creating binary classification test data...")
    features, labels, test_features = create_binary_classification_data()
    
    # Train Python LightGBM reference
    print("🐍 Training Python LightGBM reference model...")
    python_results = train_python_lightgbm_binary(features, labels, test_features)
    
    print("📈 Python LightGBM Results:")
    print(f"  Training predictions (first 5): {python_results['train_predictions'][:5]}")
    print(f"  Test predictions: {python_results['test_predictions']}")
    
    # Create Rust test program
    rust_code = create_rust_binary_test_program(features, labels, test_features)
    
    # Backup original file (the file should already have our fix applied)
    print("💾 Creating backup of current (fixed) objective.rs...")
    backup_path = backup_objective_file()
    
    try:
        # Test BEFORE fix (restore original buggy version)
        print("⏪ Testing BEFORE fix (restoring buggy version)...")
        
        # Manually restore the buggy version
        objective_path = Path("src/config/objective.rs")
        with open(objective_path, 'r') as f:
            content = f.read()
        
        # Revert to buggy version
        buggy_content = content.replace(
            "let prob = 1.0 / (1.0 + (-predictions[i]).exp());",
            "let prob = 1.0 / (1.0 + (-predictions[i] * self.config.sigmoid).exp());"
        ).replace(
            "hessians[i] = prob * (1.0 - prob);",
            "hessians[i] = prob * (1.0 - prob) * self.config.sigmoid * self.config.sigmoid;"
        )
        
        with open(objective_path, 'w') as f:
            f.write(buggy_content)
        
        rust_results_before = run_rust_binary_test(rust_code)
        
        # Test AFTER fix (restore fixed version)
        print("⏩ Testing AFTER fix (applying corrected version)...")
        restore_objective_file(backup_path)  # This restores the fixed version
        rust_results_after = run_rust_binary_test(rust_code)
        
        # Compare results
        print("\\n📊 COMPARISON RESULTS:")
        print("=" * 50)
        
        if rust_results_before and rust_results_after:
            # Calculate differences for test predictions (most important)
            diff_before = calculate_difference(
                python_results['test_predictions'], 
                rust_results_before['test_predictions']
            )
            
            diff_after = calculate_difference(
                python_results['test_predictions'], 
                rust_results_after['test_predictions']
            )
            
            print("\\n🔍 TEST PREDICTIONS COMPARISON:")
            print(f"  Python LightGBM: {python_results['test_predictions']}")
            print(f"  Rust BEFORE fix: {rust_results_before['test_predictions']}")
            print(f"  Rust AFTER fix:  {rust_results_after['test_predictions']}")
            
            print("\\n📏 NUMERICAL DIFFERENCES vs Python LightGBM:")
            print(f"  BEFORE fix:")
            print(f"    Max difference: {diff_before['max_absolute_difference']:.6e}")
            print(f"    Mean difference: {diff_before['mean_absolute_difference']:.6e}")
            print(f"    RMSE: {diff_before['rmse']:.6e}")
            
            print(f"  AFTER fix:")
            print(f"    Max difference: {diff_after['max_absolute_difference']:.6e}")
            print(f"    Mean difference: {diff_after['mean_absolute_difference']:.6e}")
            print(f"    RMSE: {diff_after['rmse']:.6e}")
            
            # Determine if improvement was achieved
            improvement = diff_before['max_absolute_difference'] > diff_after['max_absolute_difference']
            improvement_ratio = diff_before['max_absolute_difference'] / diff_after['max_absolute_difference'] if diff_after['max_absolute_difference'] > 0 else float('inf')
            
            print("\\n🎯 IMPROVEMENT ANALYSIS:")
            if improvement:
                print(f"  ✅ IMPROVEMENT DETECTED!")
                print(f"  📉 Max difference reduced by factor of {improvement_ratio:.2f}")
                print(f"  📊 Reduction: {diff_before['max_absolute_difference']:.6e} → {diff_after['max_absolute_difference']:.6e}")
                
                # Check if within acceptable tolerance
                target_tolerance = 1e-6
                if diff_after['max_absolute_difference'] <= target_tolerance:
                    print(f"  🎉 Within target tolerance of {target_tolerance:.0e}")
                    return True, diff_before, diff_after
                else:
                    print(f"  ⚠️  Still above target tolerance of {target_tolerance:.0e}")
                    return True, diff_before, diff_after
            else:
                print(f"  ❌ NO IMPROVEMENT DETECTED")
                print(f"  📈 Differences: {diff_before['max_absolute_difference']:.6e} → {diff_after['max_absolute_difference']:.6e}")
                return False, diff_before, diff_after
        else:
            print("❌ Could not run Rust tests - likely missing dependencies or compilation issues")
            print("💡 Ensure your Rust environment is properly set up")
            return None, None, None
            
    finally:
        # Always restore the fixed version
        restore_objective_file(backup_path)
        backup_path.unlink()  # Clean up backup
        
        # Clean up test files
        bin_test_path = Path("src/bin/binary_test.rs")
        if bin_test_path.exists():
            bin_test_path.unlink()

if __name__ == "__main__":
    improved, diff_before, diff_after = run_comparison_test()
    
    if improved is None:
        print("\\n❌ Test could not be completed")
        sys.exit(1)
    elif improved:
        print("\\n✅ Fix successfully improves binary classification accuracy!")
        sys.exit(0)
    else:
        print("\\n❌ Fix does not improve binary classification accuracy")
        sys.exit(1)