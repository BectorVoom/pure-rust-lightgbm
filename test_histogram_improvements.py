#!/usr/bin/env python3
"""
Test script to verify whether histogram-based split finding improvements
bring the Rust implementation closer to Python LightGBM results.

This script:
1. Runs the current implementation ("before") 
2. Allows testing after modifications ("after")
3. Uses the existing Python comparison test
4. Measures numerical differences
5. Reports whether changes improve accuracy
"""

import os
import sys
import json
import subprocess
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add the examples directory to path so we can import comparison_test
examples_dir = Path(__file__).parent / "examples" / "python-lightgbm"
sys.path.append(str(examples_dir))

try:
    from comparison_test import create_test_datasets, train_python_lightgbm
except ImportError:
    print("ERROR: Could not import comparison_test.py")
    print(f"Expected path: {examples_dir / 'comparison_test.py'}")
    print("Please ensure the comparison_test.py file exists.")
    sys.exit(1)

class HistogramImprovementTester:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.temp_dir = Path(tempfile.mkdtemp())
        self.datasets = create_test_datasets()
        self.python_results = {}
        self.before_results = {}
        self.after_results = {}
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def setup_python_baseline(self):
        """Generate Python LightGBM baseline results."""
        print("🐍 Generating Python LightGBM baseline results...")
        
        for dataset_name, (features, labels) in self.datasets.items():
            print(f"  Processing {dataset_name} dataset...")
            self.python_results[dataset_name] = train_python_lightgbm(features, labels)
            
        print(f"✅ Generated baseline for {len(self.python_results)} datasets")
        return self.python_results
    
    def create_rust_test_executable(self, dataset_files):
        """Create a standalone Rust test executable."""
        # Create a minimal Rust project in temp directory
        test_project = self.temp_dir / "histogram_test"
        test_project.mkdir()
        
        # Create Cargo.toml
        cargo_toml = f'''[package]
name = "histogram_test"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "test_runner"
path = "src/main.rs"

[dependencies]
ndarray = "0.15"
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
lightgbm-rust = {{ path = "{self.project_root}" }}
'''
        
        (test_project / "Cargo.toml").write_text(cargo_toml)
        
        # Create src directory
        src_dir = test_project / "src"
        src_dir.mkdir()
        
        # Create main.rs test runner
        main_rs = '''//! Test runner for histogram improvements
use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use serde_json;
use std::fs;
use std::collections::HashMap;
use std::env;

#[derive(serde::Deserialize)]
struct TestDataset {
    features: Vec<Vec<f32>>,
    labels: Vec<f32>,
    name: String,
}

#[derive(serde::Serialize)]
struct TestResult {
    training_predictions: Vec<f64>,
    test_predictions: Vec<f64>,
    model_info: HashMap<String, serde_json::Value>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <dataset_json_files...>", args[0]);
        std::process::exit(1);
    }
    
    let mut results = HashMap::new();
    
    for dataset_file in &args[1..] {
        let dataset_json = fs::read_to_string(dataset_file)?;
        let dataset: TestDataset = serde_json::from_str(&dataset_json)?;
        
        let features = Array2::from_shape_vec(
            (dataset.features.len(), dataset.features[0].len()),
            dataset.features.into_iter().flatten().collect::<Vec<f32>>().into_iter().map(|x| x as f64).collect()
        )?;
        
        let labels = Array1::from_vec(
            dataset.labels.into_iter().map(|x| x as f64).collect()
        );
        
        // Create dataset - using None for optional parameters for now
        let dataset_obj = Dataset::new(features.clone(), labels.clone(), None, None, None, None)?;
        
        // Configure model with same parameters as Python
        let config = ConfigBuilder::new()
            .objective(ObjectiveType::Regression)
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(7)
            .min_data_in_leaf(2)
            .lambda_l2(1.0)
            .build()?;
        
        // Train model
        let mut gbdt = GBDT::new(config, dataset_obj)?;
        gbdt.train()?;
        
        // Get predictions on training data
        let training_predictions = gbdt.predict(&features)?;
        
        // Test on specific test cases (same as Python)
        let test_features = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.5, 2.5, 1.0,   // Test case 1
                2.0, 1.0, 0.5,   // Test case 2  
                0.5, 3.5, 2.0,   // Test case 3
            ]
        )?;
        
        let test_predictions = gbdt.predict(&test_features)?;
        
        let mut model_info = HashMap::new();
        model_info.insert("num_trees".to_string(), serde_json::Value::Number(serde_json::Number::from(gbdt.num_iterations())));
        model_info.insert("dataset_name".to_string(), serde_json::Value::String(dataset.name.clone()));
        
        let result = TestResult {
            training_predictions: training_predictions.to_vec(),
            test_predictions: test_predictions.to_vec(),
            model_info,
        };
        
        results.insert(dataset.name, result);
    }
    
    // Output all results as JSON
    println!("{}", serde_json::to_string_pretty(&results)?);
    
    Ok(())
}
'''
        
        (src_dir / "main.rs").write_text(main_rs)
        
        return test_project
    
    def save_datasets_for_rust(self):
        """Save datasets in JSON format for Rust consumption."""
        dataset_files = []
        
        for dataset_name, (features, labels) in self.datasets.items():
            data = {
                'features': features.tolist(),
                'labels': labels.tolist(),
                'name': dataset_name
            }
            
            filename = self.temp_dir / f'dataset_{dataset_name}.json'
            with open(filename, 'w') as f:
                json.dump(data, f)
                
            dataset_files.append(str(filename))
        
        return dataset_files
    
    def run_rust_implementation(self, test_project, dataset_files):
        """Run the Rust implementation and return results."""
        print("🦀 Building and running Rust implementation...")
        
        # Build the test project
        build_result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=test_project,
            capture_output=True,
            text=True
        )
        
        if build_result.returncode != 0:
            print(f"❌ Build failed:")
            print(f"STDOUT: {build_result.stdout}")
            print(f"STDERR: {build_result.stderr}")
            raise RuntimeError("Failed to build Rust test project")
        
        # Run the test executable
        run_result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "test_runner"] + dataset_files,
            cwd=test_project,
            capture_output=True,
            text=True
        )
        
        if run_result.returncode != 0:
            print(f"❌ Run failed:")
            print(f"STDOUT: {run_result.stdout}")
            print(f"STDERR: {run_result.stderr}")
            raise RuntimeError("Failed to run Rust test project")
        
        # Parse JSON output
        try:
            return json.loads(run_result.stdout)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse output as JSON: {e}")
            print(f"Raw output: {run_result.stdout}")
            raise
    
    def measure_differences(self, rust_results, label):
        """Measure numerical differences between Rust and Python results."""
        differences = {}
        
        for dataset_name in self.datasets.keys():
            if dataset_name not in rust_results:
                differences[dataset_name] = {
                    'error': f'Missing results for {dataset_name}',
                    'training_mse': float('inf'),
                    'test_mse': float('inf')
                }
                continue
                
            python_result = self.python_results[dataset_name]
            rust_result = rust_results[dataset_name]
            
            # Calculate MSE for training predictions
            python_train = np.array(python_result['training_predictions'])
            rust_train = np.array(rust_result['training_predictions'])
            training_mse = np.mean((python_train - rust_train) ** 2)
            
            # Calculate MSE for test predictions  
            python_test = np.array(python_result['test_predictions'])
            rust_test = np.array(rust_result['test_predictions'])
            test_mse = np.mean((python_test - rust_test) ** 2)
            
            # Calculate maximum absolute differences
            training_max_abs_diff = np.max(np.abs(python_train - rust_train))
            test_max_abs_diff = np.max(np.abs(python_test - rust_test))
            
            differences[dataset_name] = {
                'training_mse': float(training_mse),
                'test_mse': float(test_mse),
                'training_max_abs_diff': float(training_max_abs_diff),
                'test_max_abs_diff': float(test_max_abs_diff),
                'python_train_sample': python_train[:3].tolist(),
                'rust_train_sample': rust_train[:3].tolist(),
                'python_test': python_test.tolist(),
                'rust_test': rust_test.tolist()
            }
            
        return differences
    
    def run_before_test(self):
        """Run test on current implementation (before modifications)."""
        print("\n📊 BEFORE: Testing current implementation...")
        
        dataset_files = self.save_datasets_for_rust()
        test_project = self.create_rust_test_executable(dataset_files)
        
        try:
            self.before_results = self.run_rust_implementation(test_project, dataset_files)
            before_diffs = self.measure_differences(self.before_results, "BEFORE")
            
            print("✅ Before test completed")
            return before_diffs
            
        except Exception as e:
            print(f"❌ Before test failed: {e}")
            return None
    
    def run_after_test(self):
        """Run test after modifications have been made."""
        print("\n📊 AFTER: Testing modified implementation...")
        
        dataset_files = self.save_datasets_for_rust()
        test_project = self.create_rust_test_executable(dataset_files)
        
        try:
            self.after_results = self.run_rust_implementation(test_project, dataset_files)
            after_diffs = self.measure_differences(self.after_results, "AFTER")
            
            print("✅ After test completed")
            return after_diffs
            
        except Exception as e:
            print(f"❌ After test failed: {e}")
            return None
    
    def compare_improvements(self, before_diffs, after_diffs):
        """Compare before and after results to determine if improvements were made."""
        if not before_diffs or not after_diffs:
            print("❌ Cannot compare: missing before or after results")
            return False
            
        print("\n📈 IMPROVEMENT ANALYSIS")
        print("=" * 80)
        
        improvements_found = False
        total_datasets = len(self.datasets)
        improved_datasets = 0
        
        for dataset_name in self.datasets.keys():
            if 'error' in before_diffs.get(dataset_name, {}):
                print(f"\n{dataset_name.upper()}: ❌ Before test had errors")
                continue
                
            if 'error' in after_diffs.get(dataset_name, {}):
                print(f"\n{dataset_name.upper()}: ❌ After test had errors")
                continue
            
            before = before_diffs[dataset_name]
            after = after_diffs[dataset_name]
            
            # Check improvements in training MSE
            train_improved = after['training_mse'] < before['training_mse']
            test_improved = after['test_mse'] < before['test_mse']
            train_max_improved = after['training_max_abs_diff'] < before['training_max_abs_diff']
            test_max_improved = after['test_max_abs_diff'] < before['test_max_abs_diff']
            
            any_improved = train_improved or test_improved or train_max_improved or test_max_improved
            if any_improved:
                improved_datasets += 1
                improvements_found = True
            
            print(f"\n{dataset_name.upper()} Dataset:")
            print(f"  Training MSE:     {before['training_mse']:.2e} → {after['training_mse']:.2e} {'✅' if train_improved else '❌'}")
            print(f"  Test MSE:         {before['test_mse']:.2e} → {after['test_mse']:.2e} {'✅' if test_improved else '❌'}")
            print(f"  Train Max Diff:   {before['training_max_abs_diff']:.2e} → {after['training_max_abs_diff']:.2e} {'✅' if train_max_improved else '❌'}")
            print(f"  Test Max Diff:    {before['test_max_abs_diff']:.2e} → {after['test_max_abs_diff']:.2e} {'✅' if test_max_improved else '❌'}")
            
            # Show sample predictions for context
            print(f"  Python train sample: {before['python_train_sample']}")
            print(f"  Before train sample: {before['rust_train_sample']}")
            print(f"  After train sample:  {after['rust_train_sample']}")
            
        print(f"\n📊 SUMMARY:")
        print(f"  Datasets improved: {improved_datasets}/{total_datasets}")
        print(f"  Overall improvement: {'✅ YES' if improvements_found else '❌ NO'}")
        
        return improvements_found

def main():
    """Main test runner."""
    print("🧪 Histogram-based Split Finding Improvement Test")
    print("=" * 60)
    
    with HistogramImprovementTester() as tester:
        # Setup Python baseline
        tester.setup_python_baseline()
        
        # Run before test
        before_diffs = tester.run_before_test()
        if not before_diffs:
            print("❌ Before test failed - cannot continue")
            return False
        
        # Prompt for modifications
        print("\n⏸️  PAUSE FOR MODIFICATIONS")
        print("-" * 40)
        print("Please implement the histogram-based improvements now.")
        print("When ready to test the changes, press Enter to continue...")
        input()
        
        # Run after test
        after_diffs = tester.run_after_test()
        if not after_diffs:
            print("❌ After test failed")
            return False
        
        # Compare results
        improvements_found = tester.compare_improvements(before_diffs, after_diffs)
        
        if improvements_found:
            print("\n🎉 SUCCESS: Improvements detected!")
            print("The histogram-based optimizations bring the implementation closer to Python LightGBM.")
            return True
        else:
            print("\n😞 No significant improvements detected.")
            print("The changes may need further refinement.")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)