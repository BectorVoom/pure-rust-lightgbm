#!/usr/bin/env python3
"""Test the core GBDT training algorithm implementation

This example creates a simple synthetic dataset and trains a GBDT model
to verify that the implementation is working correctly, equivalent to the Rust version.
"""

import lightgbm as lgb
import numpy as np
import time

def main():
    print("Testing GBDT Training Algorithm")
    print("==============================")
    
    # Create a simple synthetic regression dataset (same as Rust version)
    features = np.array([
        # Feature values designed to have a clear pattern: y = 2*x1 + 3*x2 - x3 + noise
        [1.0, 2.0, 0.5], [2.0, 1.0, 1.0], [3.0, 0.5, 2.0], [0.5, 3.0, 0.2],
        [2.5, 1.5, 1.2], [1.5, 2.5, 0.8], [3.2, 0.8, 1.8], [0.8, 2.8, 0.5],
        [2.0, 2.0, 1.0], [1.0, 1.0, 1.5], [3.5, 1.2, 2.2], [1.2, 3.2, 0.3],
        [2.8, 1.8, 1.5], [1.8, 2.2, 1.1], [3.0, 1.0, 2.0], [1.0, 3.0, 0.8],
        [2.2, 2.3, 1.3], [2.3, 1.7, 0.9], [3.3, 0.7, 1.9], [0.7, 2.7, 0.4],
    ], dtype=np.float32)
    
    # Calculate target values: y = 2*x1 + 3*x2 - x3 + small_noise
    labels = np.array([
        2.0 * features[i, 0] + 3.0 * features[i, 1] - features[i, 2] + (i * 0.01)
        for i in range(20)
    ], dtype=np.float32)
    
    print(f"Dataset created: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Target range: [{labels.min():.2f}, {labels.max():.2f}]")
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(features, label=labels)
    
    # Configure GBDT model with same parameters as Rust version
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_iterations': 10,        # Small number for quick testing
        'learning_rate': 0.1,        # Conservative learning rate
        'num_leaves': 7,             # Small trees
        'min_data_in_leaf': 2,       # Allow small leaves for this tiny dataset
        'lambda_l2': 1.0,            # Some regularization
        'verbose': -1,               # Suppress LightGBM output
        'seed': 42,                  # For reproducibility
    }
    
    print("\nConfiguration:")
    print(f"- Objective: {params['objective']}")
    print(f"- Iterations: {params['num_iterations']}")
    print(f"- Learning rate: {params['learning_rate']}")
    print(f"- Max leaves: {params['num_leaves']}")
    print(f"- L2 regularization: {params['lambda_l2']}")
    
    # Train GBDT model
    print("\nStarting training...")
    start_time = time.time()
    
    evals_result = {}
    model = lgb.train(params, train_data, valid_sets=[train_data], 
                      valid_names=['training'], callbacks=[lgb.record_evaluation(evals_result), lgb.log_evaluation(0)])
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time * 1000:.2f}ms")
    
    # Test predictions on training data
    print("\nTesting predictions on training data:")
    predictions = model.predict(features)
    
    # Calculate training error
    total_error = 0.0
    max_error = 0.0
    for i in range(len(labels)):
        error = abs(predictions[i] - labels[i])
        total_error += error
        max_error = max(max_error, error)
        
        if i < 5:  # Show first 5 predictions vs actual
            print(f"  Sample {i}: predicted={predictions[i]:.3f}, actual={labels[i]:.3f}, error={error:.3f}")
    
    mean_abs_error = total_error / len(labels)
    print("  ...")
    print(f"Training Mean Absolute Error: {mean_abs_error:.4f}")
    print(f"Training Max Error: {max_error:.4f}")
    
    # Test on a few new samples to verify generalization
    print("\nTesting on new samples:")
    test_features = np.array([
        [1.5, 2.5, 1.0],   # Expected: 2*1.5 + 3*2.5 - 1.0 = 9.5
        [2.0, 1.0, 0.5],   # Expected: 2*2.0 + 3*1.0 - 0.5 = 6.5  
        [0.5, 3.5, 2.0],   # Expected: 2*0.5 + 3*3.5 - 2.0 = 9.5
    ], dtype=np.float32)
    
    test_predictions = model.predict(test_features)
    expected_values = [9.5, 6.5, 9.5]
    
    for i in range(3):
        prediction_error = abs(test_predictions[i] - expected_values[i])
        print(f"  Test {i}: predicted={test_predictions[i]:.3f}, expected={expected_values[i]:.3f}, error={prediction_error:.3f}")
    
    # Show model info
    print("\nModel Information:")
    print(f"- Number of trees: {model.num_trees()}")
    print(f"- Model dump available: {model.dump_model() is not None}")
    
    # Get training history from the evaluation results
    if 'training' in evals_result and 'l2' in evals_result['training']:
        train_loss = evals_result['training']['l2']
        if len(train_loss) > 0:
            initial_loss = train_loss[0]
            final_loss = train_loss[-1]
            print(f"- Initial training loss: {initial_loss:.6f}")
            print(f"- Final training loss: {final_loss:.6f}")
            print(f"- Loss improvement: {initial_loss - final_loss:.6f}")
    
    print("\n✅ GBDT training algorithm test completed successfully!")
    
    if mean_abs_error < 1.0:
        print("✅ Model achieved good accuracy (MAE < 1.0)")
    else:
        print(f"⚠️  Model accuracy could be improved (MAE = {mean_abs_error:.4f})")
    
    # Return results for comparison
    return {
        'training_time_ms': training_time * 1000,
        'mean_abs_error': mean_abs_error,
        'max_error': max_error,
        'predictions': predictions.tolist(),
        'test_predictions': test_predictions.tolist()
    }

if __name__ == "__main__":
    results = main()