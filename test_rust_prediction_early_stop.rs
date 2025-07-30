use pure_rust_lightgbm::core::prediction_early_stop::*;

fn main() {
    println!("=== Rust Prediction Early Stop Tests ===");
    
    // Test 1: None type
    {
        let config = PredictionEarlyStopConfig::new(5, 0.5);
        let instance = create_prediction_early_stop_instance("none", &config);
        
        let pred1 = vec![0.1, 0.9];
        let pred2 = vec![0.99, 0.01];
        
        let result1 = (instance.callback_function)(&pred1);
        let result2 = (instance.callback_function)(&pred2);
        
        println!("None type - pred1: {} (expected: false)", result1);
        println!("None type - pred2: {} (expected: false)", result2);
        println!("None type - round_period: {} (expected: {})", instance.round_period, i32::MAX);
        
        assert_eq!(result1, false);
        assert_eq!(result2, false);
        assert_eq!(instance.round_period, i32::MAX);
    }
    
    // Test 2: Multiclass type
    {
        let config = PredictionEarlyStopConfig::new(5, 0.5);
        let instance = create_prediction_early_stop_instance("multiclass", &config);
        
        // Test cases where margin > threshold (should stop)
        let pred1 = vec![0.8, 0.2];  // margin = 0.6 > 0.5
        let pred2 = vec![0.9, 0.1, 0.0];  // margin = 0.8 > 0.5
        
        // Test cases where margin <= threshold (should not stop)
        let pred3 = vec![0.6, 0.4];  // margin = 0.2 <= 0.5
        let pred4 = vec![0.75, 0.25];  // margin = 0.5 = 0.5 (not >)
        
        let result1 = (instance.callback_function)(&pred1);
        let result2 = (instance.callback_function)(&pred2);
        let result3 = (instance.callback_function)(&pred3);
        let result4 = (instance.callback_function)(&pred4);
        
        println!("Multiclass - pred1 (0.8,0.2): {} (expected: true)", result1);
        println!("Multiclass - pred2 (0.9,0.1,0.0): {} (expected: true)", result2);
        println!("Multiclass - pred3 (0.6,0.4): {} (expected: false)", result3);
        println!("Multiclass - pred4 (0.75,0.25): {} (expected: false)", result4);
        println!("Multiclass - round_period: {} (expected: 5)", instance.round_period);
        
        assert_eq!(result1, true);
        assert_eq!(result2, true);
        assert_eq!(result3, false);
        assert_eq!(result4, false);
        assert_eq!(instance.round_period, 5);
        
        // Test edge case - single prediction (should return false in Rust, but C++ throws)
        let pred_single = vec![0.5];
        let result_single = (instance.callback_function)(&pred_single);
        println!("Multiclass - single pred: {} (C++ throws, Rust returns false)", result_single);
        assert_eq!(result_single, false);
        
        // Test edge case - empty prediction (should return false in Rust, but C++ throws)
        let pred_empty: Vec<f64> = vec![];
        let result_empty = (instance.callback_function)(&pred_empty);
        println!("Multiclass - empty pred: {} (C++ throws, Rust returns false)", result_empty);
        assert_eq!(result_empty, false);
    }
    
    // Test 3: Binary type
    {
        let config = PredictionEarlyStopConfig::new(3, 1.0);
        let instance = create_prediction_early_stop_instance("binary", &config);
        
        // Test cases where 2*|pred| > threshold (should stop)
        let pred1 = vec![0.6];  // 2*0.6 = 1.2 > 1.0
        let pred2 = vec![-0.7]; // 2*0.7 = 1.4 > 1.0
        
        // Test cases where 2*|pred| <= threshold (should not stop)
        let pred3 = vec![0.4];  // 2*0.4 = 0.8 <= 1.0
        let pred4 = vec![0.5];  // 2*0.5 = 1.0 = 1.0 (not >)
        let pred5 = vec![-0.3]; // 2*0.3 = 0.6 <= 1.0
        
        let result1 = (instance.callback_function)(&pred1);
        let result2 = (instance.callback_function)(&pred2);
        let result3 = (instance.callback_function)(&pred3);
        let result4 = (instance.callback_function)(&pred4);
        let result5 = (instance.callback_function)(&pred5);
        
        println!("Binary - pred1 (0.6): {} (expected: true)", result1);
        println!("Binary - pred2 (-0.7): {} (expected: true)", result2);
        println!("Binary - pred3 (0.4): {} (expected: false)", result3);
        println!("Binary - pred4 (0.5): {} (expected: false)", result4);
        println!("Binary - pred5 (-0.3): {} (expected: false)", result5);
        println!("Binary - round_period: {} (expected: 3)", instance.round_period);
        
        assert_eq!(result1, true);
        assert_eq!(result2, true);
        assert_eq!(result3, false);
        assert_eq!(result4, false);
        assert_eq!(result5, false);
        assert_eq!(instance.round_period, 3);
        
        // Test edge case - multiple predictions (should return false in Rust, but C++ throws)
        let pred_multi = vec![0.5, 0.3];
        let result_multi = (instance.callback_function)(&pred_multi);
        println!("Binary - multiple pred: {} (C++ throws, Rust returns false)", result_multi);
        assert_eq!(result_multi, false);
        
        // Test edge case - empty prediction (should return false in Rust, but C++ throws)
        let pred_empty: Vec<f64> = vec![];
        let result_empty = (instance.callback_function)(&pred_empty);
        println!("Binary - empty pred: {} (C++ throws, Rust returns false)", result_empty);
        assert_eq!(result_empty, false);        
    }
    
    // Test 4: Unknown type (should panic in Rust, like C++ throws)
    {
        let config = PredictionEarlyStopConfig::new(1, 0.0);
        
        let result = std::panic::catch_unwind(|| {
            create_prediction_early_stop_instance("unknown", &config);
        });
        
        match result {
            Ok(_) => {
                panic!("Expected panic for unknown type");
            }
            Err(_) => {
                println!("Unknown type correctly panicked (like C++ throws)");
            }
        }
    }
    
    println!("All Rust tests passed!");
}