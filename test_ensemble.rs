//! Test ensemble functionality

use lightgbm_rust::*;
use lightgbm_rust::ensemble::{ClassificationEnsemble, EnsembleConfig, EnsembleMethod, VotingStrategy};
use ndarray::{Array1, Array2};

fn main() -> Result<()> {
    // Initialize the library
    lightgbm_rust::init()?;

    println!("=== Testing ClassificationEnsemble Implementation ===");

    // Create a simple binary classification dataset
    let features = Array2::from_shape_vec((8, 2), vec![
        1.0, 2.0,
        2.0, 3.0,
        3.0, 1.0,
        4.0, 2.0,
        5.0, 1.0,
        6.0, 3.0,
        7.0, 2.0,
        8.0, 1.0,
    ]).map_err(|e| LightGBMError::data_loading(format!("Failed to create features array: {}", e)))?;
    
    let labels = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);

    // Create dataset
    let dataset = Dataset::new(features.clone(), labels, None, None, None, None)?;

    // Create multiple classifiers
    let config1 = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(10)
        .learning_rate(0.1)
        .build()?;

    let config2 = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(15)
        .learning_rate(0.05)
        .build()?;

    let mut classifier1 = LGBMClassifier::new(config1);
    let mut classifier2 = LGBMClassifier::new(config2);

    // Train the models
    println!("Training individual classifiers...");
    classifier1.fit(&dataset)?;
    classifier2.fit(&dataset)?;

    // Test individual predictions
    println!("Testing individual predictions...");
    let pred1 = classifier1.predict(&features)?;
    let pred2 = classifier2.predict(&features)?;
    println!("Classifier 1 predictions: {:?}", pred1);
    println!("Classifier 2 predictions: {:?}", pred2);

    // Create ensemble with majority voting
    println!("\n=== Testing Majority Voting Ensemble ===");
    let ensemble_config = EnsembleConfig::new()
        .with_method(EnsembleMethod::MajorityVoting)
        .with_voting_strategy(VotingStrategy::Hard);

    let ensemble = ClassificationEnsemble::new(
        vec![classifier1, classifier2], 
        ensemble_config
    )?;

    // Test ensemble predictions
    let ensemble_predictions = ensemble.predict(&features.view())?;
    println!("Ensemble (majority voting) predictions: {:?}", ensemble_predictions);

    // Test ensemble probability predictions
    let ensemble_probabilities = ensemble.predict_proba(&features.view())?;
    println!("Ensemble probabilities shape: {:?}", ensemble_probabilities.dim());
    for i in 0..ensemble_probabilities.nrows() {
        println!("Sample {}: P(class=0)={:.3}, P(class=1)={:.3}", 
                 i, ensemble_probabilities[[i, 0]], ensemble_probabilities[[i, 1]]);
    }

    // Test weighted ensemble
    println!("\n=== Testing Weighted Voting Ensemble ===");
    let mut classifier3 = LGBMClassifier::new(ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(8)
        .learning_rate(0.2)
        .build()?);
    let mut classifier4 = LGBMClassifier::new(ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(12)
        .learning_rate(0.08)
        .build()?);

    classifier3.fit(&dataset)?;
    classifier4.fit(&dataset)?;

    let weighted_config = EnsembleConfig::new()
        .with_method(EnsembleMethod::WeightedVoting)
        .with_weights(vec![0.6, 0.4])  // First model gets more weight
        .with_voting_strategy(VotingStrategy::Soft);

    let weighted_ensemble = ClassificationEnsemble::new(
        vec![classifier3, classifier4], 
        weighted_config
    )?;

    let weighted_predictions = weighted_ensemble.predict(&features.view())?;
    let weighted_probabilities = weighted_ensemble.predict_proba(&features.view())?;
    
    println!("Weighted ensemble predictions: {:?}", weighted_predictions);
    println!("Weighted ensemble probabilities shape: {:?}", weighted_probabilities.dim());

    // Test probability averaging
    println!("\n=== Testing Probability Averaging Ensemble ===");
    let avg_config = EnsembleConfig::new()
        .with_method(EnsembleMethod::Average);

    let mut classifier5 = LGBMClassifier::new(ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(10)
        .learning_rate(0.1)
        .build()?);
    let mut classifier6 = LGBMClassifier::new(ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(10)
        .learning_rate(0.1)
        .build()?);

    classifier5.fit(&dataset)?;
    classifier6.fit(&dataset)?;

    let avg_ensemble = ClassificationEnsemble::new(
        vec![classifier5, classifier6], 
        avg_config
    )?;

    let avg_predictions = avg_ensemble.predict(&features.view())?;
    let avg_probabilities = avg_ensemble.predict_proba(&features.view())?;
    
    println!("Average ensemble predictions: {:?}", avg_predictions);
    println!("Average ensemble probabilities shape: {:?}", avg_probabilities.dim());

    println!("\n✅ All ensemble tests completed successfully!");
    
    Ok(())
}