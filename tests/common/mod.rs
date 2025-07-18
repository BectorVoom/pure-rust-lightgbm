//! Common test utilities for pure Rust LightGBM integration tests.

use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::fs;
use std::path::Path;

/// Create test features for regression tasks
pub fn create_test_features_regression(num_samples: usize, num_features: usize) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut features = Array2::zeros((num_samples, num_features));
    
    for i in 0..num_samples {
        for j in 0..num_features {
            features[[i, j]] = rng.gen_range(-5.0..5.0);
        }
    }
    
    features
}

/// Create test labels for regression based on features
pub fn create_test_labels_regression(features: &Array2<f32>) -> Array1<f32> {
    let num_samples = features.nrows();
    let mut labels = Array1::zeros(num_samples);
    
    for i in 0..num_samples {
        // Simple linear combination for test labels
        let mut label = 0.0;
        for j in 0..features.ncols() {
            label += features[[i, j]] * ((j + 1) as f32 * 0.1);
        }
        labels[i] = label;
    }
    
    labels
}

/// Create test features for binary classification
pub fn create_test_features_binary(num_samples: usize, num_features: usize) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(123);
    
    let mut features = Array2::zeros((num_samples, num_features));
    
    for i in 0..num_samples {
        for j in 0..num_features {
            features[[i, j]] = rng.gen_range(-3.0..3.0);
        }
    }
    
    features
}

/// Create test labels for binary classification
pub fn create_test_labels_binary(features: &Array2<f32>) -> Array1<f32> {
    let num_samples = features.nrows();
    let mut labels = Array1::zeros(num_samples);
    
    for i in 0..num_samples {
        // Simple decision boundary
        let mut score = 0.0;
        for j in 0..features.ncols() {
            score += features[[i, j]] * if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        labels[i] = if score > 0.0 { 1.0 } else { 0.0 };
    }
    
    labels
}

/// Create test features for multiclass classification
pub fn create_test_features_multiclass(num_samples: usize, num_features: usize) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(456);
    
    let mut features = Array2::zeros((num_samples, num_features));
    
    for i in 0..num_samples {
        for j in 0..num_features {
            features[[i, j]] = rng.gen_range(-2.0..2.0);
        }
    }
    
    features
}

/// Create test labels for multiclass classification
pub fn create_test_labels_multiclass(features: &Array2<f32>, num_classes: usize) -> Array1<f32> {
    let num_samples = features.nrows();
    let mut labels = Array1::zeros(num_samples);
    
    for i in 0..num_samples {
        let mut max_score = f32::NEG_INFINITY;
        let mut best_class = 0;
        
        for class in 0..num_classes {
            let mut score = 0.0;
            for j in 0..features.ncols() {
                score += features[[i, j]] * ((class + j) as f32 * 0.1);
            }
            
            if score > max_score {
                max_score = score;
                best_class = class;
            }
        }
        
        labels[i] = best_class as f32;
    }
    
    labels
}

/// Create test weights
pub fn create_test_weights(num_samples: usize) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(789);
    let mut weights = Array1::zeros(num_samples);
    
    for i in 0..num_samples {
        weights[i] = rng.gen_range(0.1..2.0);
    }
    
    weights
}

/// Create test CSV file
pub fn create_test_csv<P: AsRef<Path>>(
    path: P,
    features: &Array2<f32>,
    labels: &Array1<f32>,
    weights: Option<&Array1<f32>>,
    feature_names: Option<&[String]>,
) -> std::io::Result<()> {
    let num_samples = features.nrows();
    let num_features = features.ncols();
    
    let mut content = String::new();
    
    // Write header
    if let Some(names) = feature_names {
        content.push_str(&names.join(","));
    } else {
        let default_names: Vec<String> = (0..num_features)
            .map(|i| format!("feature_{}", i))
            .collect();
        content.push_str(&default_names.join(","));
    }
    content.push_str(",target");
    
    if weights.is_some() {
        content.push_str(",weight");
    }
    content.push('\n');
    
    // Write data rows
    for i in 0..num_samples {
        let feature_values: Vec<String> = (0..num_features)
            .map(|j| features[[i, j]].to_string())
            .collect();
        content.push_str(&feature_values.join(","));
        content.push(',');
        content.push_str(&labels[i].to_string());
        
        if let Some(w) = weights {
            content.push(',');
            content.push_str(&w[i].to_string());
        }
        content.push('\n');
    }
    
    fs::write(path, content)
}

/// Create test JSON file
pub fn create_test_json<P: AsRef<Path>>(
    path: P,
    features: &Array2<f32>,
    labels: &Array1<f32>,
) -> std::io::Result<()> {
    use serde_json::json;
    
    let num_samples = features.nrows();
    let num_features = features.ncols();
    
    let mut data = Vec::new();
    
    for i in 0..num_samples {
        let mut row = serde_json::Map::new();
        
        for j in 0..num_features {
            row.insert(
                format!("feature_{}", j),
                json!(features[[i, j]]),
            );
        }
        
        row.insert("target".to_string(), json!(labels[i]));
        
        data.push(serde_json::Value::Object(row));
    }
    
    let json_data = json!({
        "data": data,
        "metadata": {
            "num_samples": num_samples,
            "num_features": num_features,
            "created_by": "lightgbm_rust_test"
        }
    });
    
    fs::write(path, serde_json::to_string_pretty(&json_data)?)?;
    Ok(())
}

/// Create test dataset with missing values
pub fn create_test_features_with_missing(
    num_samples: usize,
    num_features: usize,
    missing_rate: f32,
) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(999);
    let mut features = Array2::zeros((num_samples, num_features));
    
    for i in 0..num_samples {
        for j in 0..num_features {
            if rng.gen::<f32>() < missing_rate {
                features[[i, j]] = f32::NAN;
            } else {
                features[[i, j]] = rng.gen_range(-3.0..3.0);
            }
        }
    }
    
    features
}

/// Create test categorical features
pub fn create_test_categorical_features(
    num_samples: usize,
    num_features: usize,
    num_categories: usize,
) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(111);
    let mut features = Array2::zeros((num_samples, num_features));
    
    for i in 0..num_samples {
        for j in 0..num_features {
            features[[i, j]] = rng.gen_range(0..num_categories) as f32;
        }
    }
    
    features
}

/// Create test mixed features (numerical and categorical)
pub fn create_test_mixed_features(
    num_samples: usize,
    num_numerical: usize,
    num_categorical: usize,
    num_categories: usize,
) -> (Array2<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(222);
    let total_features = num_numerical + num_categorical;
    let mut features = Array2::zeros((num_samples, total_features));
    
    // Categorical feature indices
    let categorical_indices: Vec<usize> = (num_numerical..total_features).collect();
    
    for i in 0..num_samples {
        // Numerical features
        for j in 0..num_numerical {
            features[[i, j]] = rng.gen_range(-5.0..5.0);
        }
        
        // Categorical features
        for j in num_numerical..total_features {
            features[[i, j]] = rng.gen_range(0..num_categories) as f32;
        }
    }
    
    (features, categorical_indices)
}

/// Create test features with outliers
pub fn create_test_features_with_outliers(
    num_samples: usize,
    num_features: usize,
    outlier_rate: f32,
) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(333);
    let mut features = Array2::zeros((num_samples, num_features));
    
    for i in 0..num_samples {
        for j in 0..num_features {
            if rng.gen::<f32>() < outlier_rate {
                // Create outlier value
                features[[i, j]] = if rng.gen::<bool>() { 100.0 } else { -100.0 };
            } else {
                features[[i, j]] = rng.gen_range(-5.0..5.0);
            }
        }
    }
    
    features
}

/// Create test imbalanced classification data
pub fn create_test_imbalanced_classification(
    num_samples: usize,
    num_features: usize,
    minority_ratio: f32,
) -> (Array2<f32>, Array1<f32>) {
    let mut rng = StdRng::seed_from_u64(444);
    let features = Array2::zeros((num_samples, num_features));
    let mut labels = Array1::zeros(num_samples);
    
    let num_minority = (num_samples as f32 * minority_ratio) as usize;
    
    // Set first num_minority samples as minority class
    for i in 0..num_minority {
        labels[i] = 1.0;
    }
    
    // Shuffle labels
    let mut indices: Vec<usize> = (0..num_samples).collect();
    indices.shuffle(&mut rng);
    
    let mut shuffled_labels = Array1::zeros(num_samples);
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        shuffled_labels[new_idx] = labels[old_idx];
    }
    
    (features, shuffled_labels)
}

/// Validate test dataset properties
pub fn validate_test_dataset(
    features: &Array2<f32>,
    labels: &Array1<f32>,
    weights: Option<&Array1<f32>>,
) -> bool {
    let num_samples = features.nrows();
    
    // Check dimensions match
    if labels.len() != num_samples {
        return false;
    }
    
    if let Some(w) = weights {
        if w.len() != num_samples {
            return false;
        }
    }
    
    // Check for valid values
    if features.iter().any(|x| x.is_infinite()) {
        return false;
    }
    
    if labels.iter().any(|x| x.is_infinite()) {
        return false;
    }
    
    if let Some(w) = weights {
        if w.iter().any(|x| x.is_infinite() || *x < 0.0) {
            return false;
        }
    }
    
    true
}

/// Create test configuration for different objectives
pub fn create_test_config(objective: ObjectiveType) -> Result<Config> {
    match objective {
        ObjectiveType::Regression => {
            ConfigBuilder::new()
                .learning_rate(0.1)
                .num_iterations(100)
                .num_leaves(31)
                .objective(ObjectiveType::Regression)
                .build()
        }
        ObjectiveType::Binary => {
            ConfigBuilder::new()
                .learning_rate(0.1)
                .num_iterations(100)
                .num_leaves(31)
                .objective(ObjectiveType::Binary)
                .build()
        }
        ObjectiveType::Multiclass => {
            ConfigBuilder::new()
                .learning_rate(0.1)
                .num_iterations(100)
                .num_leaves(31)
                .objective(ObjectiveType::Multiclass)
                .num_class(3)
                .build()
        }
        ObjectiveType::Ranking => {
            ConfigBuilder::new()
                .learning_rate(0.1)
                .num_iterations(100)
                .num_leaves(31)
                .objective(ObjectiveType::Ranking)
                .build()
        }
        ObjectiveType::Poisson => {
            ConfigBuilder::new()
                .learning_rate(0.1)
                .num_iterations(100)
                .num_leaves(31)
                .objective(ObjectiveType::Poisson)
                .build()
        }
        ObjectiveType::Gamma => {
            ConfigBuilder::new()
                .learning_rate(0.1)
                .num_iterations(100)
                .num_leaves(31)
                .objective(ObjectiveType::Gamma)
                .build()
        }
        ObjectiveType::Tweedie => {
            ConfigBuilder::new()
                .learning_rate(0.1)
                .num_iterations(100)
                .num_leaves(31)
                .objective(ObjectiveType::Tweedie)
                .build()
        }
    }
}

/// Create test dataset configuration
pub fn create_test_dataset_config() -> DatasetConfig {
    DatasetConfig::new()
        .with_max_bin(256)
        .with_categorical_features(vec![])
        .with_target_column("target")
}

/// Macro for creating test data more easily
#[macro_export]
macro_rules! create_test_data {
    (regression, $samples:expr, $features:expr) => {{
        let features = common::create_test_features_regression($samples, $features);
        let labels = common::create_test_labels_regression(&features);
        (features, labels)
    }};
    
    (binary, $samples:expr, $features:expr) => {{
        let features = common::create_test_features_binary($samples, $features);
        let labels = common::create_test_labels_binary(&features);
        (features, labels)
    }};
    
    (multiclass, $samples:expr, $features:expr, $classes:expr) => {{
        let features = common::create_test_features_multiclass($samples, $features);
        let labels = common::create_test_labels_multiclass(&features, $classes);
        (features, labels)
    }};
}

/// Test fixture for comprehensive dataset testing
pub struct TestDataFixture {
    pub features: Array2<f32>,
    pub labels: Array1<f32>,
    pub weights: Option<Array1<f32>>,
    pub categorical_indices: Vec<usize>,
    pub feature_names: Vec<String>,
}

impl TestDataFixture {
    pub fn new_regression(num_samples: usize, num_features: usize) -> Self {
        let features = create_test_features_regression(num_samples, num_features);
        let labels = create_test_labels_regression(&features);
        let weights = Some(create_test_weights(num_samples));
        let categorical_indices = vec![];
        let feature_names = (0..num_features)
            .map(|i| format!("feature_{}", i))
            .collect();
        
        TestDataFixture {
            features,
            labels,
            weights,
            categorical_indices,
            feature_names,
        }
    }
    
    pub fn new_binary(num_samples: usize, num_features: usize) -> Self {
        let features = create_test_features_binary(num_samples, num_features);
        let labels = create_test_labels_binary(&features);
        let weights = Some(create_test_weights(num_samples));
        let categorical_indices = vec![];
        let feature_names = (0..num_features)
            .map(|i| format!("feature_{}", i))
            .collect();
        
        TestDataFixture {
            features,
            labels,
            weights,
            categorical_indices,
            feature_names,
        }
    }
    
    pub fn new_mixed(
        num_samples: usize,
        num_numerical: usize,
        num_categorical: usize,
        num_categories: usize,
    ) -> Self {
        let (features, categorical_indices) = create_test_mixed_features(
            num_samples,
            num_numerical,
            num_categorical,
            num_categories,
        );
        let labels = create_test_labels_regression(&features);
        let weights = Some(create_test_weights(num_samples));
        let feature_names = (0..features.ncols())
            .map(|i| format!("feature_{}", i))
            .collect();
        
        TestDataFixture {
            features,
            labels,
            weights,
            categorical_indices,
            feature_names,
        }
    }
    
    pub fn with_missing_values(mut self, missing_rate: f32) -> Self {
        let num_samples = self.features.nrows();
        let num_features = self.features.ncols();
        
        let mut rng = StdRng::seed_from_u64(333);
        
        for i in 0..num_samples {
            for j in 0..num_features {
                if rng.gen::<f32>() < missing_rate {
                    self.features[[i, j]] = f32::NAN;
                }
            }
        }
        
        self
    }
    
    pub fn create_dataset(&self) -> Result<Dataset> {
        Dataset::new(
            self.features.clone(),
            self.labels.clone(),
            self.weights.clone(),
            None,
            Some(self.feature_names.clone()),
            None,
        )
    }
    
    pub fn create_dataset_config(&self) -> DatasetConfig {
        DatasetConfig::new()
            .with_max_bin(256)
            .with_categorical_features(self.categorical_indices.clone())
            .with_target_column("target")
    }
    
    pub fn validate(&self) -> bool {
        validate_test_dataset(&self.features, &self.labels, self.weights.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_test_features_regression() {
        let features = create_test_features_regression(100, 5);
        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 5);
        assert!(features.iter().all(|x| x.is_finite()));
    }
    
    #[test]
    fn test_create_test_labels_regression() {
        let features = create_test_features_regression(50, 3);
        let labels = create_test_labels_regression(&features);
        assert_eq!(labels.len(), 50);
        assert!(labels.iter().all(|x| x.is_finite()));
    }
    
    #[test]
    fn test_create_test_features_binary() {
        let features = create_test_features_binary(75, 4);
        let labels = create_test_labels_binary(&features);
        assert_eq!(features.nrows(), 75);
        assert_eq!(features.ncols(), 4);
        assert_eq!(labels.len(), 75);
        assert!(labels.iter().all(|&x| x == 0.0 || x == 1.0));
    }
    
    #[test]
    fn test_create_test_features_multiclass() {
        let features = create_test_features_multiclass(60, 6);
        let labels = create_test_labels_multiclass(&features, 3);
        assert_eq!(features.nrows(), 60);
        assert_eq!(features.ncols(), 6);
        assert_eq!(labels.len(), 60);
        assert!(labels.iter().all(|&x| x >= 0.0 && x < 3.0));
    }
    
    #[test]
    fn test_test_data_fixture() {
        let fixture = TestDataFixture::new_regression(100, 5);
        assert!(fixture.validate());
        assert_eq!(fixture.features.nrows(), 100);
        assert_eq!(fixture.features.ncols(), 5);
        assert_eq!(fixture.labels.len(), 100);
        assert!(fixture.weights.is_some());
        assert_eq!(fixture.feature_names.len(), 5);
    }
    
    #[test]
    fn test_test_data_fixture_with_missing() {
        let fixture = TestDataFixture::new_regression(50, 3)
            .with_missing_values(0.1);
        assert!(fixture.validate());
        
        // Check that some values are NaN
        let nan_count = fixture.features.iter().filter(|x| x.is_nan()).count();
        assert!(nan_count > 0);
    }
}