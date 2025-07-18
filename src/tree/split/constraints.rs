//! Constraint enforcement for tree splits in the Pure Rust LightGBM framework.
//!
//! This module provides constraint enforcement mechanisms including monotonic
//! constraints, feature interaction constraints, and custom constraint validation.

use crate::core::types::{FeatureIndex, Score};
use crate::tree::split::finder::SplitInfo;
use std::collections::{HashMap, HashSet};

/// Monotonic constraint types for individual features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicConstraint {
    /// No monotonic constraint
    None,
    /// Feature must have monotonically increasing relationship with target
    Increasing,
    /// Feature must have monotonically decreasing relationship with target
    Decreasing,
}

impl Default for MonotonicConstraint {
    fn default() -> Self {
        MonotonicConstraint::None
    }
}

impl std::fmt::Display for MonotonicConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonotonicConstraint::None => write!(f, "none"),
            MonotonicConstraint::Increasing => write!(f, "increasing"),
            MonotonicConstraint::Decreasing => write!(f, "decreasing"),
        }
    }
}

/// Feature interaction constraint specification.
#[derive(Debug, Clone)]
pub struct InteractionConstraint {
    /// Groups of features that can interact with each other
    pub allowed_groups: Vec<HashSet<FeatureIndex>>,
    /// Features that are forbidden from being used together
    pub forbidden_pairs: HashSet<(FeatureIndex, FeatureIndex)>,
}

impl Default for InteractionConstraint {
    fn default() -> Self {
        InteractionConstraint {
            allowed_groups: Vec::new(),
            forbidden_pairs: HashSet::new(),
        }
    }
}

impl InteractionConstraint {
    /// Creates a new empty interaction constraint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an allowed group of features that can interact.
    pub fn add_allowed_group(&mut self, features: Vec<FeatureIndex>) {
        if !features.is_empty() {
            self.allowed_groups.push(features.into_iter().collect());
        }
    }

    /// Adds a forbidden pair of features.
    pub fn add_forbidden_pair(&mut self, feature_a: FeatureIndex, feature_b: FeatureIndex) {
        self.forbidden_pairs.insert((feature_a.min(feature_b), feature_a.max(feature_b)));
    }

    /// Checks if two features can be used together in the same tree path.
    pub fn can_interact(&self, feature_a: FeatureIndex, feature_b: FeatureIndex) -> bool {
        // Check forbidden pairs
        let pair = (feature_a.min(feature_b), feature_a.max(feature_b));
        if self.forbidden_pairs.contains(&pair) {
            return false;
        }

        // If no allowed groups are specified, allow all interactions
        if self.allowed_groups.is_empty() {
            return true;
        }

        // Check if both features are in the same allowed group
        for group in &self.allowed_groups {
            if group.contains(&feature_a) && group.contains(&feature_b) {
                return true;
            }
        }

        false
    }

    /// Checks if a feature can be used given the already used features.
    pub fn can_use_feature(&self, feature: FeatureIndex, used_features: &[FeatureIndex]) -> bool {
        for &used_feature in used_features {
            if !self.can_interact(feature, used_feature) {
                return false;
            }
        }
        true
    }
}

/// Comprehensive constraint manager for tree construction.
#[derive(Debug, Clone)]
pub struct ConstraintManager {
    /// Monotonic constraints for individual features
    monotonic_constraints: HashMap<FeatureIndex, MonotonicConstraint>,
    /// Feature interaction constraints
    interaction_constraints: InteractionConstraint,
    /// Maximum tree depth constraint
    max_depth: Option<usize>,
    /// Minimum samples per leaf constraint
    min_samples_leaf: usize,
    /// Minimum samples per split constraint
    min_samples_split: usize,
    /// Custom constraint functions
    custom_validators: Vec<Box<dyn ConstraintValidator>>,
}

impl Default for ConstraintManager {
    fn default() -> Self {
        ConstraintManager {
            monotonic_constraints: HashMap::new(),
            interaction_constraints: InteractionConstraint::new(),
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            custom_validators: Vec::new(),
        }
    }
}

impl ConstraintManager {
    /// Creates a new constraint manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets monotonic constraint for a feature.
    pub fn set_monotonic_constraint(&mut self, feature: FeatureIndex, constraint: MonotonicConstraint) {
        if constraint != MonotonicConstraint::None {
            self.monotonic_constraints.insert(feature, constraint);
        } else {
            self.monotonic_constraints.remove(&feature);
        }
    }

    /// Sets multiple monotonic constraints.
    pub fn set_monotonic_constraints(&mut self, constraints: HashMap<FeatureIndex, MonotonicConstraint>) {
        self.monotonic_constraints = constraints;
    }

    /// Sets interaction constraints.
    pub fn set_interaction_constraints(&mut self, constraints: InteractionConstraint) {
        self.interaction_constraints = constraints;
    }

    /// Sets maximum tree depth constraint.
    pub fn set_max_depth(&mut self, max_depth: Option<usize>) {
        self.max_depth = max_depth;
    }

    /// Sets minimum samples per leaf constraint.
    pub fn set_min_samples_leaf(&mut self, min_samples: usize) {
        self.min_samples_leaf = min_samples;
    }

    /// Sets minimum samples per split constraint.
    pub fn set_min_samples_split(&mut self, min_samples: usize) {
        self.min_samples_split = min_samples;
    }

    /// Adds a custom constraint validator.
    pub fn add_custom_validator(&mut self, validator: Box<dyn ConstraintValidator>) {
        self.custom_validators.push(validator);
    }

    /// Validates a split against all constraints.
    pub fn validate_split(
        &self,
        split: &SplitInfo,
        current_depth: usize,
        parent_output: Score,
        tree_path_features: &[FeatureIndex],
    ) -> ConstraintValidationResult {
        let mut result = ConstraintValidationResult::new();

        // Check depth constraint
        if let Some(max_depth) = self.max_depth {
            if current_depth >= max_depth {
                result.add_violation(
                    ConstraintType::MaxDepth,
                    format!("Current depth {} exceeds maximum depth {}", current_depth, max_depth),
                );
            }
        }

        // Check minimum samples constraints
        if (split.left_count as usize) < self.min_samples_leaf {
            result.add_violation(
                ConstraintType::MinSamplesLeaf,
                format!("Left child has {} samples, minimum required is {}", 
                    split.left_count, self.min_samples_leaf),
            );
        }

        if (split.right_count as usize) < self.min_samples_leaf {
            result.add_violation(
                ConstraintType::MinSamplesLeaf,
                format!("Right child has {} samples, minimum required is {}", 
                    split.right_count, self.min_samples_leaf),
            );
        }

        let total_samples = split.left_count + split.right_count;
        if (total_samples as usize) < self.min_samples_split {
            result.add_violation(
                ConstraintType::MinSamplesSplit,
                format!("Total samples {} is less than minimum required {}", 
                    total_samples, self.min_samples_split),
            );
        }

        // Check monotonic constraints
        if let Some(&constraint) = self.monotonic_constraints.get(&split.feature) {
            if !self.validate_monotonic_constraint(split, constraint, parent_output) {
                result.add_violation(
                    ConstraintType::Monotonic,
                    format!("Split violates {} monotonic constraint for feature {}", 
                        constraint, split.feature),
                );
            }
        }

        // Check interaction constraints
        if !self.interaction_constraints.can_use_feature(split.feature, tree_path_features) {
            result.add_violation(
                ConstraintType::Interaction,
                format!("Feature {} cannot be used with current tree path features", 
                    split.feature),
            );
        }

        // Check custom constraints
        for validator in &self.custom_validators {
            if let Some(violation) = validator.validate(split, current_depth, parent_output, tree_path_features) {
                result.add_violation(ConstraintType::Custom, violation);
            }
        }

        result
    }

    /// Validates monotonic constraint for a specific split.
    fn validate_monotonic_constraint(
        &self,
        split: &SplitInfo,
        constraint: MonotonicConstraint,
        _parent_output: Score,
    ) -> bool {
        match constraint {
            MonotonicConstraint::None => true,
            MonotonicConstraint::Increasing => {
                // For increasing constraint, left child (lower feature values) should have
                // lower or equal output compared to right child (higher feature values)
                split.left_output <= split.right_output
            }
            MonotonicConstraint::Decreasing => {
                // For decreasing constraint, left child (lower feature values) should have
                // higher or equal output compared to right child (higher feature values)
                split.left_output >= split.right_output
            }
        }
    }

    /// Filters a list of candidate features based on constraints.
    pub fn filter_candidate_features(
        &self,
        candidates: &[FeatureIndex],
        tree_path_features: &[FeatureIndex],
    ) -> Vec<FeatureIndex> {
        candidates
            .iter()
            .filter(|&&feature| self.interaction_constraints.can_use_feature(feature, tree_path_features))
            .copied()
            .collect()
    }

    /// Returns all monotonic constraints.
    pub fn monotonic_constraints(&self) -> &HashMap<FeatureIndex, MonotonicConstraint> {
        &self.monotonic_constraints
    }

    /// Returns interaction constraints.
    pub fn interaction_constraints(&self) -> &InteractionConstraint {
        &self.interaction_constraints
    }
}

/// Types of constraints that can be violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Monotonic constraint violation
    Monotonic,
    /// Feature interaction constraint violation
    Interaction,
    /// Maximum depth constraint violation
    MaxDepth,
    /// Minimum samples per leaf constraint violation
    MinSamplesLeaf,
    /// Minimum samples per split constraint violation
    MinSamplesSplit,
    /// Custom constraint violation
    Custom,
}

impl std::fmt::Display for ConstraintType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintType::Monotonic => write!(f, "monotonic"),
            ConstraintType::Interaction => write!(f, "interaction"),
            ConstraintType::MaxDepth => write!(f, "max_depth"),
            ConstraintType::MinSamplesLeaf => write!(f, "min_samples_leaf"),
            ConstraintType::MinSamplesSplit => write!(f, "min_samples_split"),
            ConstraintType::Custom => write!(f, "custom"),
        }
    }
}

/// Result of constraint validation.
#[derive(Debug, Clone)]
pub struct ConstraintValidationResult {
    violations: Vec<(ConstraintType, String)>,
}

impl ConstraintValidationResult {
    fn new() -> Self {
        ConstraintValidationResult {
            violations: Vec::new(),
        }
    }

    fn add_violation(&mut self, constraint_type: ConstraintType, message: String) {
        self.violations.push((constraint_type, message));
    }

    /// Returns true if there are no constraint violations.
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    /// Returns all constraint violations.
    pub fn violations(&self) -> &[(ConstraintType, String)] {
        &self.violations
    }

    /// Returns violations of a specific constraint type.
    pub fn violations_of_type(&self, constraint_type: ConstraintType) -> Vec<&String> {
        self.violations
            .iter()
            .filter_map(|(t, msg)| if *t == constraint_type { Some(msg) } else { None })
            .collect()
    }

    /// Returns the number of violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

/// Trait for custom constraint validators.
pub trait ConstraintValidator: Send + Sync {
    /// Validates a split and returns an error message if the constraint is violated.
    fn validate(
        &self,
        split: &SplitInfo,
        current_depth: usize,
        parent_output: Score,
        tree_path_features: &[FeatureIndex],
    ) -> Option<String>;

    /// Returns the name of this constraint validator.
    fn name(&self) -> &str;
}

/// Example custom constraint: maximum feature usage limit.
pub struct MaxFeatureUsageConstraint {
    max_usage_per_feature: HashMap<FeatureIndex, usize>,
}

impl MaxFeatureUsageConstraint {
    pub fn new(max_usage_per_feature: HashMap<FeatureIndex, usize>) -> Self {
        MaxFeatureUsageConstraint {
            max_usage_per_feature,
        }
    }
}

impl ConstraintValidator for MaxFeatureUsageConstraint {
    fn validate(
        &self,
        split: &SplitInfo,
        _current_depth: usize,
        _parent_output: Score,
        tree_path_features: &[FeatureIndex],
    ) -> Option<String> {
        if let Some(&max_usage) = self.max_usage_per_feature.get(&split.feature) {
            let current_usage = tree_path_features.iter().filter(|&&f| f == split.feature).count();
            if current_usage >= max_usage {
                return Some(format!(
                    "Feature {} already used {} times, maximum allowed is {}",
                    split.feature, current_usage, max_usage
                ));
            }
        }
        None
    }

    fn name(&self) -> &str {
        "max_feature_usage"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_constraint_increasing() {
        let mut split = SplitInfo::new();
        split.left_output = 1.0;
        split.right_output = 2.0;

        let mut manager = ConstraintManager::new();
        manager.set_monotonic_constraint(0, MonotonicConstraint::Increasing);

        assert!(manager.validate_monotonic_constraint(&split, MonotonicConstraint::Increasing, 1.5));

        split.left_output = 2.0;
        split.right_output = 1.0;
        assert!(!manager.validate_monotonic_constraint(&split, MonotonicConstraint::Increasing, 1.5));
    }

    #[test]
    fn test_monotonic_constraint_decreasing() {
        let mut split = SplitInfo::new();
        split.left_output = 2.0;
        split.right_output = 1.0;

        let mut manager = ConstraintManager::new();
        manager.set_monotonic_constraint(0, MonotonicConstraint::Decreasing);

        assert!(manager.validate_monotonic_constraint(&split, MonotonicConstraint::Decreasing, 1.5));

        split.left_output = 1.0;
        split.right_output = 2.0;
        assert!(!manager.validate_monotonic_constraint(&split, MonotonicConstraint::Decreasing, 1.5));
    }

    #[test]
    fn test_interaction_constraint() {
        let mut constraints = InteractionConstraint::new();
        constraints.add_allowed_group(vec![0, 1, 2]);
        constraints.add_allowed_group(vec![3, 4]);
        constraints.add_forbidden_pair(1, 3);

        // Features in same group can interact
        assert!(constraints.can_interact(0, 1));
        assert!(constraints.can_interact(1, 2));
        assert!(constraints.can_interact(3, 4));

        // Features in different groups cannot interact (unless explicitly allowed)
        assert!(!constraints.can_interact(0, 3));
        assert!(!constraints.can_interact(2, 4));

        // Forbidden pairs cannot interact even if in same group
        assert!(!constraints.can_interact(1, 3));
    }

    #[test]
    fn test_constraint_validation() {
        let mut manager = ConstraintManager::new();
        manager.set_max_depth(Some(3));
        manager.set_min_samples_leaf(5);
        manager.set_monotonic_constraint(0, MonotonicConstraint::Increasing);

        let mut split = SplitInfo::new();
        split.feature = 0;
        split.left_count = 3; // Below minimum
        split.right_count = 10;
        split.left_output = 2.0; // Violates increasing constraint
        split.right_output = 1.0;

        let result = manager.validate_split(&split, 4, 1.5, &[]);

        assert!(!result.is_valid());
        assert!(result.violation_count() >= 2); // At least depth and min samples violations
    }

    #[test]
    fn test_custom_constraint() {
        let mut max_usage = HashMap::new();
        max_usage.insert(0, 2);
        let custom_constraint = MaxFeatureUsageConstraint::new(max_usage);

        let mut split = SplitInfo::new();
        split.feature = 0;

        // First usage should be allowed
        assert!(custom_constraint.validate(&split, 1, 0.0, &[]).is_none());

        // Second usage should be allowed
        assert!(custom_constraint.validate(&split, 2, 0.0, &[0]).is_none());

        // Third usage should be rejected
        assert!(custom_constraint.validate(&split, 3, 0.0, &[0, 0]).is_some());
    }

    #[test]
    fn test_filter_candidate_features() {
        let mut constraints = InteractionConstraint::new();
        constraints.add_allowed_group(vec![0, 1]);
        constraints.add_allowed_group(vec![2, 3]);

        let mut manager = ConstraintManager::new();
        manager.set_interaction_constraints(constraints);

        let candidates = vec![0, 1, 2, 3];
        let tree_path = vec![0]; // Feature 0 is already used

        let filtered = manager.filter_candidate_features(&candidates, &tree_path);
        
        // Only feature 1 should be allowed (same group as 0)
        assert_eq!(filtered, vec![1]);
    }
}