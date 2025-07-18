//! Ranking metrics for model evaluation.
//!
//! This module provides comprehensive ranking metrics including
//! NDCG, MAP, MRR, precision/recall at K, and other measures for
//! evaluating learning-to-rank model performance.

use crate::core::{
    types::*,
    error::{LightGBMError, Result},
    traits::*,
};
use crate::metrics::utils;
use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Ranking metrics calculator for learning-to-rank models.
#[derive(Debug, Clone)]
pub struct RankingMetrics {
    /// Configuration for metric calculation
    config: RankingMetricsConfig,
    /// Cached metric values
    cached_metrics: HashMap<String, f64>,
}

/// Configuration for ranking metrics calculation.
#[derive(Debug, Clone)]
pub struct RankingMetricsConfig {
    /// K values for ranking metrics (e.g., NDCG@K, MAP@K)
    pub k_values: Vec<usize>,
    /// Whether to compute all metrics or just basic ones
    pub compute_all: bool,
    /// Whether to handle missing values
    pub handle_missing: bool,
    /// Tolerance for numerical stability
    pub epsilon: f64,
    /// Whether to use sample weights
    pub use_weights: bool,
    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,
    /// Confidence level for intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Whether to compute per-query metrics
    pub per_query_metrics: bool,
    /// Maximum relevance level for gain calculation
    pub max_relevance: f64,
    /// Gain function type for NDCG
    pub gain_function: GainFunction,
    /// Discount function type for DCG
    pub discount_function: DiscountFunction,
}

/// Gain function type for NDCG calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainFunction {
    /// Linear gain: relevance
    Linear,
    /// Exponential gain: 2^relevance - 1
    Exponential,
    /// Custom gain function
    Custom,
}

/// Discount function type for DCG calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscountFunction {
    /// Logarithmic discount: 1/log2(position + 1)
    Logarithmic,
    /// Linear discount: 1/position
    Linear,
    /// Custom discount function
    Custom,
}

impl Default for RankingMetricsConfig {
    fn default() -> Self {
        Self {
            k_values: vec![1, 3, 5, 10, 20],
            compute_all: true,
            handle_missing: true,
            epsilon: 1e-15,
            use_weights: false,
            compute_confidence_intervals: false,
            confidence_level: 0.95,
            per_query_metrics: false,
            max_relevance: 4.0,
            gain_function: GainFunction::Exponential,
            discount_function: DiscountFunction::Logarithmic,
        }
    }
}

impl RankingMetrics {
    /// Create a new ranking metrics calculator.
    pub fn new(config: RankingMetricsConfig) -> Self {
        Self {
            config,
            cached_metrics: HashMap::new(),
        }
    }

    /// Calculate all ranking metrics.
    pub fn calculate(
        &mut self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        groups: &ArrayView1<DataSize>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<RankingMetricsResult> {
        self.validate_inputs(predictions, targets, groups, weights)?;
        
        let mut result = RankingMetricsResult::new();
        
        // Parse groups to get query boundaries
        let query_ranges = self.parse_query_groups(groups)?;
        
        // Calculate NDCG at different K values
        for &k in &self.config.k_values {
            let ndcg = self.calculate_ndcg_at_k(predictions, targets, &query_ranges, k)?;
            result.ndcg_at_k.insert(k, ndcg);
        }
        
        // Calculate MAP at different K values
        for &k in &self.config.k_values {
            let map = self.calculate_map_at_k(predictions, targets, &query_ranges, k)?;
            result.map_at_k.insert(k, map);
        }
        
        // Calculate MRR
        result.mrr = self.calculate_mrr(predictions, targets, &query_ranges)?;
        
        // Calculate precision and recall at different K values
        for &k in &self.config.k_values {
            let precision = self.calculate_precision_at_k(predictions, targets, &query_ranges, k)?;
            let recall = self.calculate_recall_at_k(predictions, targets, &query_ranges, k)?;
            result.precision_at_k.insert(k, precision);
            result.recall_at_k.insert(k, recall);
        }
        
        if self.config.compute_all {
            result.kendall_tau = self.calculate_kendall_tau(predictions, targets, &query_ranges)?;
            result.spearman_rho = self.calculate_spearman_rho(predictions, targets, &query_ranges)?;
            result.average_precision = self.calculate_average_precision(predictions, targets, &query_ranges)?;
            result.reciprocal_rank = self.calculate_reciprocal_rank(predictions, targets, &query_ranges)?;
            result.dcg = self.calculate_dcg(predictions, targets, &query_ranges)?;
            result.idcg = self.calculate_idcg(targets, &query_ranges)?;
            result.err = self.calculate_err(predictions, targets, &query_ranges)?;
            result.rank_correlation = self.calculate_rank_correlation(predictions, targets, &query_ranges)?;
        }
        
        // Calculate per-query metrics if requested
        if self.config.per_query_metrics {
            result.per_query_ndcg = Some(self.calculate_per_query_ndcg(predictions, targets, &query_ranges)?);
            result.per_query_map = Some(self.calculate_per_query_map(predictions, targets, &query_ranges)?);
            result.per_query_mrr = Some(self.calculate_per_query_mrr(predictions, targets, &query_ranges)?);
        }
        
        // Calculate sample size and query count
        result.sample_size = predictions.len();
        result.num_queries = query_ranges.len();
        
        // Calculate confidence intervals if requested
        if self.config.compute_confidence_intervals {
            result.confidence_intervals = Some(self.calculate_confidence_intervals(
                predictions, targets, &query_ranges, weights
            )?);
        }
        
        Ok(result)
    }

    /// Validate input arrays.
    fn validate_inputs(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        groups: &ArrayView1<DataSize>,
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<()> {
        if predictions.len() != targets.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions: {}", predictions.len()),
                format!("targets: {}", targets.len()),
            ));
        }
        
        if predictions.len() != groups.len() {
            return Err(LightGBMError::dimension_mismatch(
                format!("predictions: {}", predictions.len()),
                format!("groups: {}", groups.len()),
            ));
        }
        
        if let Some(w) = weights {
            if w.len() != predictions.len() {
                return Err(LightGBMError::dimension_mismatch(
                    format!("weights: {}", w.len()),
                    format!("predictions: {}", predictions.len()),
                ));
            }
        }
        
        if predictions.is_empty() {
            return Err(LightGBMError::config("Empty input arrays"));
        }
        
        Ok(())
    }

    /// Parse query groups to get query boundaries.
    fn parse_query_groups(&self, groups: &ArrayView1<DataSize>) -> Result<Vec<RankingGroup>> {
        let mut query_ranges = Vec::new();
        let mut current_group = groups[0];
        let mut start_idx = 0;
        
        for (i, &group) in groups.iter().enumerate() {
            if group != current_group {
                query_ranges.push(RankingGroup {
                    query_id: current_group as usize,
                    start_idx,
                    end_idx: i,
                });
                current_group = group;
                start_idx = i;
            }
        }
        
        // Add the last group
        query_ranges.push(RankingGroup {
            query_id: current_group as usize,
            start_idx,
            end_idx: groups.len(),
        });
        
        Ok(query_ranges)
    }

    /// Calculate NDCG at K.
    fn calculate_ndcg_at_k(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
        k: usize,
    ) -> Result<f64> {
        let mut total_ndcg = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let dcg = self.calculate_query_dcg(&query_predictions, &query_targets, k)?;
            let idcg = self.calculate_query_idcg(&query_targets, k)?;
            
            if idcg > self.config.epsilon {
                total_ndcg += dcg / idcg;
                valid_queries += 1;
            }
        }
        
        if valid_queries > 0 {
            Ok(total_ndcg / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate MAP at K.
    fn calculate_map_at_k(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
        k: usize,
    ) -> Result<f64> {
        let mut total_ap = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let ap = self.calculate_query_average_precision(&query_predictions, &query_targets, k)?;
            total_ap += ap;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_ap / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate MRR.
    fn calculate_mrr(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        let mut total_rr = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let rr = self.calculate_query_reciprocal_rank(&query_predictions, &query_targets)?;
            total_rr += rr;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_rr / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate precision at K.
    fn calculate_precision_at_k(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
        k: usize,
    ) -> Result<f64> {
        let mut total_precision = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let precision = self.calculate_query_precision_at_k(&query_predictions, &query_targets, k)?;
            total_precision += precision;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_precision / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate recall at K.
    fn calculate_recall_at_k(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
        k: usize,
    ) -> Result<f64> {
        let mut total_recall = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let recall = self.calculate_query_recall_at_k(&query_predictions, &query_targets, k)?;
            total_recall += recall;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_recall / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate DCG for a single query.
    fn calculate_query_dcg(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> Result<f64> {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut dcg = 0.0;
        for (i, (_, relevance)) in items.iter().take(k).enumerate() {
            let gain = self.calculate_gain(*relevance);
            let discount = self.calculate_discount(i + 1);
            dcg += gain * discount;
        }

        Ok(dcg)
    }

    /// Calculate IDCG for a single query.
    fn calculate_query_idcg(
        &self,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> Result<f64> {
        let mut relevances: Vec<f32> = targets.iter().copied().collect();
        relevances.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let mut idcg = 0.0;
        for (i, &relevance) in relevances.iter().take(k).enumerate() {
            let gain = self.calculate_gain(relevance);
            let discount = self.calculate_discount(i + 1);
            idcg += gain * discount;
        }

        Ok(idcg)
    }

    /// Calculate average precision for a single query.
    fn calculate_query_average_precision(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> Result<f64> {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut ap = 0.0;
        let mut relevant_count = 0;
        let total_relevant = targets.iter().filter(|&&t| t > 0.0).count();

        if total_relevant == 0 {
            return Ok(0.0);
        }

        for (i, (_, relevance)) in items.iter().take(k).enumerate() {
            if *relevance > 0.0 {
                relevant_count += 1;
                let precision = relevant_count as f64 / (i + 1) as f64;
                ap += precision;
            }
        }

        Ok(ap / total_relevant as f64)
    }

    /// Calculate reciprocal rank for a single query.
    fn calculate_query_reciprocal_rank(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for (i, (_, relevance)) in items.iter().enumerate() {
            if *relevance > 0.0 {
                return Ok(1.0 / (i + 1) as f64);
            }
        }

        Ok(0.0)
    }

    /// Calculate precision at K for a single query.
    fn calculate_query_precision_at_k(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> Result<f64> {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let relevant_count = items.iter()
            .take(k)
            .filter(|(_, relevance)| *relevance > 0.0)
            .count();

        Ok(relevant_count as f64 / k.min(items.len()) as f64)
    }

    /// Calculate recall at K for a single query.
    fn calculate_query_recall_at_k(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        k: usize,
    ) -> Result<f64> {
        let total_relevant = targets.iter().filter(|&&t| t > 0.0).count();
        
        if total_relevant == 0 {
            return Ok(0.0);
        }

        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let relevant_retrieved = items.iter()
            .take(k)
            .filter(|(_, relevance)| *relevance > 0.0)
            .count();

        Ok(relevant_retrieved as f64 / total_relevant as f64)
    }

    /// Calculate gain based on relevance.
    fn calculate_gain(&self, relevance: f32) -> f64 {
        match self.config.gain_function {
            GainFunction::Linear => relevance as f64,
            GainFunction::Exponential => (2.0_f64.powf(relevance as f64) - 1.0),
            GainFunction::Custom => relevance as f64, // Placeholder for custom implementation
        }
    }

    /// Calculate discount based on position.
    fn calculate_discount(&self, position: usize) -> f64 {
        match self.config.discount_function {
            DiscountFunction::Logarithmic => 1.0 / (position as f64 + 1.0).log2(),
            DiscountFunction::Linear => 1.0 / position as f64,
            DiscountFunction::Custom => 1.0 / (position as f64 + 1.0).log2(), // Placeholder for custom implementation
        }
    }

    /// Calculate Kendall's tau.
    fn calculate_kendall_tau(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        let mut total_tau = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let tau = self.calculate_query_kendall_tau(&query_predictions, &query_targets)?;
            total_tau += tau;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_tau / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate Kendall's tau for a single query.
    fn calculate_query_kendall_tau(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        let n = predictions.len();
        if n < 2 {
            return Ok(0.0);
        }

        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let pred_diff = predictions[i] - predictions[j];
                let target_diff = targets[i] - targets[j];
                
                if pred_diff * target_diff > 0.0 {
                    concordant += 1;
                } else if pred_diff * target_diff < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            Ok(0.0)
        } else {
            Ok((concordant as f64 - discordant as f64) / total_pairs as f64)
        }
    }

    /// Calculate Spearman's rho.
    fn calculate_spearman_rho(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        let mut total_rho = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let rho = self.calculate_query_spearman_rho(&query_predictions, &query_targets)?;
            total_rho += rho;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_rho / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate Spearman's rho for a single query.
    fn calculate_query_spearman_rho(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        let n = predictions.len();
        if n < 2 {
            return Ok(0.0);
        }

        let pred_ranks = self.calculate_ranks(predictions);
        let target_ranks = self.calculate_ranks(targets);

        let mut sum_d_squared = 0.0;
        for i in 0..n {
            let d = pred_ranks[i] - target_ranks[i];
            sum_d_squared += d * d;
        }

        let rho = 1.0 - (6.0 * sum_d_squared) / (n * (n * n - 1)) as f64;
        Ok(rho)
    }

    /// Calculate ranks for an array.
    fn calculate_ranks(&self, values: &ArrayView1<f32>) -> Vec<f64> {
        let n = values.len();
        let mut indexed_values: Vec<(usize, f32)> = values.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut ranks = vec![0.0; n];
        for (rank, (original_index, _)) in indexed_values.iter().enumerate() {
            ranks[*original_index] = (rank + 1) as f64;
        }

        ranks
    }

    /// Calculate average precision across all queries.
    fn calculate_average_precision(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        let mut total_ap = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let ap = self.calculate_query_average_precision(&query_predictions, &query_targets, usize::MAX)?;
            total_ap += ap;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_ap / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate reciprocal rank across all queries.
    fn calculate_reciprocal_rank(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        self.calculate_mrr(predictions, targets, query_ranges)
    }

    /// Calculate DCG across all queries.
    fn calculate_dcg(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        let mut total_dcg = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let dcg = self.calculate_query_dcg(&query_predictions, &query_targets, usize::MAX)?;
            total_dcg += dcg;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_dcg / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate IDCG across all queries.
    fn calculate_idcg(
        &self,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        let mut total_idcg = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let idcg = self.calculate_query_idcg(&query_targets, usize::MAX)?;
            total_idcg += idcg;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_idcg / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate Expected Reciprocal Rank (ERR).
    fn calculate_err(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        let mut total_err = 0.0;
        let mut valid_queries = 0;
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let err = self.calculate_query_err(&query_predictions, &query_targets)?;
            total_err += err;
            valid_queries += 1;
        }
        
        if valid_queries > 0 {
            Ok(total_err / valid_queries as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate ERR for a single query.
    fn calculate_query_err(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
    ) -> Result<f64> {
        let mut items: Vec<(f32, f32)> = predictions.iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred, target))
            .collect();

        // Sort by predictions in descending order
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut err = 0.0;
        let mut prob_stop = 0.0;

        for (i, (_, relevance)) in items.iter().enumerate() {
            let utility = (2.0_f64.powf(*relevance as f64) - 1.0) / 2.0_f64.powf(self.config.max_relevance);
            let prob_examine = 1.0 - prob_stop;
            
            err += prob_examine * utility / (i + 1) as f64;
            prob_stop += prob_examine * utility;
            
            if prob_stop >= 1.0 - self.config.epsilon {
                break;
            }
        }

        Ok(err)
    }

    /// Calculate rank correlation.
    fn calculate_rank_correlation(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<f64> {
        self.calculate_spearman_rho(predictions, targets, query_ranges)
    }

    /// Calculate per-query NDCG.
    fn calculate_per_query_ndcg(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<Vec<f64>> {
        let mut per_query_ndcg = Vec::new();
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let dcg = self.calculate_query_dcg(&query_predictions, &query_targets, usize::MAX)?;
            let idcg = self.calculate_query_idcg(&query_targets, usize::MAX)?;
            
            let ndcg = if idcg > self.config.epsilon {
                dcg / idcg
            } else {
                0.0
            };
            
            per_query_ndcg.push(ndcg);
        }
        
        Ok(per_query_ndcg)
    }

    /// Calculate per-query MAP.
    fn calculate_per_query_map(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<Vec<f64>> {
        let mut per_query_map = Vec::new();
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let ap = self.calculate_query_average_precision(&query_predictions, &query_targets, usize::MAX)?;
            per_query_map.push(ap);
        }
        
        Ok(per_query_map)
    }

    /// Calculate per-query MRR.
    fn calculate_per_query_mrr(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
    ) -> Result<Vec<f64>> {
        let mut per_query_mrr = Vec::new();
        
        for group in query_ranges {
            let query_predictions = predictions.slice(s![group.start_idx..group.end_idx]);
            let query_targets = targets.slice(s![group.start_idx..group.end_idx]);
            
            let rr = self.calculate_query_reciprocal_rank(&query_predictions, &query_targets)?;
            per_query_mrr.push(rr);
        }
        
        Ok(per_query_mrr)
    }

    /// Calculate confidence intervals for ranking metrics.
    fn calculate_confidence_intervals(
        &self,
        predictions: &ArrayView1<Score>,
        targets: &ArrayView1<Label>,
        query_ranges: &[RankingGroup],
        weights: Option<&ArrayView1<Label>>,
    ) -> Result<RankingConfidenceIntervals> {
        // Simple bootstrap confidence intervals
        let n_bootstrap = 1000;
        let mut bootstrap_ndcg = Vec::new();
        let mut bootstrap_map = Vec::new();
        
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_entropy();
        
        for _ in 0..n_bootstrap {
            let mut bootstrap_query_ranges = query_ranges.to_vec();
            bootstrap_query_ranges.shuffle(&mut rng);
            
            // Calculate NDCG@5 for bootstrap sample
            let ndcg = self.calculate_ndcg_at_k(predictions, targets, &bootstrap_query_ranges, 5)?;
            bootstrap_ndcg.push(ndcg);
            
            // Calculate MAP@5 for bootstrap sample
            let map = self.calculate_map_at_k(predictions, targets, &bootstrap_query_ranges, 5)?;
            bootstrap_map.push(map);
        }
        
        bootstrap_ndcg.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bootstrap_map.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let alpha = 1.0 - self.config.confidence_level;
        let lower_percentile = (alpha / 2.0 * n_bootstrap as f64) as usize;
        let upper_percentile = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
        
        Ok(RankingConfidenceIntervals {
            confidence_level: self.config.confidence_level,
            ndcg_interval: (bootstrap_ndcg[lower_percentile], bootstrap_ndcg[upper_percentile]),
            map_interval: (bootstrap_map[lower_percentile], bootstrap_map[upper_percentile]),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &RankingMetricsConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: RankingMetricsConfig) {
        self.config = config;
        self.cached_metrics.clear();
    }
}

/// Result of ranking metrics calculation.
#[derive(Debug, Clone)]
pub struct RankingMetricsResult {
    /// NDCG at different K values
    pub ndcg_at_k: HashMap<usize, f64>,
    /// MAP at different K values
    pub map_at_k: HashMap<usize, f64>,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Precision at different K values
    pub precision_at_k: HashMap<usize, f64>,
    /// Recall at different K values
    pub recall_at_k: HashMap<usize, f64>,
    /// Kendall's tau correlation
    pub kendall_tau: f64,
    /// Spearman's rho correlation
    pub spearman_rho: f64,
    /// Average Precision
    pub average_precision: f64,
    /// Reciprocal Rank
    pub reciprocal_rank: f64,
    /// Discounted Cumulative Gain
    pub dcg: f64,
    /// Ideal Discounted Cumulative Gain
    pub idcg: f64,
    /// Expected Reciprocal Rank
    pub err: f64,
    /// Rank correlation
    pub rank_correlation: f64,
    /// Per-query NDCG
    pub per_query_ndcg: Option<Vec<f64>>,
    /// Per-query MAP
    pub per_query_map: Option<Vec<f64>>,
    /// Per-query MRR
    pub per_query_mrr: Option<Vec<f64>>,
    /// Sample size
    pub sample_size: usize,
    /// Number of queries
    pub num_queries: usize,
    /// Confidence intervals (if computed)
    pub confidence_intervals: Option<RankingConfidenceIntervals>,
}

impl RankingMetricsResult {
    /// Create a new result.
    pub fn new() -> Self {
        Self {
            ndcg_at_k: HashMap::new(),
            map_at_k: HashMap::new(),
            mrr: 0.0,
            precision_at_k: HashMap::new(),
            recall_at_k: HashMap::new(),
            kendall_tau: 0.0,
            spearman_rho: 0.0,
            average_precision: 0.0,
            reciprocal_rank: 0.0,
            dcg: 0.0,
            idcg: 0.0,
            err: 0.0,
            rank_correlation: 0.0,
            per_query_ndcg: None,
            per_query_map: None,
            per_query_mrr: None,
            sample_size: 0,
            num_queries: 0,
            confidence_intervals: None,
        }
    }

    /// Get the primary metric (NDCG@5).
    pub fn primary_metric(&self) -> f64 {
        self.ndcg_at_k.get(&5).copied().unwrap_or(0.0)
    }

    /// Get a summary of the metrics.
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Ranking Metrics Summary (n={}, queries={}):\n", self.sample_size, self.num_queries));
        
        // NDCG at different K values
        let mut k_values: Vec<usize> = self.ndcg_at_k.keys().copied().collect();
        k_values.sort();
        for k in k_values {
            if let Some(ndcg) = self.ndcg_at_k.get(&k) {
                summary.push_str(&format!("  NDCG@{}: {:.6}\n", k, ndcg));
            }
        }
        
        // MAP at different K values
        let mut k_values: Vec<usize> = self.map_at_k.keys().copied().collect();
        k_values.sort();
        for k in k_values {
            if let Some(map) = self.map_at_k.get(&k) {
                summary.push_str(&format!("  MAP@{}: {:.6}\n", k, map));
            }
        }
        
        summary.push_str(&format!("  MRR: {:.6}\n", self.mrr));
        summary.push_str(&format!("  Kendall's Tau: {:.6}\n", self.kendall_tau));
        summary.push_str(&format!("  Spearman's Rho: {:.6}\n", self.spearman_rho));
        summary.push_str(&format!("  Average Precision: {:.6}\n", self.average_precision));
        summary.push_str(&format!("  ERR: {:.6}\n", self.err));
        
        if let Some(ref ci) = self.confidence_intervals {
            summary.push_str(&format!("\nConfidence Intervals ({:.0}%):\n", ci.confidence_level * 100.0));
            summary.push_str(&format!("  NDCG@5: [{:.6}, {:.6}]\n", ci.ndcg_interval.0, ci.ndcg_interval.1));
            summary.push_str(&format!("  MAP@5: [{:.6}, {:.6}]\n", ci.map_interval.0, ci.map_interval.1));
        }
        
        summary
    }
}

/// Information about a ranking group (query).
#[derive(Debug, Clone)]
pub struct RankingGroup {
    /// Query ID
    pub query_id: usize,
    /// Start index in the data
    pub start_idx: usize,
    /// End index in the data
    pub end_idx: usize,
}

/// Individual ranking item.
#[derive(Debug, Clone)]
pub struct RankingItem {
    /// Query ID
    pub query_id: usize,
    /// Document ID
    pub doc_id: usize,
    /// Relevance score
    pub relevance: f64,
    /// Predicted score
    pub prediction: f64,
}

/// Confidence intervals for ranking metrics.
#[derive(Debug, Clone)]
pub struct RankingConfidenceIntervals {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// NDCG confidence interval
    pub ndcg_interval: (f64, f64),
    /// MAP confidence interval
    pub map_interval: (f64, f64),
}

/// Builder for ranking metrics configuration.
#[derive(Debug)]
pub struct RankingMetricsConfigBuilder {
    config: RankingMetricsConfig,
}

impl RankingMetricsConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: RankingMetricsConfig::default(),
        }
    }

    /// Set the K values.
    pub fn k_values(mut self, k_values: Vec<usize>) -> Self {
        self.config.k_values = k_values;
        self
    }

    /// Set whether to compute all metrics.
    pub fn compute_all(mut self, compute_all: bool) -> Self {
        self.config.compute_all = compute_all;
        self
    }

    /// Set whether to handle missing values.
    pub fn handle_missing(mut self, handle_missing: bool) -> Self {
        self.config.handle_missing = handle_missing;
        self
    }

    /// Set the epsilon value.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set whether to use weights.
    pub fn use_weights(mut self, use_weights: bool) -> Self {
        self.config.use_weights = use_weights;
        self
    }

    /// Set whether to compute confidence intervals.
    pub fn compute_confidence_intervals(mut self, compute_ci: bool) -> Self {
        self.config.compute_confidence_intervals = compute_ci;
        self
    }

    /// Set the confidence level.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level;
        self
    }

    /// Set whether to compute per-query metrics.
    pub fn per_query_metrics(mut self, per_query: bool) -> Self {
        self.config.per_query_metrics = per_query;
        self
    }

    /// Set the maximum relevance level.
    pub fn max_relevance(mut self, max_relevance: f64) -> Self {
        self.config.max_relevance = max_relevance;
        self
    }

    /// Set the gain function.
    pub fn gain_function(mut self, gain_function: GainFunction) -> Self {
        self.config.gain_function = gain_function;
        self
    }

    /// Set the discount function.
    pub fn discount_function(mut self, discount_function: DiscountFunction) -> Self {
        self.config.discount_function = discount_function;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> RankingMetricsConfig {
        self.config
    }
}

impl Default for RankingMetricsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RankingMetrics {
    fn default() -> Self {
        Self::new(RankingMetricsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_ranking_metrics_config_default() {
        let config = RankingMetricsConfig::default();
        assert_eq!(config.k_values, vec![1, 3, 5, 10, 20]);
        assert!(config.compute_all);
        assert!(config.handle_missing);
        assert_eq!(config.epsilon, 1e-15);
        assert!(!config.use_weights);
        assert!(!config.compute_confidence_intervals);
        assert_eq!(config.confidence_level, 0.95);
        assert!(!config.per_query_metrics);
        assert_eq!(config.max_relevance, 4.0);
        assert_eq!(config.gain_function, GainFunction::Exponential);
        assert_eq!(config.discount_function, DiscountFunction::Logarithmic);
    }

    #[test]
    fn test_ranking_metrics_config_builder() {
        let config = RankingMetricsConfigBuilder::new()
            .k_values(vec![1, 5, 10])
            .compute_all(false)
            .handle_missing(false)
            .epsilon(1e-10)
            .use_weights(true)
            .compute_confidence_intervals(true)
            .confidence_level(0.99)
            .per_query_metrics(true)
            .max_relevance(5.0)
            .gain_function(GainFunction::Linear)
            .discount_function(DiscountFunction::Linear)
            .build();

        assert_eq!(config.k_values, vec![1, 5, 10]);
        assert!(!config.compute_all);
        assert!(!config.handle_missing);
        assert_eq!(config.epsilon, 1e-10);
        assert!(config.use_weights);
        assert!(config.compute_confidence_intervals);
        assert_eq!(config.confidence_level, 0.99);
        assert!(config.per_query_metrics);
        assert_eq!(config.max_relevance, 5.0);
        assert_eq!(config.gain_function, GainFunction::Linear);
        assert_eq!(config.discount_function, DiscountFunction::Linear);
    }

    #[test]
    fn test_parse_query_groups() {
        let mut metrics = RankingMetrics::default();
        let groups = Array1::from_vec(vec![0, 0, 0, 1, 1, 2, 2, 2, 2]);
        
        let query_ranges = metrics.parse_query_groups(&groups.view()).unwrap();
        assert_eq!(query_ranges.len(), 3);
        assert_eq!(query_ranges[0].query_id, 0);
        assert_eq!(query_ranges[0].start_idx, 0);
        assert_eq!(query_ranges[0].end_idx, 3);
        assert_eq!(query_ranges[1].query_id, 1);
        assert_eq!(query_ranges[1].start_idx, 3);
        assert_eq!(query_ranges[1].end_idx, 5);
        assert_eq!(query_ranges[2].query_id, 2);
        assert_eq!(query_ranges[2].start_idx, 5);
        assert_eq!(query_ranges[2].end_idx, 9);
    }

    #[test]
    fn test_calculate_query_dcg() {
        let mut metrics = RankingMetrics::default();
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![3.0, 2.0, 1.0, 0.0]);
        
        let dcg = metrics.calculate_query_dcg(&predictions.view(), &targets.view(), 3).unwrap();
        assert!(dcg > 0.0);
    }

    #[test]
    fn test_calculate_query_idcg() {
        let mut metrics = RankingMetrics::default();
        let targets = Array1::from_vec(vec![3.0, 2.0, 1.0, 0.0]);
        
        let idcg = metrics.calculate_query_idcg(&targets.view(), 3).unwrap();
        assert!(idcg > 0.0);
    }

    #[test]
    fn test_calculate_query_average_precision() {
        let mut metrics = RankingMetrics::default();
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let ap = metrics.calculate_query_average_precision(&predictions.view(), &targets.view(), 4).unwrap();
        assert!(ap > 0.0);
        assert!(ap <= 1.0);
    }

    #[test]
    fn test_calculate_query_reciprocal_rank() {
        let mut metrics = RankingMetrics::default();
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        
        let rr = metrics.calculate_query_reciprocal_rank(&predictions.view(), &targets.view()).unwrap();
        assert!((rr - 0.5).abs() < 1e-10); // Should be 1/2 = 0.5
    }

    #[test]
    fn test_calculate_query_precision_at_k() {
        let mut metrics = RankingMetrics::default();
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let precision = metrics.calculate_query_precision_at_k(&predictions.view(), &targets.view(), 2).unwrap();
        assert!((precision - 1.0).abs() < 1e-10); // Should be 1.0 (2/2)
    }

    #[test]
    fn test_calculate_query_recall_at_k() {
        let mut metrics = RankingMetrics::default();
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6]);
        let targets = Array1::from_vec(vec![1.0, 1.0, 0.0, 1.0]);
        
        let recall = metrics.calculate_query_recall_at_k(&predictions.view(), &targets.view(), 2).unwrap();
        assert!((recall - 2.0/3.0).abs() < 1e-10); // Should be 2/3 (2 out of 3 relevant retrieved)
    }

    #[test]
    fn test_gain_functions() {
        let mut metrics = RankingMetrics::default();
        
        // Test exponential gain
        assert_eq!(metrics.calculate_gain(0.0), 0.0);
        assert_eq!(metrics.calculate_gain(1.0), 1.0);
        assert_eq!(metrics.calculate_gain(2.0), 3.0);
        assert_eq!(metrics.calculate_gain(3.0), 7.0);
        
        // Test linear gain
        metrics.config.gain_function = GainFunction::Linear;
        assert_eq!(metrics.calculate_gain(0.0), 0.0);
        assert_eq!(metrics.calculate_gain(1.0), 1.0);
        assert_eq!(metrics.calculate_gain(2.0), 2.0);
        assert_eq!(metrics.calculate_gain(3.0), 3.0);
    }

    #[test]
    fn test_discount_functions() {
        let mut metrics = RankingMetrics::default();
        
        // Test logarithmic discount
        assert_eq!(metrics.calculate_discount(1), 1.0);
        assert!((metrics.calculate_discount(2) - 1.0 / 2.0_f64.log2()).abs() < 1e-10);
        
        // Test linear discount
        metrics.config.discount_function = DiscountFunction::Linear;
        assert_eq!(metrics.calculate_discount(1), 1.0);
        assert_eq!(metrics.calculate_discount(2), 0.5);
        assert!((metrics.calculate_discount(3) - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_ranks() {
        let mut metrics = RankingMetrics::default();
        let values = Array1::from_vec(vec![0.9, 0.7, 0.8, 0.6]);
        
        let ranks = metrics.calculate_ranks(&values.view());
        assert_eq!(ranks, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_validation_errors() {
        let mut metrics = RankingMetrics::default();
        let predictions = Array1::from_vec(vec![0.9, 0.8]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let groups = Array1::from_vec(vec![0, 0]);
        
        let result = metrics.calculate(&predictions.view(), &targets.view(), &groups.view(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ranking_result_summary() {
        let mut result = RankingMetricsResult::new();
        result.sample_size = 100;
        result.num_queries = 20;
        result.ndcg_at_k.insert(5, 0.85);
        result.map_at_k.insert(5, 0.75);
        result.mrr = 0.80;
        result.kendall_tau = 0.65;
        result.spearman_rho = 0.70;
        result.average_precision = 0.78;
        result.err = 0.82;
        
        let summary = result.summary();
        assert!(summary.contains("Ranking Metrics Summary"));
        assert!(summary.contains("n=100"));
        assert!(summary.contains("queries=20"));
        assert!(summary.contains("NDCG@5: 0.850000"));
        assert!(summary.contains("MAP@5: 0.750000"));
        assert!(summary.contains("MRR: 0.800000"));
        assert!(summary.contains("Kendall's Tau: 0.650000"));
        assert!(summary.contains("Spearman's Rho: 0.700000"));
        assert!(summary.contains("Average Precision: 0.780000"));
        assert!(summary.contains("ERR: 0.820000"));
    }

    #[test]
    fn test_ranking_group_creation() {
        let group = RankingGroup {
            query_id: 42,
            start_idx: 10,
            end_idx: 20,
        };
        
        assert_eq!(group.query_id, 42);
        assert_eq!(group.start_idx, 10);
        assert_eq!(group.end_idx, 20);
    }

    #[test]
    fn test_ranking_item_creation() {
        let item = RankingItem {
            query_id: 1,
            doc_id: 100,
            relevance: 2.5,
            prediction: 0.8,
        };
        
        assert_eq!(item.query_id, 1);
        assert_eq!(item.doc_id, 100);
        assert_eq!(item.relevance, 2.5);
        assert_eq!(item.prediction, 0.8);
    }
}