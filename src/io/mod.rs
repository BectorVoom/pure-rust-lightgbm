//! Input/Output operations and data serialization for LightGBM.
//!
//! This module provides functionality for reading, writing, and serializing
//! LightGBM models and data structures.

pub mod bin;
pub mod dataset;
pub mod dense_bin;
pub mod multi_val_dense_bin;
pub mod multi_val_sparse_bin;
pub mod sparse_bin;
pub mod train_share_states;
pub mod tree;