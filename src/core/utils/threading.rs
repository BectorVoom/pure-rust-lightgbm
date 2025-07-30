//! LightGBM Threading utilities implemented in Rust using Rayon
//!
//! This module provides parallel threading utilities equivalent to the C++ LightGBM implementation.
//! Key components:
//! - BlockInfo and BlockInfoForceSize for workload distribution
//! - Parallel For loop equivalent
//! - ParallelPartitionRunner for in-place partitioning with optional two-buffer support

use crate::core::utils::openmp_wrapper::omp_num_threads;
use rayon::prelude::*;
use std::cmp::min;
use std::panic::{self, AssertUnwindSafe};

/// Memory alignment size constant - equivalent to C++ kAlignedSize
const K_ALIGNED_SIZE: usize = 32;

/// SIZE_ALIGNED macro equivalent - ensures 32-byte memory alignment for SIMD optimization
/// Equivalent to C++ macro: #define SIZE_ALIGNED(t) ((t) + kAlignedSize - 1) / kAlignedSize * kAlignedSize
fn size_aligned(t: usize) -> usize {
    ((t + K_ALIGNED_SIZE - 1) / K_ALIGNED_SIZE) * K_ALIGNED_SIZE
}

/// Threading utility struct to match C++ LightGBM::Threading class
#[derive(Debug)]
pub struct Threading;

impl Threading {
    /// Compute number of blocks and block size (equivalent to C++ BlockInfo)
    pub fn block_info<IndexT: Copy + Into<usize> + From<usize>>(
        num_threads: i32,
        cnt: IndexT,
        min_cnt_per_block: IndexT,
        out_nblock: &mut i32,
        block_size: &mut IndexT,
    ) {
        let cnt_usize = cnt.into();
        let min_usize = min_cnt_per_block.into();
        let n = min(
            num_threads,
            ((cnt_usize + min_usize - 1) / min_usize) as i32,
        );
        *out_nblock = n;
        if n > 1 {
            let size = (cnt_usize + n as usize - 1) / n as usize;
            // Apply SIZE_ALIGNED for 32-byte memory alignment (SIMD optimization)
            let aligned = size_aligned(size);
            *block_size = IndexT::from(aligned);
        } else {
            *block_size = cnt;
        }
    }

    /// Overload with default thread count (equivalent to C++ template overload)
    pub fn block_info_default<IndexT: Copy + Into<usize> + From<usize>>(
        cnt: IndexT,
        min_cnt_per_block: IndexT,
        out_nblock: &mut i32,
        block_size: &mut IndexT,
    ) {
        Self::block_info(
            omp_num_threads() as i32,
            cnt,
            min_cnt_per_block,
            out_nblock,
            block_size,
        );
    }

    /// Compute blocks forcing block size to be multiple of min_cnt_per_block (equivalent to C++ BlockInfoForceSize)
    pub fn block_info_force_size<IndexT: Copy + Into<usize> + From<usize>>(
        num_threads: i32,
        cnt: IndexT,
        min_cnt_per_block: IndexT,
        out_nblock: &mut i32,
        block_size: &mut IndexT,
    ) {
        let cnt_usize = cnt.into();
        let min_usize = min_cnt_per_block.into();
        let n = min(
            num_threads,
            ((cnt_usize + min_usize - 1) / min_usize) as i32,
        );
        *out_nblock = n;
        if n > 1 {
            let mut size = (cnt_usize + n as usize - 1) / n as usize;
            // force the block size to the times of min_cnt_per_block
            size = ((size + min_usize - 1) / min_usize) * min_usize;
            *block_size = IndexT::from(size);
        } else {
            *block_size = cnt;
        }
    }

    /// Overload with default thread count (equivalent to C++ template overload)
    pub fn block_info_force_size_default<IndexT: Copy + Into<usize> + From<usize>>(
        cnt: IndexT,
        min_cnt_per_block: IndexT,
        out_nblock: &mut i32,
        block_size: &mut IndexT,
    ) {
        Self::block_info_force_size(
            omp_num_threads() as i32,
            cnt,
            min_cnt_per_block,
            out_nblock,
            block_size,
        );
    }

    /// Parallel For: splits [start, end) into blocks and executes inner_fun(thread_id, start, end)
    /// Equivalent to C++ Threading::For template function
    pub fn for_loop<IndexT, F>(
        start: IndexT,
        end: IndexT,
        min_block_size: IndexT,
        inner_fun: F,
    ) -> i32
    where
        IndexT: Copy + Into<usize> + From<usize> + Send + Sync,
        F: Fn(i32, IndexT, IndexT) + Send + Sync,
    {
        let num_inner = end.into() - start.into();
        let mut n_block = 1i32;
        let mut block_size = IndexT::from(0);

        Self::block_info(
            omp_num_threads() as i32,
            IndexT::from(num_inner),
            min_block_size,
            &mut n_block,
            &mut block_size,
        );

        // OMP_INIT_EX() equivalent - catch unwinds from threads
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            (0..n_block).into_par_iter().for_each(|i| {
                let inner_start = start.into() + block_size.into() * i as usize;
                let inner_end = min(end.into(), inner_start + block_size.into());
                if inner_start < inner_end {
                    inner_fun(i, IndexT::from(inner_start), IndexT::from(inner_end));
                }
            });
        }));

        // OMP_THROW_EX() equivalent
        if let Err(payload) = result {
            panic::resume_unwind(payload);
        }

        n_block
    }
}

/// Runner for parallel partitioning - equivalent to C++ ParallelPartitionRunner
#[derive(Debug)]
pub struct ParallelPartitionRunner<IndexT: Copy + Into<usize> + From<usize>> {
    num_threads_: i32,
    min_block_size_: IndexT,
    left_: Vec<IndexT>,
    right_: Vec<IndexT>,
    offsets_: Vec<IndexT>,
    left_cnts_: Vec<IndexT>,
    right_cnts_: Vec<IndexT>,
    left_write_pos_: Vec<IndexT>,
    right_write_pos_: Vec<IndexT>,
}

impl<IndexT> ParallelPartitionRunner<IndexT>
where
    IndexT: Copy + Into<usize> + From<usize> + Send + Sync,
{
    /// Constructor - equivalent to C++ constructor
    pub fn new(num_data: IndexT, min_block_size: IndexT) -> Self {
        let num_threads = omp_num_threads() as i32;
        let ndata = num_data.into();
        ParallelPartitionRunner {
            num_threads_: num_threads,
            min_block_size_: min_block_size,
            left_: vec![IndexT::from(0); ndata],
            right_: vec![IndexT::from(0); ndata],
            offsets_: vec![IndexT::from(0); num_threads as usize],
            left_cnts_: vec![IndexT::from(0); num_threads as usize],
            right_cnts_: vec![IndexT::from(0); num_threads as usize],
            left_write_pos_: vec![IndexT::from(0); num_threads as usize],
            right_write_pos_: vec![IndexT::from(0); num_threads as usize],
        }
    }

    /// Resize internal buffers - equivalent to C++ ReSize
    pub fn resize(&mut self, num_data: IndexT) {
        let n = num_data.into();
        self.left_.resize(n, IndexT::from(0));
        self.right_.resize(n, IndexT::from(0));
    }

    /// Run partitioning - equivalent to C++ template<bool FORCE_SIZE> Run
    /// func(thread_id, start, cnt, left_ptr, right_ptr) -> left_count
    pub fn run<F>(
        &mut self,
        cnt: IndexT,
        func: F,
        out: &mut [IndexT],
        two_buffer: bool,
        force_size: bool,
    ) -> IndexT
    where
        F: Fn(i32, IndexT, IndexT, &mut [IndexT], Option<&mut [IndexT]>) -> IndexT + Sync,
    {
        let total = cnt.into();
        let mut nblock = 1i32;
        let mut inner_size = IndexT::from(0);

        if force_size {
            Threading::block_info_force_size(
                self.num_threads_,
                cnt,
                self.min_block_size_,
                &mut nblock,
                &mut inner_size,
            );
        } else {
            Threading::block_info(
                self.num_threads_,
                cnt,
                self.min_block_size_,
                &mut nblock,
                &mut inner_size,
            );
        }

        let inner_size_usize = inner_size.into();

        // Pre-compute offsets
        for i in 0..(nblock as usize) {
            let cur_start = IndexT::from(i * inner_size_usize);
            self.offsets_[i] = cur_start;
        }

        // First parallel loop: partition - collect results to avoid borrowing conflicts
        let results: Vec<_> = (0..nblock)
            .into_par_iter()
            .map(|i| {
                let i_usize = i as usize;
                let cur_start = IndexT::from(i_usize * inner_size_usize);
                let cur_cnt =
                    IndexT::from(min(inner_size_usize, total - i_usize * inner_size_usize));

                if cur_cnt.into() == 0 {
                    return (IndexT::from(0), IndexT::from(0), Vec::new(), Vec::new());
                }

                let start_pos = cur_start.into();
                let cnt_val = cur_cnt.into();

                // Create temporary buffers for this iteration
                let mut left_temp = vec![IndexT::from(0); cnt_val];
                let mut right_temp = if two_buffer {
                    vec![IndexT::from(0); cnt_val]
                } else {
                    Vec::new()
                };

                // Copy data to temporary buffers to avoid borrowing issues
                left_temp.copy_from_slice(&self.left_[start_pos..start_pos + cnt_val]);
                if two_buffer {
                    right_temp.copy_from_slice(&self.right_[start_pos..start_pos + cnt_val]);
                }

                // split data inner, reduce the times of function called
                let cur_left_count = func(
                    i,
                    cur_start,
                    cur_cnt,
                    &mut left_temp[..],
                    if two_buffer {
                        Some(&mut right_temp[..])
                    } else {
                        None
                    },
                );

                if !two_buffer {
                    // reverse for one buffer
                    left_temp[cur_left_count.into()..cnt_val].reverse();
                }

                (
                    cur_left_count,
                    IndexT::from(cnt_val - cur_left_count.into()),
                    left_temp,
                    right_temp,
                )
            })
            .collect();

        // Store results back to internal buffers
        for (i, (left_cnt, right_cnt, left_data, right_data)) in results.into_iter().enumerate() {
            self.left_cnts_[i] = left_cnt;
            self.right_cnts_[i] = right_cnt;

            // Copy back processed data
            if left_cnt.into() > 0 || right_cnt.into() > 0 {
                let start_pos = self.offsets_[i].into();
                let total_cnt = left_cnt.into() + right_cnt.into();
                self.left_[start_pos..start_pos + total_cnt]
                    .copy_from_slice(&left_data[..total_cnt]);
                if !right_data.is_empty() {
                    self.right_[start_pos..start_pos + total_cnt]
                        .copy_from_slice(&right_data[..total_cnt]);
                }
            }
        }

        // Combine write positions
        self.left_write_pos_[0] = IndexT::from(0);
        self.right_write_pos_[0] = IndexT::from(0);
        for i in 1..(nblock as usize) {
            self.left_write_pos_[i] =
                IndexT::from(self.left_write_pos_[i - 1].into() + self.left_cnts_[i - 1].into());
            self.right_write_pos_[i] =
                IndexT::from(self.right_write_pos_[i - 1].into() + self.right_cnts_[i - 1].into());
        }

        let left_cnt = IndexT::from(
            self.left_write_pos_[nblock as usize - 1].into()
                + self.left_cnts_[nblock as usize - 1].into(),
        );
        let right_start = left_cnt.into();

        // Second parallel copy - sequential to avoid thread safety issues
        for i in 0..nblock {
            let i_usize = i as usize;
            let off = self.offsets_[i_usize].into();
            let lcnt = self.left_cnts_[i_usize].into();
            let rcnt = self.right_cnts_[i_usize].into();
            let lpos = self.left_write_pos_[i_usize].into();
            let rpos = self.right_write_pos_[i_usize].into();

            // Copy left partition
            if lcnt > 0 {
                out[lpos..lpos + lcnt].copy_from_slice(&self.left_[off..off + lcnt]);
            }

            // Copy right partition
            if rcnt > 0 {
                if two_buffer {
                    out[right_start + rpos..right_start + rpos + rcnt]
                        .copy_from_slice(&self.right_[off..off + rcnt]);
                } else {
                    out[right_start + rpos..right_start + rpos + rcnt]
                        .copy_from_slice(&self.left_[off + lcnt..off + lcnt + rcnt]);
                }
            }
        }

        left_cnt
    }
}
