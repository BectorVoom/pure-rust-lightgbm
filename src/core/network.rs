/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

//! # LightGBM Network Communication Module
//!
//! This module provides distributed communication functionality for LightGBM,
//! including collective operations like all-reduce, all-gather, and reduce-scatter.
//! It supports various algorithms including Bruck and recursive halving for
//! efficient communication in distributed training environments.

use crate::core::config::Config;
use crate::core::meta::{AllgatherFunction, CommSizeT, ReduceFunction, ReduceScatterFunction};
use crate::core::utils::log::Log;
use std::sync::{Mutex, OnceLock};
use std::vec::Vec;

/// The network structure for all_gather using Bruck algorithm
#[derive(Debug, Clone, Default)]
pub struct BruckMap {
    /// The communication times for one all gather operation
    pub k: i32,
    /// in_ranks[i] means the incoming rank on i-th communication
    pub in_ranks: Vec<i32>,
    /// out_ranks[i] means the out rank on i-th communication
    pub out_ranks: Vec<i32>,
}

impl BruckMap {
    /// Default constructor
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructor with number of machines
    pub fn with_machines(n: i32) -> Self {
        if n <= 1 {
            return Self::default();
        }

        let mut map = Self::default();
        let mut k = 0;
        let mut power = 1;

        // Calculate k (number of communication rounds)
        while power < n {
            k += 1;
            power *= 2;
        }

        map.k = k;
        // Match C++ initialization: default set to -1
        map.in_ranks.resize(k as usize, -1);
        map.out_ranks.resize(k as usize, -1);

        map
    }

    /// Create the object of bruck map
    ///
    /// # Arguments
    /// * `rank` - Rank of this machine
    /// * `num_machines` - The total number of machines
    ///
    /// # Returns
    /// The object of bruck map
    pub fn construct(rank: i32, num_machines: i32) -> Self {
        if num_machines <= 1 {
            return Self::default();
        }

        let mut map = Self::with_machines(num_machines);

        // Fill in the communication pattern for Bruck algorithm
        for i in 0..map.k {
            let step = 1 << i; // 2^i
            // Match C++ implementation: in_rank = (rank + distance) % num_machines
            map.in_ranks[i as usize] = (rank + step) % num_machines;
            // Match C++ implementation: out_rank = (rank - distance + num_machines) % num_machines
            map.out_ranks[i as usize] = (rank - step + num_machines) % num_machines;
        }

        map
    }
}

/// Node type on recursive halving algorithm
/// When number of machines is not power of 2, need group machines into power of 2 group.
/// And we can let each group has at most 2 machines.
/// if the group only has 1 machine. this machine is the normal node
/// if the group has 2 machines, this group will have two type of nodes, one is the leader.
/// leader will represent this group and communication with others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursiveHalvingNodeType {
    /// Normal node, 1 group only have 1 machine
    Normal,
    /// Leader of group when number of machines in this group is 2
    GroupLeader,
    /// Non-leader machines in group
    Other,
}

impl Default for RecursiveHalvingNodeType {
    fn default() -> Self {
        RecursiveHalvingNodeType::Normal
    }
}

/// Network structure for recursive halving algorithm
#[derive(Debug, Clone, Default)]
pub struct RecursiveHalvingMap {
    /// Communication times for one recursive halving algorithm
    pub k: i32,
    /// Node type
    pub node_type: RecursiveHalvingNodeType,
    /// Whether the number of machines is power of 2
    pub is_power_of_2: bool,
    /// Neighbor rank for non-power-of-2 cases
    pub neighbor: i32,
    /// ranks[i] means the machines that will communicate with on i-th communication
    pub ranks: Vec<i32>,
    /// send_block_start[i] means send block start index at i-th communication
    pub send_block_start: Vec<i32>,
    /// send_block_len[i] means send block size at i-th communication
    pub send_block_len: Vec<i32>,
    /// recv_block_start[i] means recv block start index at i-th communication
    pub recv_block_start: Vec<i32>,
    /// recv_block_len[i] means recv block size at i-th communication
    pub recv_block_len: Vec<i32>,
}

impl RecursiveHalvingMap {
    /// Default constructor
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructor with parameters
    pub fn with_params(k: i32, node_type: RecursiveHalvingNodeType, is_power_of_2: bool) -> Self {
        let mut map = Self::default();
        map.k = k;
        map.node_type = node_type;
        map.is_power_of_2 = is_power_of_2;
        map.neighbor = -1;

        if k > 0 {
            map.ranks.resize(k as usize, 0);
            map.send_block_start.resize(k as usize, 0);
            map.send_block_len.resize(k as usize, 0);
            map.recv_block_start.resize(k as usize, 0);
            map.recv_block_len.resize(k as usize, 0);
        }

        map
    }

    /// Create the object of recursive halving map
    ///
    /// # Arguments
    /// * `rank` - Rank of this machine
    /// * `num_machines` - The total number of machines
    ///
    /// # Returns
    /// The object of recursive halving map
    pub fn construct(rank: i32, num_machines: i32) -> Self {
        if num_machines <= 1 {
            return Self::default();
        }

        let is_power_of_2 = (num_machines & (num_machines - 1)) == 0;
        let mut k = 0;
        let mut temp = num_machines;

        // Calculate log2(num_machines) rounded up
        while temp > 1 {
            k += 1;
            temp = (temp + 1) / 2;
        }

        let node_type = if is_power_of_2 {
            RecursiveHalvingNodeType::Normal
        } else {
            // For non-power-of-2, determine node type
            let group_size = num_machines - (1 << (k - 1));
            if rank < group_size {
                if rank % 2 == 0 {
                    RecursiveHalvingNodeType::GroupLeader
                } else {
                    RecursiveHalvingNodeType::Other
                }
            } else {
                RecursiveHalvingNodeType::Normal
            }
        };

        let mut map = Self::with_params(k, node_type, is_power_of_2);

        // Set up communication pattern
        if !is_power_of_2 && node_type == RecursiveHalvingNodeType::Other {
            map.neighbor = rank - 1; // Partner with the group leader
        } else if !is_power_of_2 && node_type == RecursiveHalvingNodeType::GroupLeader {
            map.neighbor = rank + 1; // Partner with the other node
        }

        // Fill communication ranks for each step
        for i in 0..k {
            let step = 1 << i; // 2^i
            map.ranks[i as usize] = rank ^ step; // XOR with step size
        }

        map
    }
}

/// Thread-local storage for network state
struct NetworkState {
    /// Number of all machines
    num_machines: i32,
    /// Rank of local machine
    rank: i32,
    /// Bruck map for all gather algorithm
    bruck_map: BruckMap,
    /// Recursive halving map for reduce scatter
    recursive_halving_map: RecursiveHalvingMap,
    /// Buffer to store block start index
    block_start: Vec<CommSizeT>,
    /// Buffer to store block size
    block_len: Vec<CommSizeT>,
    /// Communication buffer
    buffer: Vec<u8>,
    /// Size of buffer
    buffer_size: CommSizeT,
    /// External reduce scatter function
    reduce_scatter_ext_fun: Option<ReduceScatterFunction>,
    /// External allgather function
    allgather_ext_fun: Option<AllgatherFunction>,
}

impl Default for NetworkState {
    fn default() -> Self {
        Self {
            num_machines: 1,
            rank: 0,
            bruck_map: BruckMap::default(),
            recursive_halving_map: RecursiveHalvingMap::default(),
            block_start: Vec::new(),
            block_len: Vec::new(),
            buffer: Vec::new(),
            buffer_size: 0,
            reduce_scatter_ext_fun: None,
            allgather_ext_fun: None,
        }
    }
}

/// Global network state (thread-local in original C++)
static NETWORK_STATE: OnceLock<Mutex<NetworkState>> = OnceLock::new();

/// A static class that contains some collective communication algorithm
#[derive(Debug)]
pub struct Network;

impl Network {
    /// Get or initialize the network state
    fn get_state() -> &'static Mutex<NetworkState> {
        NETWORK_STATE.get_or_init(|| Mutex::new(NetworkState::default()))
    }

    /// Initialize
    ///
    /// # Arguments
    /// * `config` - Config of network setting
    pub fn init(config: Config) {
        let num_machines = config.num_machines as i32;
        // TODO: Get actual rank from distributed setup (MPI_Comm_rank, environment variables, etc.)
        // For now, assume single machine setup
        let rank = 0;

        Self::init_with_params(num_machines, rank, None, None);
    }

    /// Initialize with explicit parameters
    ///
    /// # Arguments
    /// * `num_machines` - Number of machines
    /// * `rank` - Rank of this machine
    /// * `reduce_scatter_ext_fun` - External reduce scatter function
    /// * `allgather_ext_fun` - External allgather function
    pub fn init_with_params(
        num_machines: i32,
        rank: i32,
        reduce_scatter_ext_fun: Option<ReduceScatterFunction>,
        allgather_ext_fun: Option<AllgatherFunction>,
    ) {
        let state = Self::get_state();
        let mut state = state.lock().unwrap();

        state.num_machines = num_machines;
        state.rank = rank;
        state.reduce_scatter_ext_fun = reduce_scatter_ext_fun;
        state.allgather_ext_fun = allgather_ext_fun;

        // Initialize communication maps
        state.bruck_map = BruckMap::construct(rank, num_machines);
        state.recursive_halving_map = RecursiveHalvingMap::construct(rank, num_machines);

        // Initialize buffers
        state.block_start.resize(num_machines as usize, 0);
        state.block_len.resize(num_machines as usize, 0);

        Log::info(&format!(
            "Network initialized: rank={}, num_machines={}",
            rank, num_machines
        ));
    }

    /// Dispose network resources
    pub fn dispose() {
        let state = Self::get_state();
        let mut state = state.lock().unwrap();
        *state = NetworkState::default();
    }

    /// Get rank of this machine
    pub fn rank() -> i32 {
        let state = Self::get_state();
        let state = state.lock().unwrap();
        state.rank
    }

    /// Get total number of machines
    pub fn num_machines() -> i32 {
        let state = Self::get_state();
        let state = state.lock().unwrap();
        state.num_machines
    }

    /// Perform all_reduce. if data size is small,
    /// will perform AllreduceByAllGather, else with call ReduceScatter followed allgather
    ///
    /// # Arguments
    /// * `input` - Input data
    /// * `input_size` - The size of input data
    /// * `type_size` - The size of one object in the reduce function
    /// * `output` - Output result
    /// * `reducer` - Reduce function
    pub fn allreduce(
        input: &[u8],
        input_size: CommSizeT,
        type_size: i32,
        output: &mut [u8],
        reducer: &ReduceFunction,
    ) {
        let num_machines = Self::num_machines();
        if num_machines <= 1 {
            // Single machine - just copy input to output
            output[..input_size as usize].copy_from_slice(&input[..input_size as usize]);
            return;
        }

        // Decide whether to use all-gather or reduce-scatter approach
        // For small data, all-gather is more efficient
        let threshold = 1024 * 1024; // 1MB threshold

        if input_size < threshold {
            Self::allreduce_by_allgather(input, input_size, type_size, output, reducer);
        } else {
            // TODO: Implement reduce-scatter + all-gather approach for large data
            // This should use ReduceScatter followed by Allgather for better bandwidth utilization
            Self::allreduce_by_allgather(input, input_size, type_size, output, reducer);
        }
    }

    /// Perform all_reduce by using all_gather. it can be use to reduce communication time when data is small
    ///
    /// # Arguments
    /// * `input` - Input data
    /// * `input_size` - The size of input data
    /// * `type_size` - The size of one object in the reduce function
    /// * `output` - Output result
    /// * `reducer` - Reduce function
    pub fn allreduce_by_allgather(
        input: &[u8],
        input_size: CommSizeT,
        type_size: i32,
        output: &mut [u8],
        reducer: &ReduceFunction,
    ) {
        let num_machines = Self::num_machines();
        if num_machines <= 1 {
            output[..input_size as usize].copy_from_slice(&input[..input_size as usize]);
            return;
        }

        // Prepare block information
        let state = Self::get_state();
        let mut state = state.lock().unwrap();

        for i in 0..num_machines as usize {
            state.block_start[i] = (i as CommSizeT) * input_size;
            state.block_len[i] = input_size;
        }

        let all_size = input_size * num_machines as CommSizeT;

        // Ensure buffer is large enough
        if state.buffer_size < all_size {
            state.buffer.resize(all_size as usize, 0);
            state.buffer_size = all_size;
        }

        drop(state); // Release lock before calling allgather

        // Perform all-gather
        Self::allgather_with_blocks(
            input,
            &Self::get_block_start(),
            &Self::get_block_len(),
            &mut Self::get_buffer_mut(),
            all_size,
        );

        // Reduce all gathered data
        let buffer = Self::get_buffer();
        output[..input_size as usize].copy_from_slice(&input[..input_size as usize]);

        for i in 1..num_machines {
            let start = (i as CommSizeT * input_size) as usize;
            let end = start + input_size as usize;
            reducer(
                &buffer[start..end],
                &mut output[..input_size as usize],
                type_size as usize,
                input_size,
            );
        }
    }

    /// Performing all_gather by using Bruck algorithm.
    /// Communication times is O(log(n)), and communication cost is O(send_size * number_machine)
    /// It can be used when all nodes have same input size.
    ///
    /// # Arguments
    /// * `input` - Input data
    /// * `send_size` - The size of input data
    /// * `output` - Output result
    pub fn allgather(input: &[u8], send_size: CommSizeT, output: &mut [u8]) {
        let num_machines = Self::num_machines();
        if num_machines <= 1 {
            output[..send_size as usize].copy_from_slice(&input[..send_size as usize]);
            return;
        }

        // Prepare equal-sized blocks
        let state = Self::get_state();
        let mut state = state.lock().unwrap();

        for i in 0..num_machines as usize {
            state.block_start[i] = (i as CommSizeT) * send_size;
            state.block_len[i] = send_size;
        }

        let all_size = send_size * num_machines as CommSizeT;
        drop(state);

        Self::allgather_with_blocks(
            input,
            &Self::get_block_start(),
            &Self::get_block_len(),
            output,
            all_size,
        );
    }

    /// Performing all_gather by using Bruck algorithm.
    /// Communication times is O(log(n)), and communication cost is O(all_size)
    /// It can be used when nodes have different input size.
    ///
    /// # Arguments
    /// * `input` - Input data
    /// * `block_start` - The block start for different machines
    /// * `block_len` - The block size for different machines
    /// * `output` - Output result
    /// * `all_size` - The size of output data
    pub fn allgather_with_blocks(
        input: &[u8],
        block_start: &[CommSizeT],
        block_len: &[CommSizeT],
        output: &mut [u8],
        all_size: CommSizeT,
    ) {
        // Check if external allgather function is available
        let state = Self::get_state();
        let has_ext_fun = {
            let state = state.lock().unwrap();
            state.allgather_ext_fun.is_some()
        };

        if has_ext_fun {
            let state = state.lock().unwrap();
            if let Some(ref ext_fun) = state.allgather_ext_fun {
                // Use external implementation
                ext_fun(
                    &mut input.to_vec(),
                    block_len[Self::rank() as usize],
                    block_start,
                    block_len,
                    input.len(),
                    output,
                    all_size,
                );
            }
            return;
        }

        // Use internal Bruck algorithm implementation
        Self::allgather_bruck(input, block_start, block_len, output, all_size);
    }

    /// Perform reduce scatter by using recursive halving algorithm.
    /// Communication times is O(log(n)), and communication cost is O(input_size)
    ///
    /// # Arguments
    /// * `input` - Input data
    /// * `input_size` - The size of input data
    /// * `type_size` - The size of one object in the reduce function
    /// * `block_start` - The block start for different machines
    /// * `block_len` - The block size for different machines
    /// * `output` - Output result
    /// * `output_size` - size of output data
    /// * `reducer` - Reduce function
    pub fn reduce_scatter(
        input: &[u8],
        input_size: CommSizeT,
        type_size: i32,
        block_start: &[CommSizeT],
        block_len: &[CommSizeT],
        output: &mut [u8],
        output_size: CommSizeT,
        reducer: &ReduceFunction,
    ) {
        let num_machines = Self::num_machines();
        if num_machines <= 1 {
            let rank = Self::rank() as usize;
            let start = block_start[rank] as usize;
            let len = block_len[rank] as usize;
            output[..len].copy_from_slice(&input[start..start + len]);
            return;
        }

        // Check if external reduce scatter function is available
        let state = Self::get_state();
        let has_ext_fun = {
            let state = state.lock().unwrap();
            state.reduce_scatter_ext_fun.is_some()
        };

        if has_ext_fun {
            let state = state.lock().unwrap();
            if let Some(ref ext_fun) = state.reduce_scatter_ext_fun {
                // Use external implementation
                ext_fun(
                    &mut input.to_vec(),
                    input_size,
                    type_size as usize,
                    block_start,
                    block_len,
                    input.len(),
                    output,
                    output_size,
                    reducer,
                );
            }
            return;
        }

        // Use internal recursive halving implementation
        Self::reduce_scatter_recursive_halving(
            input,
            input_size,
            type_size,
            block_start,
            block_len,
            output,
            output_size,
            reducer,
        );
    }

    /// Global synchronization by minimum value
    pub fn global_sync_up_by_min<T>(local: T) -> T
    where
        T: Copy + PartialOrd + Default + std::fmt::Debug + 'static,
    {
        if Self::num_machines() <= 1 {
            return local;
        }

        let type_size = std::mem::size_of::<T>();
        let local_bytes = unsafe {
            std::slice::from_raw_parts(&local as *const T as *const u8, type_size).to_vec()
        };
        let mut global_bytes = local_bytes.clone();

        let reducer: ReduceFunction = Box::new(
            move |src: &[u8], dst: &mut [u8], type_size: usize, len: CommSizeT| {
                let num_elements = len as usize / type_size;
                for i in 0..num_elements {
                    let src_start = i * type_size;
                    let dst_start = i * type_size;
                    if src_start + type_size <= src.len() && dst_start + type_size <= dst.len() {
                        let src_val = unsafe { *(src.as_ptr().add(src_start) as *const T) };
                        let dst_ref = unsafe { &mut *(dst.as_mut_ptr().add(dst_start) as *mut T) };
                        if src_val < *dst_ref {
                            *dst_ref = src_val;
                        }
                    }
                }
            },
        );

        Self::allreduce(
            &local_bytes,
            local_bytes.len() as CommSizeT,
            type_size as i32,
            &mut global_bytes,
            &reducer,
        );

        unsafe { *(global_bytes.as_ptr() as *const T) }
    }

    /// Global synchronization by maximum value
    pub fn global_sync_up_by_max<T>(local: T) -> T
    where
        T: Copy + PartialOrd + Default + std::fmt::Debug + 'static,
    {
        if Self::num_machines() <= 1 {
            return local;
        }

        let type_size = std::mem::size_of::<T>();
        let local_bytes = unsafe {
            std::slice::from_raw_parts(&local as *const T as *const u8, type_size).to_vec()
        };
        let mut global_bytes = local_bytes.clone();

        let reducer: ReduceFunction = Box::new(
            move |src: &[u8], dst: &mut [u8], type_size: usize, len: CommSizeT| {
                let num_elements = len as usize / type_size;
                for i in 0..num_elements {
                    let src_start = i * type_size;
                    let dst_start = i * type_size;
                    if src_start + type_size <= src.len() && dst_start + type_size <= dst.len() {
                        let src_val = unsafe { *(src.as_ptr().add(src_start) as *const T) };
                        let dst_ref = unsafe { &mut *(dst.as_mut_ptr().add(dst_start) as *mut T) };
                        if src_val > *dst_ref {
                            *dst_ref = src_val;
                        }
                    }
                }
            },
        );

        Self::allreduce(
            &local_bytes,
            local_bytes.len() as CommSizeT,
            type_size as i32,
            &mut global_bytes,
            &reducer,
        );

        unsafe { *(global_bytes.as_ptr() as *const T) }
    }

    /// Global synchronization by sum
    pub fn global_sync_up_by_sum<T>(local: T) -> T
    where
        T: Copy + Default + std::ops::AddAssign + std::fmt::Debug + 'static,
    {
        if Self::num_machines() <= 1 {
            return local;
        }

        let type_size = std::mem::size_of::<T>();
        let local_bytes = unsafe {
            std::slice::from_raw_parts(&local as *const T as *const u8, type_size).to_vec()
        };
        let mut global_bytes = unsafe {
            let zero = T::default();
            std::slice::from_raw_parts(&zero as *const T as *const u8, type_size).to_vec()
        };

        let reducer: ReduceFunction = Box::new(
            move |src: &[u8], dst: &mut [u8], type_size: usize, len: CommSizeT| {
                let num_elements = len as usize / type_size;
                for i in 0..num_elements {
                    let src_start = i * type_size;
                    let dst_start = i * type_size;
                    if src_start + type_size <= src.len() && dst_start + type_size <= dst.len() {
                        let src_val = unsafe { *(src.as_ptr().add(src_start) as *const T) };
                        let dst_ref = unsafe { &mut *(dst.as_mut_ptr().add(dst_start) as *mut T) };
                        *dst_ref += src_val;
                    }
                }
            },
        );

        Self::allreduce(
            &local_bytes,
            local_bytes.len() as CommSizeT,
            type_size as i32,
            &mut global_bytes,
            &reducer,
        );

        unsafe { *(global_bytes.as_ptr() as *const T) }
    }

    /// Global synchronization by mean
    pub fn global_sync_up_by_mean<T>(local: T) -> T
    where
        T: Copy
            + Default
            + std::ops::AddAssign
            + std::ops::Div<T, Output = T>
            + From<i32>
            + std::fmt::Debug
            + 'static,
    {
        let sum = Self::global_sync_up_by_sum(local);
        sum / T::from(Self::num_machines())
    }

    /// Global sum for vectors
    pub fn global_sum<T>(local: &[T]) -> Vec<T>
    where
        T: Copy + Default + std::ops::AddAssign + std::fmt::Debug + 'static,
    {
        if Self::num_machines() <= 1 {
            return local.to_vec();
        }

        let type_size = std::mem::size_of::<T>();
        let local_bytes = unsafe {
            std::slice::from_raw_parts(local.as_ptr() as *const u8, local.len() * type_size).to_vec()
        };
        let mut global_bytes = vec![0u8; local.len() * type_size];
        
        // Initialize global_bytes with zero values
        for i in 0..local.len() {
            let zero = T::default();
            let zero_bytes = unsafe {
                std::slice::from_raw_parts(&zero as *const T as *const u8, type_size)
            };
            global_bytes[i * type_size..(i + 1) * type_size].copy_from_slice(zero_bytes);
        }

        let reducer: ReduceFunction = Box::new(
            move |src: &[u8], dst: &mut [u8], type_size: usize, len: CommSizeT| {
                let num_elements = len as usize / type_size;
                for i in 0..num_elements {
                    let src_start = i * type_size;
                    let dst_start = i * type_size;
                    if src_start + type_size <= src.len() && dst_start + type_size <= dst.len() {
                        let src_val = unsafe { *(src.as_ptr().add(src_start) as *const T) };
                        let dst_ref = unsafe { &mut *(dst.as_mut_ptr().add(dst_start) as *mut T) };
                        *dst_ref += src_val;
                    }
                }
            },
        );

        Self::allreduce(
            &local_bytes,
            local_bytes.len() as CommSizeT,
            type_size as i32,
            &mut global_bytes,
            &reducer,
        );

        (0..local.len())
            .map(|i| unsafe { *(global_bytes.as_ptr().add(i * type_size) as *const T) })
            .collect()
    }

    /// Global array gathering
    pub fn global_array<T>(local: T) -> Vec<T>
    where
        T: Copy + Default + std::fmt::Debug + 'static,
    {
        let num_machines = Self::num_machines();
        if num_machines <= 1 {
            return vec![local];
        }

        let type_size = std::mem::size_of::<T>();

        let state = Self::get_state();
        let mut state = state.lock().unwrap();

        for i in 0..num_machines as usize {
            state.block_start[i] = (i * type_size) as CommSizeT;
            state.block_len[i] = type_size as CommSizeT;
        }

        drop(state);

        let local_bytes = unsafe {
            std::slice::from_raw_parts(&local as *const T as *const u8, type_size).to_vec()
        };
        let mut global_bytes = vec![0u8; type_size * num_machines as usize];

        Self::allgather_with_blocks(
            &local_bytes,
            &Self::get_block_start(),
            &Self::get_block_len(),
            &mut global_bytes,
            (type_size * num_machines as usize) as CommSizeT,
        );

        (0..num_machines as usize)
            .map(|i| unsafe { *(global_bytes.as_ptr().add(i * type_size) as *const T) })
            .collect()
    }

    // Private helper methods

    fn get_block_start() -> Vec<CommSizeT> {
        let state = Self::get_state();
        let state = state.lock().unwrap();
        state.block_start.clone()
    }

    fn get_block_len() -> Vec<CommSizeT> {
        let state = Self::get_state();
        let state = state.lock().unwrap();
        state.block_len.clone()
    }

    fn get_buffer() -> Vec<u8> {
        let state = Self::get_state();
        let state = state.lock().unwrap();
        state.buffer.clone()
    }

    fn get_buffer_mut() -> Vec<u8> {
        let state = Self::get_state();
        let mut state = state.lock().unwrap();
        std::mem::take(&mut state.buffer)
    }

    /// Private method: Allgather using Bruck algorithm
    fn allgather_bruck(
        input: &[u8],
        block_start: &[CommSizeT],
        block_len: &[CommSizeT],
        output: &mut [u8],
        all_size: CommSizeT,
    ) {
        let num_machines = Self::num_machines();
        let rank = Self::rank();

        if num_machines <= 1 {
            output[..input.len()].copy_from_slice(input);
            return;
        }

        // Initialize output with local data
        let local_start = block_start[rank as usize] as usize;
        let local_len = block_len[rank as usize] as usize;
        output[local_start..local_start + local_len].copy_from_slice(&input[..local_len]);

        // TODO: Implement actual Bruck algorithm communication
        // Full implementation requires:
        // 1. Network sockets/MPI communication with other nodes
        // 2. Iterative data exchange following bruck_map communication pattern
        // 3. Data gathering and concatenation from all nodes
        // For now, only handles single-node case
        Log::warning("Bruck algorithm: Network communication not implemented - placeholder only");
    }

    /// Private method: Reduce scatter using recursive halving
    fn reduce_scatter_recursive_halving(
        input: &[u8],
        input_size: CommSizeT,
        type_size: i32,
        block_start: &[CommSizeT],
        block_len: &[CommSizeT],
        output: &mut [u8],
        output_size: CommSizeT,
        reducer: &ReduceFunction,
    ) {
        let num_machines = Self::num_machines();
        let rank = Self::rank();

        if num_machines <= 1 {
            let start = block_start[rank as usize] as usize;
            let len = block_len[rank as usize] as usize;
            output[..len].copy_from_slice(&input[start..start + len]);
            return;
        }

        // TODO: Implement actual recursive halving algorithm
        // Full implementation requires:
        // 1. Network communication following recursive_halving_map pattern
        // 2. Iterative reduce operations with neighboring nodes
        // 3. Data scattering according to block assignments
        // For now, extract local portion only
        let start = block_start[rank as usize] as usize;
        let len = block_len[rank as usize] as usize;
        output[..len].copy_from_slice(&input[start..start + len]);

        Log::warning("Recursive halving: Network communication not implemented - placeholder only");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bruck_map_construction() {
        let map = BruckMap::construct(0, 4);
        assert!(map.k > 0);
        assert_eq!(map.in_ranks.len(), map.k as usize);
        assert_eq!(map.out_ranks.len(), map.k as usize);
    }

    #[test]
    fn test_recursive_halving_map_construction() {
        let map = RecursiveHalvingMap::construct(0, 4);
        assert!(map.k > 0);
        assert_eq!(map.ranks.len(), map.k as usize);
        assert!(map.is_power_of_2);
    }

    #[test]
    fn test_network_initialization() {
        Network::init_with_params(4, 0, None, None);
        assert_eq!(Network::num_machines(), 4);
        assert_eq!(Network::rank(), 0);
        Network::dispose();
    }

    #[test]
    fn test_single_machine_operations() {
        Network::init_with_params(1, 0, None, None);

        let input = vec![1u8, 2, 3, 4];
        let mut output = vec![0u8; 4];

        let reducer: ReduceFunction = Box::new(
            |src: &[u8], dst: &mut [u8], _type_size: usize, len: CommSizeT| {
                for i in 0..len as usize {
                    dst[i] = src[i] + dst[i];
                }
            },
        );

        Network::allreduce(&input, 4, 1, &mut output, &reducer);
        assert_eq!(output, input);

        Network::dispose();
    }

    #[test]
    fn test_global_sync_operations() {
        Network::init_with_params(1, 0, None, None);

        // Test with single machine (should return input value)
        assert_eq!(Network::global_sync_up_by_min(42u8), 42u8);
        assert_eq!(Network::global_sync_up_by_max(42u8), 42u8);
        assert_eq!(Network::global_sync_up_by_sum(42u8), 42u8);

        Network::dispose();
    }

    #[test]
    fn test_allgather_single_machine() {
        Network::init_with_params(1, 0, None, None);

        let input = vec![1u8, 2, 3, 4];
        let mut output = vec![0u8; 4];

        Network::allgather(&input, 4, &mut output);
        assert_eq!(output, input);

        Network::dispose();
    }
}
