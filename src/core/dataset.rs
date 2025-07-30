use crate::core::bin::{BinIterator, BinMapper};
use crate::core::config::Config;
use crate::core::feature_group::FeatureGroup;
use crate::core::meta::{DataSizeT, LabelT};
use crate::io::train_share_states::TrainingShareStates;
use crate::core::utils::byte_buffer::ByteBuffer;
use crate::core::utils::common::Common;
use crate::core::utils::log::Log;
use crate::core::utils::openmp_wrapper::omp_num_threads;
use arrow2::array::*;
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::error::Result;
// CUDAColumnData, CUDAMetadata → GPU処理用 (#ifdef USE_CUDAで条件コンパイル)
