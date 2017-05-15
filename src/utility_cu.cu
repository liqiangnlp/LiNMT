/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#include "utility_cu.h"


namespace neural_machine_translation {

// add in 09/06/2016
bool source_side_mode__ = false;
bool train_source_input_embedding_mode__ = true;
bool train_target_input_embedding_mode__ = true;
bool train_target_output_embedding_mode__ = true;
bool train_source_rnn_mode__ = true;
bool train_target_rnn_mode__ = true;
bool train_attention_target_rnn_mode__ = true;
bool soft_regularizer_mode__ = false;

precision train_source_input_embedding_lambda__ = 0;
precision train_target_input_embedding_lambda__ = 0;
precision train_target_output_embedding_lambda__ = 0;
precision train_source_rnn_lambda__ = 0;
precision train_target_rnn_lambda__ = 0;
precision train_attentin_target_rnn_lambda__ = 0;






// for t-sne stuff for paper
precision *p_host_dump_ht__ = NULL;
bool tsne_dump_mode__ = false;
std::ofstream tsne_dump_stream__;  



OutputLogger logger;

bool continue_train_mode__ = false;
bool shuffle_data_mode__ = true;

// for ensembling pre-normalization
bool pre_normalization_mode__ = false;


// for dumping the best model
bool dump_every_best__ = false;
int curr_dump_num__ = 1;
int dump_last_epoch__ = 0;
int curr_dump_minibatch_num__ = 1;

// stuff for unk replacement using attention
bool unk_replacement_mode__ = false;
std::string unk_rep_file_name__;
//std::ofstream unk_rep_file_stream__;
std::vector<int> viterbi_alignments__;           // beamsize
std::vector<int> all_viterbi_alignments__;
std::vector<precision> alignment_scores__;       // longest_sentence * beamsize, for ensembling alignment values
int *p_host_align_indices__;                     // (2 * d_ + 1) * beamsize
precision *p_host_alignment_values__;            // (2 * d_ + 1) * beamsize



unsigned int curr_seed__ = 0;

bool force_decode_mode__ = false;


boost::random::mt19937 generator__;
double lower__ = -0.08;
double upper__ = 0.08;


// global gradient clipping
bool global_grad_clip_mode__ = false;
precision global_norm_clip__ = 0;
precision global_norm_clip_threshold__;
// for stats on gradient norms
double recent_sum__ = 0;

// clip errors with respect to h_t and c_t
bool cell_clip_mode__ = false;
precision cell_clip_threshold__ = 50;
precision error_clip_threshold__ = 1000;



/* individual gradient clipping */
bool individual_grad_clip_mode__ = false;
precision individual_norm_clip_threshold__ = 0.1;


// for gettings only NCE scores (used for reranking, etc ...)
bool nce_score_mode__ = false;


bool dump_nce_stats__ = false;

/* partition function calculation for NCE */
bool print_partition_function_mode__ = false;
/* all the partition function values */
std::vector<double> full_partition_values__;


void PrintPartitionStats() {
  double total_sum = 0;
  double mean = 0;
  double variance = 0;

  for (int i = 0; i < full_partition_values__.size(); ++i) {
    total_sum += full_partition_values__[i];
  }
  mean = total_sum / full_partition_values__.size();

  for (int i = 0; i < full_partition_values__.size(); ++i) {
    variance += (full_partition_values__[i] - mean) * (full_partition_values__[i] - mean);
  }

  variance = variance / full_partition_values__.size();

  std::cerr<<"\n##### NCE PARTITION STATS #####\n";
  std::cerr<<"Partition mean: "<<mean<<"\n";
  std::cerr<<"Partition function standard deviation: "<<std::sqrt(variance)<<"\n\n\n";

  full_partition_values__.clear();
  return;
}

}



std::string CublasErrorString(cublasStatus_t error) {
  switch(error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}


// CHECK: OK //
void CublasErrorWrapper(cublasStatus_t cuda_stat, std::string error_message) {
  //if (cuda_stat != cudaSuccess) {
  if (cuda_stat != CUBLAS_STATUS_SUCCESS) {
    std::string message = CublasErrorString(cuda_stat);
    std::cerr<<error_message<<"\n"
             <<message<<"\n"<<std::flush;
    exit(EXIT_FAILURE);
  }
}


// CHECK: OK //
void CudaErrorWrapper(cudaError_t cuda_stat, std::string error_message) {
  if (cudaSuccess != cuda_stat) {
    std::cerr<<"Error:\n";
    fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(cuda_stat));
    std::cerr<<error_message<<"\n";
    exit(EXIT_FAILURE);
  }
}


void CudaGetLastError() {
  cudaError_t code = cudaGetLastError();
  if (cudaSuccess != code) {
    std::cerr<<"Error in kernel\n";
    std::cerr<<"NO MESSAGE\n";
    fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(code));
    exit(EXIT_FAILURE);
  }
  return;
}


// CHECK: OK //
void CudaGetLastError(std::string message) {
  cudaError_t code = cudaGetLastError();
  if (cudaSuccess != code) {
    std::cerr<<"Error in kernel\n";
    fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(code));
    std::cerr<<message<<"\n";
    exit(EXIT_FAILURE);
  }
}


// CHECK: OK //
void CurandGenerateUniformWrapper(float *p_device_mask, int size, curandGenerator_t &generator) {
  curandGenerateUniform(generator, p_device_mask, size);
}


// CHECK: OK //
void CurandGenerateUniformWrapper(double *p_device_mask, int size, curandGenerator_t &generator) {
  curandGenerateUniformDouble(generator, p_device_mask, size);
}



// CHECK: OK //
void DeviceSyncAll() {

  int num_devices;
  int origin_device;
  cudaGetDevice(&origin_device);
  cudaGetDeviceCount(&num_devices);

  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(origin_device);

}

