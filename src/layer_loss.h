/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef LOSS_LAYER_H_
#define LOSS_LAYER_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>


#include "layer_gpu.h"
#include "global_configuration.h"
#include "layer_transfer.h"
#include "deep_rnn.h"
#include "deep_rnn_kernel.h"



//#define DEBUG_LOSS_LAYER_


namespace neural_machine_translation {



template <typename>
class LowerTransferLayer;


////////////////////////////////////// BaseLossLayer //////////////////////////////////////
////////////////////////////////////// BaseLossLayer //////////////////////////////////////
////////////////////////////////////// BaseLossLayer //////////////////////////////////////
template <typename T>
class BaseLossLayer {
  
public:
  virtual SoftmaxLayerGpuInformation InitGpu(int device_number) = 0;

public:
  virtual void InitLossLayer(NeuralMachineTranslation<precision> *p_neural_mt, GlobalConfiguration &configuration) = 0;
  virtual void ForwardProp(int index) = 0;

public:
  virtual void BackProp1(int index) = 0;  // This is done for multi GPUs paralleism
  virtual void BackProp2(int index) = 0;

public:
  virtual void BackPropPreprocessGpu(T *p_device_h_t, int step) = 0;
  virtual void BackPropPreprocessGpuMGpu(int step) = 0;

public:
  virtual void PreprocessGpuVocabIndices(int *p_host_output_vocab_indices_target, int current_target_length) = 0;

public:
  virtual void UpdateWeights() = 0;

public:
  virtual void ClearGradients() = 0;

public:
  virtual double ComputeLossGpu(int index) = 0;

public:
  virtual void CalculateGlobalNorm() = 0;
  virtual void UpdateGlobalParams() = 0;

public:
  virtual void CheckAllGradients(T epsilon) = 0;

public:
  virtual void UpdateLearningRate(T learning_rate) = 0;

public:
  virtual void InitLowerTransferLayer(bool lower_input, bool copy_d_err_ht, InputToHiddenLayer<T> *p_input_layer, HiddenToHiddenLayer<T> *p_hidden_layer) = 0;
  
public:
  virtual T *GetHTPtr(int index) = 0;
  virtual void SetHTPtr(int index, T *p_device_h_t) = 0;

public:
  virtual cudaEvent_t GetErrHTEvent() = 0;

public:
  virtual void LoadWeights(std::ifstream &input_stream) = 0;
  virtual void DumpWeights(std::ofstream &output_stream) = 0;
  
public:
  virtual double GetTrainPerplexity() = 0;

public:
  virtual void GetDistributionGpuDecoderWrapper() = 0;

public:
  virtual T *GetDistPtr() = 0;
};



////////////////////////////////////// SoftMaxNode //////////////////////////////////////
////////////////////////////////////// SoftMaxNode //////////////////////////////////////
////////////////////////////////////// SoftMaxNode //////////////////////////////////////
template <typename T>
class SoftmaxNode {
public:
  // each node stores the unnormalized probabilities, plus the h_t
  T *p_device_outputdist_;                // output_vocab_size_ x minibatch_size_
  T *p_device_h_t_;                       // lstm_size_ x minibatch_size_
  T *p_device_d_errt_ht_;                 // lstm_size_ x minibatch_size_
  T *p_device_dropout_mask_;              // lstm_size_ x minibatch_size_
  int index_;

public:
  // Constructor
  SoftmaxNode() {}

public:
  void Init(int lstm_size, int minibatch_size, int output_vocab_size, int index, bool dropout_mode);
};


template <typename T>
void SoftmaxNode<T>::Init(int lstm_size, int minibatch_size, int output_vocab_size, int index, bool dropout_mode) {

  if (!force_decode_mode__) {
    CudaErrorWrapper(cudaMalloc((void**)&p_device_outputdist_, output_vocab_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  }

  CudaErrorWrapper(cudaMalloc((void**)&p_device_h_t_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void**)&p_device_d_errt_ht_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  index_ = index;
  if (dropout_mode) {
    CudaErrorWrapper(cudaMalloc((void**)&p_device_dropout_mask_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  }
}



////////////////////////////////////// SoftmaxLayer //////////////////////////////////////
////////////////////////////////////// SoftmaxLayer //////////////////////////////////////
////////////////////////////////////// SoftmaxLayer //////////////////////////////////////
template <typename T>
class SoftmaxLayer : public BaseLossLayer<T> {

  // GPU parameters
public:
  SoftmaxLayerGpuInformation softmax_layer_gpu_information_;

public:
  // host pointers
  T *p_host_d_;
  T *p_host_h_t_;
  T *p_host_b_d_;
  T *p_host_d_errt_ht_;
  T *p_host_ones_;
  int *p_host_output_vocab_indices_;
  int *p_host_output_vocab_indices_01_;
  T *p_host_d_grad_;
  T *p_host_output_vocab_indices_01_float_;
  T *p_host_b_d_grad_;

  thrust::host_vector<T> thrust_host_outputdist_;                  // output_vocab_size x minibatch_size
  thrust::host_vector<T> thrust_host_normalization_;               // 1 x minibatch_size


public:
  // device pointers
  T *p_device_d_;                        // output_vocab_size_ x lstm_size_, embeddings
  T *p_device_h_t_;                      // lstm_size_ x minibatch_size_, for backprop
  T *p_device_b_d_;                      // output_vocab_size_ x 1, bias
  T *p_device_d_errt_ht_;
  T *p_device_ones_;                     // output_vocab_size_, init with 1
  int *p_device_output_vocab_indices_;              // lstm_size_ x longest_sent_, column major, (33 71 140 95 5 1 -1) => (33 71 140 95 5 1 0)
  int *p_device_output_vocab_indices_01_;           // lstm_size_ x longest_sent_, column major, (33 71 140 95 5 1 -1) => (1 1 1 1 1 1 0)
  T *p_device_d_grad_;                              // output_vocab_size_ x lstm_size_
  T *p_device_output_vocab_indices_01_float_;       // lstm_size_ x longest_sent_, column major, (33 71 140 95 5 1 -1) => (1 1 1 1 1 1 0)
  T *p_device_b_d_grad_;                            // output_vocab_size_
  T *p_device_outputdist_;                          // output_vocab_size_ x minibatch_size_
  T *p_device_normalization_;                       // 1 x minibatch_size_

  thrust::device_vector<T> thrust_device_outputdist_;        // output_vocab_size x minibatch_size
  thrust::device_vector<T> thrust_device_normalization_;     // 1 x minibatch_size


public:
  // truncated softmax information
  int *p_host_truncated_vocab_mapping_;  // truncated softmax mapping for sampled indices
  int *p_device_truncated_vocab_mapping_;
  bool truncated_softmax_mode_;          // trunacted softmax information
  int shortlist_size_;
  int trunc_size_;
  T sample_correction_;
  int shortlist_size_plus_;      // shortlist plus the unique words sampled in minibatch

  T *p_device_subset_d_;         // stores this for the shortlist + sampled vocabulary
  T *p_host_subset_d_;           // stores this for the shortlist + sampled vocabulary
  T *p_device_subset_d_grad_;
  T *p_host_subset_d_grad_;
  T *p_device_subset_b_d_;       // stores this for the shortlist + sampled vocabulary
  T *p_host_subset_b_d_;         // stores this for the shortlist + sampled vocabulary
  T *p_device_subset_b_d_grad_;  
  T *p_host_subset_b_d_grad_;


  // thrust device pointers
  thrust::device_ptr<T> p_thrust_device_subset_d_grad_;
  thrust::device_ptr<T> p_thrust_device_subset_b_d_grad_;
  
public:
  double *p_device_train_perplexity_;                        // size = 1
  double *p_device_outputdist_perplexity_;                   // output_vocab_size_ x minibatch_size_ 

public:
  thrust::device_ptr<T> p_thrust_device_d_grad_;             // output_vocab_size_ x lstm_size_
  thrust::device_ptr<T> p_thrust_device_b_d_grad_;           // output_vocab_size_

public:
  // for norm clip
  T *p_device_result_;                                        // 1
  T *p_device_result_tmp_;                                    // NORM_THREADS (inited with 256)

  int *p_device_output_vocab_indices_single_;                 // minibatch_size_ x longest_sent_, use minibatch_size_ x 1 (nodes i)
                                                              // (33 71 140 95 5 1 0)
  int *p_device_output_vocab_indices_01_single_;              // minibatch_size_ x longest_sent_, use minibatch_size_ x 1 (nodes i)
                                                              // (33 71 140 95 5 1 -1) = (1 1 1 1 1 1 0)
  T *p_device_output_vocab_indices_01_float_single_;          // minibatch_size_ x longest_sent_, use minibatch_size_ x 1 (nodes i)
                                                              // (33 71 140 95 5 1 -1) = (1 1 1 1 1 1 0)

public:
  bool gradient_clip_mode_;  // If true then clip gradients
  T norm_clip_;              // for gradient clipping
  int minibatch_size_;
  int output_vocab_size_;    // equal to config.target_vocab_size_
  int lstm_size_;
  T learning_rate_;
  bool softmax_scaled_mode_;

public:
  // dropout stuff
  bool dropout_mode_;
  T dropout_rate_;

public:
  NeuralMachineTranslation<T> *p_neural_mt_;

public:
  bool training_perplexity_mode_;

public:
  LowerTransferLayer<T> lower_layer_;

public:
  std::vector<SoftmaxNode<T>> v_nodes_;

public:
  curandGenerator_t rand_generator_;

public:
  SoftmaxLayer() {}        // Constructor

public:
  void InitLossLayer(NeuralMachineTranslation<precision> *p_neural_mt, GlobalConfiguration &configuration);
  void InitSoftmaxLayerGpu(int output_vocab_size, int minibatch_size, NeuralMachineTranslation<precision> *p_neural_mt, \
      T norm_clip, int lstm_size, bool clip_gradient_mode, T learning_rate, int longest_sentence);

public:
  void ClearGradients();
  void ClearGradientsGpu();

public:
  void ForwardProp(int index);
  void ForwardPropGpu(int index);

public:
  void BackProp1(int index);
  void BackProp1Gpu(int index);

  void BackProp2(int index);
  void BackProp2Gpu(int index);

public:
  void UpdateWeights();
  void UpdateWeightsGpu();

public:
  void CalculateGlobalNorm();
  void UpdateGlobalParams();

public:
  void DumpWeights(std::ofstream &output);
  void DumpWeightsGpu(std::ofstream &output);

public:
  void LoadWeights(std::ifstream &input_stream);
  void LoadWeightsGpu(std::ifstream &input_stream);

public:
  void CheckAllGradients(T epsilon);
  void CheckAllGradientsGpu(T epsilon);

public:
  void GetPerplexityGpu(T *p_device_h_t, int index);

public:
  void GetDistributionGpu(int output_vocab_size, T *p_device_outputdist, T *p_device_d, T *p_device_b_d, T *p_device_h_t);

public:
  void GetHTGradientGpu(int output_vocab_size, T *p_device_d, T *p_device_outputdist, T *p_device_d_errt_ht, int index);

public:
  void ComputeDGradientGpu(int output_vocab_size, T *p_device_outputdist, T *p_device_d_grad, T *p_device_h_t);

public:
  void ComputeBDGradientGpu(int output_vocab_size, T *p_device_outputdist, T *p_device_b_d_grad);

public:
  double ComputeLossGpu(int index);

public:
  void CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols);

public:
  void PreprocessGpuVocabIndices(int *p_host_output_vocab_indices_target, int current_target_length);

public:
  void BackPropPreprocessGpu(T *p_device_h_t, int step);
  void BackPropPreprocessGpuMGpu(int step);

public:
  void UpdateLearningRate(T learning_rate);

public:
  double GetTrainPerplexity();

public:
  void GetDistributionGpuDecoderWrapper();

public:
  SoftmaxLayerGpuInformation InitGpu(int device_number);

public:
  void InitLowerTransferLayer(bool lower_input, bool copy_d_err_ht, InputToHiddenLayer<T> *p_input_layer, HiddenToHiddenLayer<T> *p_hidden_layer);

public:
  T *GetHTPtr(int index);
  void SetHTPtr(int index, T *p_device_h_t);

public:
  cudaEvent_t GetErrHTEvent();

public:
  T *GetDistPtr();
};


/////////////////// Implementations for Class Template SoftmaxLayer ///////////////////
template <typename T>
SoftmaxLayerGpuInformation SoftmaxLayer<T>::InitGpu(int device_number) {
  softmax_layer_gpu_information_.Init(device_number);
  return softmax_layer_gpu_information_;
}


// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::InitLossLayer(NeuralMachineTranslation<precision> *p_neural_mt, GlobalConfiguration &configuration) {
  
  output_vocab_size_ = configuration.target_vocab_size_;
  lstm_size_ = configuration.lstm_size_;
  gradient_clip_mode_ = configuration.clip_gradient_mode_;
  p_neural_mt_ = p_neural_mt;
  norm_clip_ = configuration.norm_clip_;
  minibatch_size_ = configuration.minibatch_size_;
  learning_rate_ = configuration.learning_rate_;

  // configuration.softmax_scaled_mode_;
  softmax_scaled_mode_ = true;
  // configuration.truncated_softmax_mode_;
  truncated_softmax_mode_ = false;

  training_perplexity_mode_ = configuration.training_perplexity_mode_;

  dropout_mode_ = configuration.dropout_mode_;
  dropout_rate_ = configuration.dropout_rate_;

  InitSoftmaxLayerGpu(output_vocab_size_, minibatch_size_, p_neural_mt_, norm_clip_, \
                      lstm_size_, gradient_clip_mode_, learning_rate_, configuration.longest_sentence_);
};


// CHECK DECODER: OK //
// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::InitSoftmaxLayerGpu(int output_vocab_size, int minibatch_size, NeuralMachineTranslation<precision> *neural_mt, T norm_clip, int lstm_size, bool clip_gradient_mode, T learning_rate, int longest_sentence) {
  
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  thrust_host_outputdist_.resize(output_vocab_size * minibatch_size);
  thrust_host_normalization_.resize(1 * minibatch_size);
  thrust_device_outputdist_.resize(output_vocab_size * minibatch_size);
  thrust_device_normalization_.resize(1 * minibatch_size);

  InitThrustVector(thrust_host_outputdist_, output_vocab_size * minibatch_size);
  InitThrustVector(thrust_host_normalization_, 1 * minibatch_size);

  thrust_device_outputdist_ = thrust_host_outputdist_;
  thrust_device_normalization_ = thrust_host_normalization_;

  p_device_outputdist_ = thrust::raw_pointer_cast(&thrust_device_outputdist_[0]);
  p_device_normalization_ = thrust::raw_pointer_cast(&thrust_device_normalization_[0]);

  FullMatrixSetup(&p_host_d_, &p_device_d_, output_vocab_size, lstm_size);
  FullMatrixSetup(&p_host_h_t_, &p_device_h_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_b_d_, &p_device_b_d_, output_vocab_size, 1);
  FullMatrixSetup(&p_host_d_errt_ht_, &p_device_d_errt_ht_, lstm_size, minibatch_size);

  FullVectorSetupOnes(&p_host_ones_, &p_device_ones_, output_vocab_size);

  // saving space during --force-decoding
  if (!force_decode_mode__) {
    FullMatrixSetup(&p_host_d_grad_, &p_device_d_grad_, output_vocab_size, lstm_size);
  }

  FullMatrixSetupZeros(&p_host_output_vocab_indices_, &p_device_output_vocab_indices_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_output_vocab_indices_01_, &p_device_output_vocab_indices_01_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_output_vocab_indices_01_float_, &p_device_output_vocab_indices_01_float_, minibatch_size, longest_sentence);
  FullVectorSetup(&p_host_b_d_grad_, &p_device_b_d_grad_, output_vocab_size);

  p_thrust_device_d_grad_ = thrust::device_pointer_cast(p_device_d_grad_);
  p_thrust_device_b_d_grad_ = thrust::device_pointer_cast(p_device_b_d_grad_);

  CudaErrorWrapper(cudaMalloc((void**)&p_device_result_, 1 * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void**)&p_device_result_tmp_, NORM_THREADS * sizeof(T)),"GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void**)&p_device_outputdist_perplexity_,output_vocab_size * minibatch_size * sizeof(double)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void**)&p_device_train_perplexity_, 1 * sizeof(double)), "GPU memory allocation failed\n");
  cudaMemset(p_device_train_perplexity_, 0, 1 * sizeof(double));

  curandCreateGenerator(&rand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
  //boost::uniform_int<> unif_boost(1, 1000000);
  //curandSetPseudoRandomGeneratorSeed(rand_generator_, unif_boost(generator__));
  curandSetPseudoRandomGeneratorSeed(rand_generator_, curr_seed__);
  curr_seed__ += 7;

  for (int i = 0; i < longest_sentence; ++i) {
    SoftmaxNode<T> softmax_node;
    softmax_node.Init(lstm_size, minibatch_size, output_vocab_size, i, dropout_mode_);
    v_nodes_.push_back(softmax_node);
    //v_nodes_.push_back(SoftmaxNode<T>(lstm_size, minibatch_size, output_vocab_size, i, dropout_mode_));
  }

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);
  if (!force_decode_mode__) {
    ClearGradients();
  }

  if (dump_nce_stats__) {
    // dump_nce_stats is not written //
  }

  cudaSetDevice(0);
}


// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::InitLowerTransferLayer(bool lower_input, bool copy_d_err_ht, InputToHiddenLayer<T> *p_input_layer, HiddenToHiddenLayer<T> *p_hidden_layer) {
  lower_layer_.InitLowerTransferLayer(lower_input, copy_d_err_ht, p_input_layer, p_hidden_layer);
}


// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::ForwardProp(int index) {
  ForwardPropGpu(index);
}


// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::ForwardPropGpu(int index) {
#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  // Wait for the h_t transfer to start
  if (lower_layer_.lower_input_mode_) {
    cudaStreamWaitEvent(softmax_layer_gpu_information_.s00_, lower_layer_.p_input_layer_->input_hidden_layer_information_.h_t_below_transfer_, 0);
  } else {
    cudaStreamWaitEvent(softmax_layer_gpu_information_.s00_, lower_layer_.p_hidden_layer_->hidden_hidden_layer_information_.h_t_below_transfer_, 0);
  }

  // dropout mode and no attention
  if (dropout_mode_ && !p_neural_mt_->attention_configuration_.attention_model_mode_) {
    curandSetStream(rand_generator_, softmax_layer_gpu_information_.s00_);
    CurandGenerateUniformWrapper(v_nodes_[index].p_device_dropout_mask_, lstm_size_ * minibatch_size_, rand_generator_);

    // v_nodes_[index].p_device_h_t_[i] = (v_nodes_[index].p_device_dropout_mask_[i] < dropout_rate_) * (1/dropout_rate_) * v_nodes_[index].p_device_h_t_[i];
    DropoutKernel<<<256, 256, 0, softmax_layer_gpu_information_.s00_>>>(v_nodes_[index].p_device_dropout_mask_, dropout_rate_, v_nodes_[index].p_device_h_t_, lstm_size_ * minibatch_size_);
  }

  GetDistributionGpu(output_vocab_size_, v_nodes_[index].p_device_outputdist_, p_device_d_, p_device_b_d_, v_nodes_[index].p_device_h_t_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

}

template <typename T>
void SoftmaxLayer<T>::BackProp1(int index) {
  BackProp1Gpu(index);
}


template <typename T>
void SoftmaxLayer<T>::BackProp1Gpu(int index) {
  GetHTGradientGpu(output_vocab_size_, p_device_d_, v_nodes_[index].p_device_outputdist_, v_nodes_[index].p_device_d_errt_ht_, index);
}



template <typename T>
void SoftmaxLayer<T>::BackProp2(int index) {
  BackProp2Gpu(index);
}

template <typename T>
void SoftmaxLayer<T>::BackProp2Gpu(int index) {
  ComputeDGradientGpu(output_vocab_size_, v_nodes_[index].p_device_outputdist_, p_device_d_grad_, v_nodes_[index].p_device_h_t_);
  ComputeBDGradientGpu(output_vocab_size_, v_nodes_[index].p_device_outputdist_, p_device_b_d_grad_);
}


// get the error for the softmax with respect to p_device_h_t
// output_vocab_indices should contain no -1's
// output_vocab_indices should contain all 1's except for zeros where the column should be zeroed out
template <typename T>
void SoftmaxLayer<T>::GetHTGradientGpu(int output_vocab_size, T *p_device_d, T *p_device_outputdist, T *p_device_d_errt_ht, int index) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaStreamWaitEvent(softmax_layer_gpu_information_.s01_, softmax_layer_gpu_information_.output_dist_done_, 0);

  T alpha = -1;
  T beta = 0;

  // stream 01
  cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s01_);

  // multiply p_device_outputdist by p_device_d
  // p_device_d_errt_ht (lstm_size_ x minibatch_size_) = (-1) * p_device_d^T (lstm_size_ x output_vocab_size) * p_device_outputdist (output_vocab_size x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, output_vocab_size, &alpha, p_device_d, output_vocab_size, p_device_outputdist, output_vocab_size, &beta, p_device_d_errt_ht, lstm_size_), "SoftmaxLayer::GetHTGradientGpu p_device_d_errt_ht failed");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // add in the p_device_d rows
  int threads_per_block = 128;
  // bug ????? or int [[[num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block]]] is right
  int num_block = (output_vocab_size + threads_per_block - 1) / threads_per_block;
  dim3 kernel_dim(minibatch_size_, num_block, 1);
  // p_device_d_errt_ht[i][j] = p_device_d_errt_ht[i][j] + p_device_d[p_device_output_vocab_indices_single_[j]][i]
  // p_device_d_errt_ht (lstm_size_ x minibatch_size_)
  // p_device_d (output_vocab_size x lstm_size_)
  // p_device_output_vocab_indices_single_ (minibatch_size_)
  MatrixRowToMatrixColumnKernel<<<kernel_dim, threads_per_block, 0, softmax_layer_gpu_information_.s01_>>>(p_device_d_errt_ht, p_device_d_errt_ht, p_device_d, p_device_output_vocab_indices_single_, lstm_size_, output_vocab_size);
  CudaGetLastError();

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // zero out columns
  int num_block_2 = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel_dim_2(minibatch_size_, num_block_2, 1);
  // p_device_d_errt_ht[i][j] = p_device_d_errt_ht[i][j] * p_device_output_vocab_indices_01_single_[j]
  ZeroColumnsKernel128<<<kernel_dim_2, threads_per_block, 0, softmax_layer_gpu_information_.s01_>>>(lstm_size_, p_device_d_errt_ht, p_device_output_vocab_indices_01_single_, p_device_d_errt_ht);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  // dropout mode & no attention, then dropout for p_device_d_errt_ht
  if (dropout_mode_ && !p_neural_mt_->attention_configuration_.attention_model_mode_) {
    DropoutKernel<<<256, 256, 0, softmax_layer_gpu_information_.s01_>>>(v_nodes_[index].p_device_dropout_mask_, dropout_rate_, p_device_d_errt_ht, lstm_size_ * minibatch_size_);
  }

#ifdef DEBUG_DROPOUT_2
  std::cerr<<"   copy_d_err_ht_mode_: "<<lower_layer_.copy_d_err_ht_mode_<<"\n"<<std::flush;
#endif

  // multi-GPUs stuff
  if (lower_layer_.copy_d_err_ht_mode_) {
    if (lower_layer_.lower_input_mode_) {
      cudaMemcpyAsync(lower_layer_.p_input_layer_->v_nodes_[index].p_device_d_errt_ht_, p_device_d_errt_ht, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, softmax_layer_gpu_information_.s01_);
    } else {
      cudaMemcpyAsync(lower_layer_.p_hidden_layer_->v_nodes_[index].p_device_d_errt_ht_, p_device_d_errt_ht, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, softmax_layer_gpu_information_.s01_);
    }
  } else {
    if (lower_layer_.lower_input_mode_) {
      lower_layer_.p_input_layer_->v_nodes_[index].p_device_d_errt_ht_ = p_device_d_errt_ht;
    } else {
      lower_layer_.p_hidden_layer_->v_nodes_[index].p_device_d_errt_ht_ = p_device_d_errt_ht;
    }
  }

  cudaEventRecord(softmax_layer_gpu_information_.d_error_ht_done_, softmax_layer_gpu_information_.s01_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif
}



template <typename T>
void SoftmaxLayer<T>::BackPropPreprocessGpu(T *p_device_h_t, int step) {
  p_device_h_t_ = p_device_h_t;
  p_device_output_vocab_indices_single_ = p_device_output_vocab_indices_ + step;
  p_device_output_vocab_indices_01_single_ = p_device_output_vocab_indices_01_ + step;
  p_device_output_vocab_indices_01_float_single_ = p_device_output_vocab_indices_01_float_ + step;
}


// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::BackPropPreprocessGpuMGpu(int step) {
  p_device_output_vocab_indices_single_ = p_device_output_vocab_indices_ + step;
  p_device_output_vocab_indices_01_single_ = p_device_output_vocab_indices_01_ + step;
  p_device_output_vocab_indices_01_float_single_ = p_device_output_vocab_indices_01_float_ + step;
}




// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::ClearGradients() {
  ClearGradientsGpu();
}


// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::ClearGradientsGpu() {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);
  if (truncated_softmax_mode_) {
    ;
  } else {
    cudaMemsetAsync(p_device_d_grad_, 0, output_vocab_size_ * lstm_size_ * sizeof(T), softmax_layer_gpu_information_.s00_);
    cudaMemsetAsync(p_device_b_d_grad_, 0, output_vocab_size_ * 1 * sizeof(T), softmax_layer_gpu_information_.s01_);
  }
  cudaDeviceSynchronize();
  //cudaSetDevice(0);
}


template <typename T>
double SoftmaxLayer<T>::ComputeLossGpu(int index) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  double loss = 0;
  DeviceSyncAll();

#ifdef DEBUG_DROPOUT_5
  std::cerr<<"   nce_score_mode__: "<<nce_score_mode__<<"\n"<<std::flush;
#endif

  if (nce_score_mode__) {
    std::cerr<<"nce_score_mode__ is not written!\n"<<std::flush;
    exit(EXIT_FAILURE);
  } else {
    GetPerplexityGpu(v_nodes_[index].p_device_h_t_, index);
  }
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  DeviceSyncAll();

  thrust::device_ptr<int> thrust_device_ptr = thrust::device_pointer_cast(p_device_output_vocab_indices_single_);
  thrust::device_ptr<int> thrust_device_ptr_01 = thrust::device_pointer_cast(p_device_output_vocab_indices_01_single_);
  thrust::device_ptr<double> thrust_device_ptr_sm = thrust::device_pointer_cast(p_device_outputdist_perplexity_);

  for (int i = 0; i < minibatch_size_; ++i) {
    if (1 == thrust_device_ptr_01[i]) {
      loss += thrust_device_ptr_sm[IDX2C(thrust_device_ptr[i], i, output_vocab_size_)];
    }
  }

  return loss;
}


template <typename T>
void SoftmaxLayer<T>::GetPerplexityGpu(T *p_device_h_t, int index) {

#ifdef DEBUG_DROPOUT_5
  std::cerr<<"   dropout_mode_: "<<dropout_mode_<<"\n"
           <<"   train_mode_: "<<p_neural_mt_->train_mode_<<"\n"
           <<"   grad_check_flag_: "<<p_neural_mt_->grad_check_flag_<<"\n"
           <<"   attention_model_mode_: "<<p_neural_mt_->attention_configuration_.attention_model_mode_<<"\n"
           <<std::flush;
#endif

  // for passing gradient checking with dropout
  if (dropout_mode_ && p_neural_mt_->train_mode_ && p_neural_mt_->grad_check_flag_ && !p_neural_mt_->attention_configuration_.attention_model_mode_) {
    DropoutKernel<<<256, 256, 0, softmax_layer_gpu_information_.s00_>>>(v_nodes_[index].p_device_dropout_mask_, dropout_rate_, p_device_h_t_, lstm_size_ * minibatch_size_);
  }

  DeviceSyncAll();

  // multiply the D matrix with the hidden state matrix
  T alpha = 1;
  T beta = 0;
  // stream 00
  cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s00_);
  // p_device_outputdist_ (output_vocab_size_ x minibatch_size_) = p_device_d_ (output_vocab_size_ x lstm_size_) * 
  //                                                               p_device_h_t (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_vocab_size_, minibatch_size_, lstm_size_, &alpha, p_device_d_, output_vocab_size_, p_device_h_t, lstm_size_, &beta, p_device_outputdist_, output_vocab_size_), "SoftmaxLayer::GetPerplexityGpu p_device_outputdist_ failed\n");

  // add the bias vector to the matrix
  int threads_per_block = 128;
  int num_block = (output_vocab_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel_dim(minibatch_size_, num_block, 1);
  // p_device_outputdist_[i][j] = p_device_outputdist_[i][j] + p_device_b_d_[i]
  MatrixBiasKernel<<<kernel_dim, threads_per_block, 0, softmax_layer_gpu_information_.s00_>>>(output_vocab_size_, p_device_outputdist_, p_device_b_d_, p_device_outputdist_);
  CudaGetLastError("SoftmaxLayer::GetPerplexityGpu p_device_outputdist_ failed");

  if (dump_nce_stats__) {
    std::cerr<<"   dump_nce_stats is not written!\n";
    exit(EXIT_FAILURE);
  } else {
    OutputdistPerplexityKernel<<<minibatch_size_, SOFTMAX_THREADS, 0, softmax_layer_gpu_information_.s00_>>>(p_device_outputdist_perplexity_, p_device_outputdist_, output_vocab_size_, false, NULL);
    CudaGetLastError("SoftmaxLayer::GetPerplexityGpu OutputdistPerplexityKernel");
  }
  cudaDeviceSynchronize();
}



// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::PreprocessGpuVocabIndices(int *p_host_output_vocab_indices_target, int current_target_length) {

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  cudaMemcpy(p_device_output_vocab_indices_, p_host_output_vocab_indices_target, minibatch_size_ * current_target_length * sizeof(int), cudaMemcpyHostToDevice);

  int threads_per_block = 128;
  int blocks_per_grid = 128;
  VocabSoftmaxKernel<<<blocks_per_grid,threads_per_block>>>(p_device_output_vocab_indices_, p_device_output_vocab_indices_01_, p_device_output_vocab_indices_01_float_, current_target_length * minibatch_size_);
  CudaGetLastError("SoftmaxLayer::p_device_output_vocab_indices_ preprocess");
  cudaSetDevice(0);
}


// CHECK: OK //
template <typename T>
T *SoftmaxLayer<T>::GetHTPtr(int index) {
  return v_nodes_[index].p_device_h_t_;
}


// CHECK: OK //
template <typename T>
void SoftmaxLayer<T>::SetHTPtr(int index, T *p_device_h_t) {
  v_nodes_[index].p_device_h_t_ = p_device_h_t;
}


template <typename T>
cudaEvent_t SoftmaxLayer<T>::GetErrHTEvent() {
  return softmax_layer_gpu_information_.d_error_ht_done_;
}


template <typename T>
T *SoftmaxLayer<T>::GetDistPtr() {
  return p_device_outputdist_;
}


template <typename T>
void SoftmaxLayer<T>::GetDistributionGpu(int output_vocab_size, T *p_device_outputdist, T *p_device_d, T *p_device_b_d, T *p_device_h_t) {
    
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  // wait until previous h_t, d, and b_d gradients are finished because they need the output dist
  // also wait until the previous backpropinit has finished
  cudaStreamWaitEvent(softmax_layer_gpu_information_.s00_, softmax_layer_gpu_information_.d_error_ht_done_, 0);
  cudaStreamWaitEvent(softmax_layer_gpu_information_.s00_, softmax_layer_gpu_information_.d_d_grad_done_, 0);
  cudaStreamWaitEvent(softmax_layer_gpu_information_.s00_, softmax_layer_gpu_information_.d_b_d_grad_done_, 0);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // multiply the D matrix with the hidden state matrix
  T alpha = 1;
  T beta = 0;

  // stream 00
  cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s00_);

  // p_device_outputdist (output_vocab_size x minibatch_size_) = p_device_d (output_vocab_size x lstm_size_) *
  //                                                             p_device_h_t (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_vocab_size, minibatch_size_, lstm_size_, &alpha, p_device_d, output_vocab_size, p_device_h_t, lstm_size_, &beta, p_device_outputdist, output_vocab_size), "SoftmaxLayer::GetDistributionGpu p_device_outputdist failed\n");

  // add the bias vector to the matrix
  int threads_per_block = 128;
  int num_block = (output_vocab_size + threads_per_block - 1) / threads_per_block;
  dim3 kernel_dim(minibatch_size_, num_block, 1);
  // p_device_outputdist[IDX2C(i, j, output_vocab_size)] = p_device_outputdist[IDX2C(i, j, output_vocab_size)] + p_device_b_d[i]
  MatrixBiasKernel<<<kernel_dim, threads_per_block, 0, softmax_layer_gpu_information_.s00_>>>(output_vocab_size, p_device_outputdist, p_device_b_d, p_device_outputdist);
  CudaGetLastError();


  // this is for decoding
  if (pre_normalization_mode__) {
    DeviceSyncAll();
    // now exp all elements
    thrust::for_each(thrust_device_outputdist_.begin(), thrust_device_outputdist_.end(), ExpFunctorGpu());

    cudaEventRecord(softmax_layer_gpu_information_.output_dist_done_, softmax_layer_gpu_information_.s00_);
    return;
  }


  if (!softmax_scaled_mode_) {

    cudaDeviceSynchronize();

    // exp every element in the outputDist matrix
    thrust::for_each(thrust_device_outputdist_.begin(), thrust_device_outputdist_.end(), ExpFunctorGpu());

    // get the normalization vector
    // p_device_normalization_ (minibatch_size_) = p_device_outputdist^T (minibatch_size_ x output_vocab_size_) * 
    //                                             p_device_ones_ (output_vocab_size_)
    CublasErrorWrapper(CublasGemvWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_T, output_vocab_size, minibatch_size_, &alpha, p_device_outputdist, output_vocab_size, p_device_ones_, 1, &beta, p_device_normalization_, 1), "SoftmaxLayer::GetDistributionGpu p_device_normalization_ failed\n");

    // invert the values in the normalization matrix
    thrust::for_each(thrust_device_normalization_.begin(), thrust_device_normalization_.end(), InvFunctorGpu());

    // renormalize outputdist with the normalization vector
    // p_device_outputdist (output_vocab_size x minibatch_size_) = p_device_outputdist (output_vocab_size x minibatch_size_) *
    //                                                             diag(p_device_normalization_) (minibatch_size_ x minibatch_size_)
    CublasErrorWrapper(CublasDgmmWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_SIDE_RIGHT, output_vocab_size, minibatch_size_, p_device_outputdist, output_vocab_size, p_device_normalization_, 1, p_device_outputdist, output_vocab_size), "SoftmaxLayer::GetDistributionGpu p_device_outputdist 2 failed\n");
    cudaDeviceSynchronize();

  } else {

    // softmax function
    // p_device_outputdist[i][j] (output_vocab_size x lstm_size_) = (exp(p_device_outputdist[i][j] - (max_k in every minibatch))) /
    //                                                              sum_i'(exp(p_device_outputdist[i'][j] - (max_k in every minibatch)))
    OutputdistOverflowPreventionKernel<<<minibatch_size_, SOFTMAX_THREADS, 0, softmax_layer_gpu_information_.s00_>>>(p_device_outputdist, p_device_outputdist, output_vocab_size);
    CudaGetLastError();
  }

  if (training_perplexity_mode_) {
    // p_device_train_perplexity_[0] += log((double)p_device_outputdist[p_device_output_vocab_indices_single_[i]][i], 
    // i (is minibatch) = (0 -> 63)
    TrainPerplexityKernel<<<1, 1, 0, softmax_layer_gpu_information_.s00_>>>(p_device_output_vocab_indices_single_, p_device_output_vocab_indices_01_single_, p_device_outputdist, p_device_train_perplexity_, minibatch_size_, output_vocab_size);
  }

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaEventRecord(softmax_layer_gpu_information_.output_dist_done_, softmax_layer_gpu_information_.s00_);
}


template <typename T>
void SoftmaxLayer<T>::ComputeDGradientGpu(int output_vocab_size, T *p_device_outputdist, T *p_device_d_grad, T *p_device_h_t) {
#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  // zero out p_device_h_t_
  cudaStreamWaitEvent(softmax_layer_gpu_information_.s02_, softmax_layer_gpu_information_.output_dist_done_, 0);
  int threads_per_block = 128;
  int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel_dim(minibatch_size_, num_block, 1);
  // p_device_h_t[i][j] = p_device_h_t[i][j] * p_device_output_vocab_indices_01_single_[j]
  ZeroColumnsKernel128<<<kernel_dim, threads_per_block, 0, softmax_layer_gpu_information_.s02_>>>(lstm_size_, p_device_h_t, p_device_output_vocab_indices_01_single_, p_device_h_t);
  CudaGetLastError();

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  T alpha = -1;
  T beta = 1;
  // stream 02
  cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s02_);
  // multiple p_device_outputdist and p_device_h_t
  // p_device_d_grad (output_vocab_size x lstm_size_) = -1 * p_device_outputdist (output_vocab_size x minibatch_size_) * 
  //                                                    p_device_h_t^T (minibatch_size_ x lstm_size_) +
  //                                                    p_device_d_grad
  CublasErrorWrapper(CublasGemmWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, output_vocab_size, lstm_size_, minibatch_size_, &alpha, p_device_outputdist, output_vocab_size, p_device_h_t, lstm_size_, &beta, p_device_d_grad, output_vocab_size), "SoftmaxLayer::ComputeDGradientGpu p_device_d_grad failed\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // Add columns of p_device_h_t to p_device_d_grad
  // p_device_d_grad[p_device_output_vocab_indices_single_[i]][j] += p_device_h_t[j][i]
  // output_vocab_size x lstm_size_             lstm_size_ x minibatch_size_
  MatrixColumnToMatrixRowKernel<<<kernel_dim, threads_per_block, 0, softmax_layer_gpu_information_.s02_>>>(p_device_d_grad, p_device_h_t, p_device_d_grad, p_device_output_vocab_indices_single_, lstm_size_, output_vocab_size);
  CudaGetLastError();

  cudaEventRecord(softmax_layer_gpu_information_.d_d_grad_done_, softmax_layer_gpu_information_.s02_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif
  
  cudaSetDevice(0);
}


template <typename T>
void SoftmaxLayer<T>::ComputeBDGradientGpu(int output_vocab_size, T *p_device_outputdist, T *p_device_b_d_grad) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);
  cudaStreamWaitEvent(softmax_layer_gpu_information_.s03_, softmax_layer_gpu_information_.output_dist_done_, 0);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  T alpha = -1;
  T beta = 1;
  // stream 03
  cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s03_);
  // p_device_b_d_grad (output_vocab_size x 1) = (-1) * p_device_outputdist (output_vocab_size x minibatch_size) *
  //                                             p_device_output_vocab_indices_01_float_single_ (minibatch_size_ x 1) +
  //                                             p_device_b_d_grad
  CublasErrorWrapper(CublasGemvWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, output_vocab_size, minibatch_size_, &alpha, p_device_outputdist, output_vocab_size, p_device_output_vocab_indices_01_float_single_, 1, &beta, p_device_b_d_grad, 1), "SoftmaxLayer::ComputeBDGradientGpu p_device_b_d_grad failed");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // add ones
  int threads_per_block = 128;
  int num_block = (minibatch_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel_dim(1, num_block, 1);
  // p_device_b_d_grad[p_device_output_vocab_indices_single_[i]] += 1
  AddOnesBDGrad<<<kernel_dim, threads_per_block, 0, softmax_layer_gpu_information_.s03_>>>(p_device_b_d_grad, p_device_output_vocab_indices_01_single_, p_device_output_vocab_indices_single_, minibatch_size_);

  cudaEventRecord(softmax_layer_gpu_information_.d_b_d_grad_done_, softmax_layer_gpu_information_.s03_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif
}

template <typename T>
void SoftmaxLayer<T>::CheckAllGradients(T epsilon) {
  CheckAllGradientsGpu(epsilon);
  return;
}


template <typename T>
void SoftmaxLayer<T>::CheckAllGradientsGpu(T epsilon) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);
  std::cerr<<"#### GRADIENT CHECKING FOR SOFTMAX LAYER GPU ####\n";
  std::cerr<<"GRADIENT CHECKING FOR p_device_d_\n";
  CheckGradientGpu(epsilon, p_device_d_, p_device_d_grad_, output_vocab_size_, lstm_size_);
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  std::cerr<<"GRADIENT CHECKING FOR p_device_b_d_\n";
  CheckGradientGpu(epsilon, p_device_b_d_, p_device_b_d_grad_, output_vocab_size_, 1);
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  cudaSetDevice(0);
  return;
}


template <typename T>
void SoftmaxLayer<T>::CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);
  cudaDeviceSynchronize();
  thrust::device_ptr<T> p_thrust_device_mat = thrust::device_pointer_cast(p_device_mat);
  thrust::device_ptr<T> p_thrust_device_grad = thrust::device_pointer_cast(p_device_grad);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      T loss = 0;
      p_thrust_device_mat[IDX2C(i, j, rows)] += epsilon;
      loss = p_neural_mt_->GetError(true);
      cudaSetDevice(softmax_layer_gpu_information_.device_number_);
      cudaDeviceSynchronize();

      p_thrust_device_mat[IDX2C(i, j, rows)] += -2 * epsilon;
      loss -= p_neural_mt_->GetError(true);
      cudaSetDevice(softmax_layer_gpu_information_.device_number_);
      cudaDeviceSynchronize();

      p_thrust_device_mat[IDX2C(i, j, rows)] += epsilon;
      std::cerr<<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"  my gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n";
      if ((std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))) > 1 / (T)1000.0 || (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))) > 1 / 1000.0) {
        std::cerr<<"Gradient for gradient check: "<<loss / (2 * epsilon)<<"\n";
        std::cerr<<"My gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n";
        std::cerr<<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"\n";
        std::cerr<<"Gradient difference (Equation 2): "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))<<"\n\n";
      }
    }
  }
  return;
}


template <typename T>
void SoftmaxLayer<T>::CalculateGlobalNorm() {

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  ScaleFunctor unary_op(minibatch_size_);
  thrust::for_each(p_thrust_device_d_grad_, p_thrust_device_d_grad_ + output_vocab_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_d_grad_, p_thrust_device_b_d_grad_ + output_vocab_size_ * 1, unary_op);
  
  NormClipGpuV2P1(p_thrust_device_d_grad_, p_device_d_grad_, norm_clip_, output_vocab_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_b_d_grad_, p_device_b_d_grad_, norm_clip_, output_vocab_size_ * 1, p_device_result_tmp_, p_device_result_);

  DeviceSyncAll();
}

template <typename T>
void SoftmaxLayer<T>::UpdateGlobalParams() {

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  T alpha = learning_rate_;
  T beta = 1;

#ifdef DEBUG_DROPOUT_4
  std::cerr<<"   norm_clip_: "<<norm_clip_<<"\n"<<std::flush;
#endif

  NormClipGpuV2P2(p_thrust_device_d_grad_, p_device_d_grad_, norm_clip_, output_vocab_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_b_d_grad_, p_device_b_d_grad_, norm_clip_, output_vocab_size_ * 1, p_device_result_tmp_, p_device_result_);

  cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s00_);
  CublasErrorWrapper(CublasGeamWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_vocab_size_, lstm_size_, &alpha, p_device_d_grad_, output_vocab_size_, &beta, p_device_d_, output_vocab_size_, p_device_d_, output_vocab_size_), "CUBLAS addition update parameter failed\n");

  cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s01_);
  CublasErrorWrapper(CublasGeamWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_vocab_size_, 1, &alpha, p_device_b_d_grad_, output_vocab_size_, &beta, p_device_b_d_, output_vocab_size_, p_device_b_d_, output_vocab_size_), "CUBLAS addition update parameter failed\n");

  DeviceSyncAll();
}


template <typename T>
void SoftmaxLayer<T>::UpdateWeights() {
  UpdateWeightsGpu();
}


template <typename T>
void SoftmaxLayer<T>::UpdateWeightsGpu() {

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  ScaleFunctor unary_op(minibatch_size_);

#ifdef DEBUG_DROPOUT_4
  std::cerr<<"   truncated_softmax_mode_: "<<truncated_softmax_mode_<<"\n"<<std::flush;
#endif

  if (truncated_softmax_mode_) {
    thrust::for_each(p_thrust_device_subset_d_grad_, p_thrust_device_subset_d_grad_ + trunc_size_ * lstm_size_, unary_op);
    thrust::for_each(p_thrust_device_subset_b_d_grad_, p_thrust_device_subset_b_d_grad_ + trunc_size_ * 1, unary_op);
  } else {
    // p_thrust_device_d_grad_[i] = (1/minibatch_size_) * p_thrust_device_d_grad_[i]
    thrust::for_each(p_thrust_device_d_grad_, p_thrust_device_d_grad_ + output_vocab_size_ * lstm_size_, unary_op);
    // p_thrust_device_b_d_grad_[i] = (1/minibatch_size_) * p_thrust_device_b_d_grad_[i]
    thrust::for_each(p_thrust_device_b_d_grad_, p_thrust_device_b_d_grad_ + output_vocab_size_ * 1, unary_op);
  }

#ifdef DEBUG_DROPOUT_4
  std::cerr<<"   individual_grad_clip_mode__: "<<individual_grad_clip_mode__<<"\n"<<std::flush;
  std::cerr<<"   individual_norm_clip_threshold__: "<<individual_norm_clip_threshold__<<"\n"<<std::flush;
  std::cerr<<"   gradient_clip_mode_: "<<gradient_clip_mode_<<"\n"<<std::flush;
  std::cerr<<"   norm_clip_: "<<norm_clip_<<"\n"<<std::flush;
  std::cerr<<"   train_target_output_embedding_mode__: "<<train_target_output_embedding_mode__<<"\n"<<std::flush;
#endif

  if (individual_grad_clip_mode__) {
    // p_device_d_grad_[i] in [-individual_norm_clip_threshold__, individual_norm_clip_threshold__]
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, softmax_layer_gpu_information_.s00_>>>(p_device_d_grad_, individual_norm_clip_threshold__, lstm_size_ * output_vocab_size_);
    // p_device_b_d_grad_[i] in [-individual_norm_clip_threshold__, individual_norm_clip_threshold__]
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, softmax_layer_gpu_information_.s00_>>>(p_device_b_d_grad_, individual_norm_clip_threshold__, output_vocab_size_ * 1);
    DeviceSyncAll();
  }

  if (gradient_clip_mode_) {

    if (truncated_softmax_mode_) {
      NormClipGpuV2(p_thrust_device_subset_d_grad_, p_device_subset_d_grad_, norm_clip_, trunc_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
      NormClipGpuV2(p_thrust_device_subset_b_d_grad_, p_device_subset_b_d_grad_, norm_clip_, trunc_size_ * 1, p_device_result_tmp_, p_device_result_);
    } else {
      NormClipGpuV2(p_thrust_device_d_grad_, p_device_d_grad_, norm_clip_, output_vocab_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
      NormClipGpuV2(p_thrust_device_b_d_grad_, p_device_b_d_grad_, norm_clip_, output_vocab_size_ * 1, p_device_result_tmp_, p_device_result_);
    }
  }

  T alpha = learning_rate_;
  T beta = 1;

  if (truncated_softmax_mode_) {
    // p_device_d_
    TruncDGradNonshort<<<256, 256, 0, softmax_layer_gpu_information_.s00_>>>(p_device_subset_d_grad_, p_device_d_, p_device_truncated_vocab_mapping_, lstm_size_, trunc_size_, output_vocab_size_, learning_rate_, shortlist_size_);
    CudaGetLastError();
    TruncDGradShort<<<256, 256, 0, softmax_layer_gpu_information_.s00_>>>(p_device_subset_d_grad_, p_device_subset_d_, lstm_size_, shortlist_size_, learning_rate_, trunc_size_);
    CudaGetLastError();

    // p_device_b_d_
    // this is d_d, but with lstm size of 1
    TruncDGradNonshort<<<256, 256, 0, softmax_layer_gpu_information_.s01_>>>(p_device_subset_b_d_grad_, p_device_b_d_, p_device_truncated_vocab_mapping_, 1, trunc_size_, output_vocab_size_, learning_rate_, shortlist_size_);
    CudaGetLastError();
    TruncDGradShort<<<256, 256, 0, softmax_layer_gpu_information_.s01_>>>(p_device_subset_b_d_grad_, p_device_subset_b_d_, 1, shortlist_size_, learning_rate_, trunc_size_);
    CudaGetLastError();
  } else {

    if (train_target_output_embedding_mode__) {
      // stream 00
      cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s00_);
      // p_device_d_ = learning_rate_ * p_device_d_grad_ + p_device_d_
      // output_vocab_size_ x lstm_size_
      CublasErrorWrapper(CublasGeamWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_vocab_size_, lstm_size_, &alpha, p_device_d_grad_, output_vocab_size_, &beta, p_device_d_, output_vocab_size_, p_device_d_, output_vocab_size_), "SoftmaxLayer::UpdateWeightsGpu p_device_d_ failed\n");

      // stream 01
      cublasSetStream(softmax_layer_gpu_information_.handle_, softmax_layer_gpu_information_.s01_);
      // p_device_b_d_ = learning_rate_ * p_device_b_d_grad_ + p_device_b_d_
      // output_vocab_size_ x 1
      CublasErrorWrapper(CublasGeamWrapper(softmax_layer_gpu_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, output_vocab_size_, 1, &alpha, p_device_b_d_grad_, output_vocab_size_, &beta, p_device_b_d_, output_vocab_size_, p_device_b_d_, output_vocab_size_), "CUBLAS addition update parameter failed\n");
    }
  }

  DeviceSyncAll();
}


template <typename T>
double SoftmaxLayer<T>::GetTrainPerplexity() {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);
  double tmp_perp;
  cudaMemcpy(&tmp_perp, p_device_train_perplexity_, 1 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemset(p_device_train_perplexity_, 0, 1 * sizeof(double));
  return tmp_perp;
}


template <typename T>
void SoftmaxLayer<T>::GetDistributionGpuDecoderWrapper() {
  GetDistributionGpu(output_vocab_size_, p_device_outputdist_, p_device_d_, p_device_b_d_, p_device_h_t_);
}


template <typename T>
void SoftmaxLayer<T>::UpdateLearningRate(T learning_rate) {
  learning_rate_ = learning_rate;
}


template <typename T>
void SoftmaxLayer<T>::LoadWeights(std::ifstream &input_stream) {
  LoadWeightsGpu(input_stream);
}

template <typename T>
void SoftmaxLayer<T>::LoadWeightsGpu(std::ifstream &input_stream) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  ReadMatrixGpu(p_device_d_, output_vocab_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_d_, output_vocab_size_, 1, input_stream);
}

template <typename T>
void SoftmaxLayer<T>::DumpWeights(std::ofstream &output) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  if (truncated_softmax_mode_) {
    LoadShortlistD<<<256, 256>>>(p_device_subset_d_, p_device_d_, lstm_size_, trunc_size_, output_vocab_size_, shortlist_size_);
    LoadShortlistD<<<256, 256>>>(p_device_subset_b_d_, p_device_b_d_, 1, trunc_size_, output_vocab_size_, shortlist_size_);
    cudaDeviceSynchronize();
  }

  DumpWeightsGpu(output);
}


template <typename T>
void SoftmaxLayer<T>::DumpWeightsGpu(std::ofstream &output) {
  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  WriteMatrixGpu(p_device_d_, output_vocab_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_d_, output_vocab_size_, 1, output);
}




////////////////////////////////////// NceNode //////////////////////////////////////
////////////////////////////////////// NceNode //////////////////////////////////////
////////////////////////////////////// NceNode //////////////////////////////////////
template <typename T>
class NceNode {
public:
  T *p_device_h_t_;
  int index_;
};




////////////////////////////////////// NceLayer //////////////////////////////////////
////////////////////////////////////// NceLayer //////////////////////////////////////
////////////////////////////////////// NceLayer //////////////////////////////////////
template <typename T>
class NceLayer : public BaseLossLayer<T> {

public:
  SoftmaxLayerGpuInformation softmax_layer_gpu_information_;


public:
  NeuralMachineTranslation<T> *p_neural_mt_;

public:
  int lstm_size_;
  int minibatch_size_;
  int output_vocab_size_;
  int num_negative_samples_;
  int longest_sentence_;
  T learning_rate_;
  bool dropout_mode_;
  T dropout_rate_;
  bool clip_gradients_mode_; // If true then clip gradients
  T norm_clip_;              // for gradient clipping


public:
  bool share_samples_mode_ = true; // share the noise samples across the minibatch



public:
  T *p_device_d_;                          // lstm_size_ x output_vocab_size, output embeddings, in softmax it is the other way around
  T *p_device_b_d_;                        // output_vocab_size_ x 1, bias
  T *p_device_dot_products_;
  T *p_device_outputdist_;                 // output_vocab_size_ x minibatch_size
  T *p_device_b_d_grad_;                   // output_vocab_size_
  T *p_device_ones_;                       // minibatch_size_


public:
  T *p_device_tmp_d_grad_;

public:
  std::vector<NceNode<T>> v_nodes_;

public:
  NceLayer() {}


public:
  SoftmaxLayerGpuInformation InitGpu(int device_number);

  void InitLossLayer(NeuralMachineTranslation<precision> *neural_mt, GlobalConfiguration &configuration);

public:
  void InitLowerTransferLayer(bool lower_input, bool copy_d_err_ht, InputToHiddenLayer<T> *p_input_layer, HiddenToHiddenLayer<T> *p_hidden_layer);

public:
  void ForwardProp(int index);

public:
  void BackProp1(int index);
  void BackProp2(int index);

public:
  void BackPropPreprocessGpu(T *p_device_h_t, int step);
  void BackPropPreprocessGpuMGpu(int step);

public:
  void ClearGradients();

public:
  double ComputeLossGpu(int index);


public:
  void PreprocessGpuVocabIndices(int *p_host_output_vocab_indices_target, int current_target_length);

public:
  T *GetHTPtr(int index);
  void SetHTPtr(int index, T *p_device_h_t);

public:
  cudaEvent_t GetErrHTEvent();

public:
  void CheckAllGradients(T epsilon);

public:
  void CalculateGlobalNorm();

public:
  void UpdateGlobalParams();

public:
  void UpdateWeights();

public:
  double GetTrainPerplexity();

public:
  void GetDistributionGpuDecoderWrapper();

public:
  void UpdateLearningRate(T learning_rate);

public:
  void LoadWeights(std::ifstream &input_stream);

public:
  void DumpWeights(std::ofstream &output);

public:
  T *GetDistPtr();
};


/////////////////// Implementations for Class Template NceLayer ///////////////////
template <typename T>
SoftmaxLayerGpuInformation NceLayer<T>::InitGpu(int device_number) {
    softmax_layer_gpu_information_.Init(device_number);
    return softmax_layer_gpu_information_;
}


template <typename T>
void NceLayer<T>::InitLossLayer(NeuralMachineTranslation<precision> *p_neural_mt, GlobalConfiguration &configuration){
  std::cerr<<"\n\nInitLossLayer: NCE loss is not written!\n\n"<<std::flush;

  lstm_size_ = configuration.lstm_size_;
  minibatch_size_ = configuration.minibatch_size_;
  output_vocab_size_ = configuration.target_vocab_size_;
  num_negative_samples_ = configuration.negative_samples_number_;
  longest_sentence_ = configuration.longest_sentence_;
  p_neural_mt_ = p_neural_mt;
  learning_rate_ = configuration.learning_rate_;
  dropout_mode_ = configuration.dropout_mode_;
  dropout_rate_ = configuration.dropout_rate_;
  clip_gradients_mode_ = configuration.clip_gradient_mode_;
  norm_clip_ = configuration.norm_clip_;
  share_samples_mode_ = configuration.share_samples_mode_;

  cudaSetDevice(softmax_layer_gpu_information_.device_number_);

  T *p_host_tmp;
  FullMatrixSetup(&p_host_tmp, &p_device_outputdist_, output_vocab_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_d_, lstm_size_, output_vocab_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_b_d_, output_vocab_size_, 1);

  if (share_samples_mode_) {
    FullMatrixSetup(&p_host_tmp, &p_device_tmp_d_grad_, lstm_size_, num_negative_samples_);
    FullMatrixSetup(&p_host_tmp, &p_device_dot_products_, num_negative_samples_ + minibatch_size_, minibatch_size_);
  } else {
    FullMatrixSetup(&p_host_tmp, &p_device_dot_products_, num_negative_samples_ + 1, minibatch_size_);
  }

  FullVectorSetup(&p_host_tmp, &p_device_b_d_grad_, output_vocab_size_);
  FullVectorSetupOnes(&p_host_tmp, &p_device_ones_, minibatch_size_);

  return;
}


template <typename T>
void NceLayer<T>::InitLowerTransferLayer(bool lower_input, bool copy_d_err_ht, InputToHiddenLayer<T> *p_input_layer, HiddenToHiddenLayer<T> *p_hidden_layer) {
  std::cerr<<"\n\nInitLowerTransferLayer: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::ForwardProp(int index) {
  std::cerr<<"\n\nForwardProp: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::BackProp1(int index) {
  std::cerr<<"\n\nBackProp1: NCE loss is not written!\n\n"<<std::flush;
  return;
}

template <typename T>
void NceLayer<T>::BackProp2(int index) {
  std::cerr<<"\n\nBackProp2: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::BackPropPreprocessGpu(T *p_device_h_t, int step) {
  std::cerr<<"\n\nBackPropPreprocessGpu: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::BackPropPreprocessGpuMGpu(int step) {
  std::cerr<<"\n\nBackPropPreprocessGpuMGpu: NCE loss is not written!\n\n"<<std::flush;
  return;
}




template <typename T>
void NceLayer<T>::ClearGradients() {
  std::cerr<<"\n\nClearGradients: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
double NceLayer<T>::ComputeLossGpu(int index) {
  std::cerr<<"\n\nComputeLossGpu: NCE loss is not written!\n\n"<<std::flush;
  double loss = 0.0;
  return loss;
}



template <typename T>
void NceLayer<T>::PreprocessGpuVocabIndices(int *p_host_output_vocab_indices_target, int current_target_length) {
  std::cerr<<"\n\nPreprocessGpuVocabIndices: NCE loss is not written!\n\n"<<std::flush;
  return;
}

template <typename T>
T *NceLayer<T>::GetHTPtr(int index) {
  std::cerr<<"\n\nGetHTPtr: NCE loss is not written!\n\n"<<std::flush;
  return v_nodes_[index].p_device_h_t_;
}


template <typename T>
void NceLayer<T>::SetHTPtr(int index, T *p_device_h_t) {
  std::cerr<<"\n\nSetHTPtr: NCE loss is not written!\n\n"<<std::flush;
  v_nodes_[index].p_device_h_t_ = p_device_h_t;
}

template <typename T>
cudaEvent_t NceLayer<T>::GetErrHTEvent() {
  std::cerr<<"\n\nGetErrHTEvent: NCE loss is not written!\n\n"<<std::flush;
  return softmax_layer_gpu_information_.d_error_ht_done_;
}


template <typename T>
void NceLayer<T>::CheckAllGradients(T epsilon) {
  std::cerr<<"\n\nCheckAllGradients: NCE loss is not written!\n\n"<<std::flush;
  return;
}

template <typename T>
void NceLayer<T>::CalculateGlobalNorm() {
  std::cerr<<"\n\nCalculateGlobalNorm: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::UpdateGlobalParams() {
  std::cerr<<"\n\nUpdateGlobalParams: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::UpdateWeights() {
  std::cerr<<"\n\nUpdateWeights: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
double NceLayer<T>::GetTrainPerplexity() {
  std::cerr<<"\n\nGetTrainPerplexity: NCE loss is not written!\n\n"<<std::flush;
  double f = 0;
  return f;
}


template <typename T>
void NceLayer<T>::GetDistributionGpuDecoderWrapper() {
  std::cerr<<"\n\nGetDistributionGpuDecoderWrapper: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::UpdateLearningRate(T learning_rate) {
  std::cerr<<"\n\nUpdateLearningRate: NCE loss is not written!\n\n"<<std::flush;
  return;
}


template <typename T>
void NceLayer<T>::LoadWeights(std::ifstream &input_stream) {
  std::cerr<<"\n\nLoadWeights: NCE loss is not written!\n\n"<<std::flush;
  return;
}

template <typename T>
void NceLayer<T>::DumpWeights(std::ofstream &output) {
  std::cerr<<"\n\nDumpWeights: NCE loss is not written!\n\n"<<std::flush;
  return;
}

template <typename T>
T *NceLayer<T>::GetDistPtr() {
  std::cerr<<"\n\nGetDistPtr: NCE loss is not written!\n\n"<<std::flush;
  return p_device_outputdist_;
}


}


#endif




