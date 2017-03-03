/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#ifndef HIDDEN_TO_HIDDEN_LAYER_H_
#define HIDDEN_TO_HIDDEN_LAYER_H_

#include "layer_gpu.h"
#include "deep_rnn.h"
#include "global_configuration.h"
#include "layer_attention.h"
#include "node_hidden_to_hidden.h"
#include "layer_transfer.h"

#include <Eigen/Core>

namespace neural_machine_translation {

template <typename T>
class NeuralMachineTranslation;


template <typename T>
class LowerTransferLayer;


template <typename T>
class HiddenToHiddenLayer {

public:
  std::vector<LstmHiddenHiddenNode<T>> v_nodes_;            // stores all the LSTM nodes for forward and backward propagation
  
  // GPU parameters
public:
  LayerGpuInformation hidden_hidden_layer_information_;

public:
  LowerTransferLayer<T> lower_layer_;
  
public:
  // host pointers
  T *p_host_tmp1_;
  T *p_host_tmp2_;
  T *p_host_tmp3_;
  T *p_host_tmp4_;


  T *p_host_w_ho_;
  T *p_host_w_hf_;
  T *p_host_w_hi_;
  T *p_host_w_hc_;

  T *p_host_w_ho_grad_;
  T *p_host_w_hf_grad_;
  T *p_host_w_hi_grad_;
  T *p_host_w_hc_grad_;

  T *p_host_m_i_grad_;
  T *p_host_m_f_grad_;
  T *p_host_m_o_grad_;
  T *p_host_m_c_grad_;

  T *p_host_b_i_grad_;
  T *p_host_b_f_grad_;
  T *p_host_b_o_grad_;
  T *p_host_b_c_grad_;

  T *p_host_ones_minibatch_;

  T *p_host_m_i_;
  T *p_host_m_f_;
  T *p_host_m_o_;
  T *p_host_m_c_;

  T *p_host_b_i_;
  T *p_host_b_f_;
  T *p_host_b_o_;
  T *p_host_b_c_;
  
  T *p_host_tmp5_;
  T *p_host_tmp6_;
  T *p_host_tmp7_;
  T *p_host_tmp8_;

public:
  // Convert this into 0/1's and to one with no -1's as indices
  int *p_host_input_vocab_indices_;
  int *p_device_input_vocab_indices_;    // minibatch_size_ x longest_sent_, (-1 -1 -1 2 179 0 10 22 0)

public:
  int current_length_;                   // This is the current length of this target or source sequence
  
public:
  // contains the entire input sequence, use pointer arithmetic to pass correct segments to LSTM cells
  int *p_host_input_vocab_indices_01_full_;
  int *p_device_input_vocab_indices_01_full_;  // minibatch_size_ x longest_sent_, use minibatch_size_ x 1
                                               // (-1 -1 -1 2 179 0 10 22 0) => (0 0 0 1 1 1 1 1 1)
  
public:
  // for setting inital cell and hidden state values
  T *p_host_init_hidden_vector_;
  T *p_host_init_cell_vector_;
  T *p_device_init_hidden_vector_;                  // lstm_size_ x minibatch_size_, for init LstmHiddenHiddenNode.v_nodes_[0].p_device_h_t_
  T *p_device_init_cell_vector_;                    // lstm_size_ x minibatch_size_, for init LstmHiddenHiddenNode.v_nodes_[0].p_device_c_t_

public:
  T *p_host_init_d_errn_to_tp1_ht_;
  T *p_host_init_d_errn_to_tp1_ct_;
  T *p_device_init_d_errn_to_tp1_ht_;               // lstm_size_ x minibatch_size_,
  T *p_device_init_d_errn_to_tp1_ct_;               // lstm_size_ x minibatch_size_,

public:
  // pass this in for backprop gpu prep from surce size (all zero error matrix)
  T *p_device_zeros_;

public:
  // stuff for morm clipping
  T *p_device_result_;
  T *p_device_result_tmp_;

public:
  // device pointers 
  T *p_device_tmp1_;                // lstm_size_ x minibatch_size_, this = p_hidden_to_hidden_layer_->p_device_m_i_ * p_device_h_t_below_, for p_device_i_t_
  T *p_device_tmp2_;                // ...                         , this = p_hidden_to_hidden_layer_->p_device_w_hi_ * p_device_h_t_prev_, for p_device_i_t_
  T *p_device_tmp3_;                // ...                         , this = p_hidden_to_hidden_layer_->p_device_m_f_ * p_device_h_t_below_, for p_device_f_t_
  T *p_device_tmp4_;                // ...                         , this = p_hidden_to_hidden_layer_->p_device_w_hf_ * p_device_h_t_prev_, for p_device_f_t_

  T *p_device_w_ho_;                // lstm_size_ x lstm_size_, used for p_device_h_t_prev_
  T *p_device_w_hf_;                // lstm_size_ x lstm_size_, used for p_device_h_t_prev_
  T *p_device_w_hi_;                // lstm_size_ x lstm_size_, used for p_device_h_t_prev_
  T *p_device_w_hc_;                // lstm_size_ x lstm_size_, used for p_device_h_t_prev_

  T *p_device_w_ho_grad_;           // lstm_size_ x lstm_size_
  T *p_device_w_hf_grad_;           // ...
  T *p_device_w_hi_grad_;           // ...
  T *p_device_w_hc_grad_;           // ...

  T *p_device_m_i_grad_;            // lstm_size_ x lstm_size_
  T *p_device_m_f_grad_;            // ...
  T *p_device_m_o_grad_;            // ...
  T *p_device_m_c_grad_;            // ... 
  
  T *p_device_b_i_grad_;            // lstm_size_ x 1
  T *p_device_b_f_grad_;            // ...
  T *p_device_b_o_grad_;            // ...
  T *p_device_b_c_grad_;            // ...

  T *p_device_ones_minibatch_;  // minibatch_size_ x 1

  T *p_device_m_i_;             // lstm_size_ x lstm_size_, used for p_device_h_t_below_
  T *p_device_m_f_;             // ...
  T *p_device_m_o_;             // ...
  T *p_device_m_c_;             // ...

  T *p_device_b_i_;             // lstm_size_ x 1
  T *p_device_b_f_;             // ...
  T *p_device_b_o_;             // ...
  T *p_device_b_c_;             // ...

  T *p_device_tmp5_;            // lstm_size_ x minibatch_size_, this = p_hidden_to_hidden_layer_->p_device_m_c_ * p_device_h_t_below_, 
                                // for p_device_c_prime_t_tanh_
  T *p_device_tmp6_;            // lstm_size_ x minibatch_size_, this = p_hidden_to_hidden_layer_->p_device_w_hc_ * p_device_h_t_prev_, 
                                // for p_device_c_prime_t_tanh_
  T *p_device_tmp7_;            // lstm_size_ x minibatch_size_, this = p_hidden_to_hidden_layer_->p_device_m_o_ * p_device_h_t_below_, for p_device_o_t_
  T *p_device_tmp8_;            // lstm_size_ x minibatch_size_, this = p_hidden_to_hidden_layer_->p_device_w_ho_ * p_device_h_t_prev_, for p_device_o_t_

public:
  T *p_host_h_t_below_;
  T *p_device_h_t_below_;

public:
  // new for saving space in the LSTM
  T *p_host_d_errn_to_t_ht_;
  T *p_host_d_errt_ct_;
  T *p_host_d_errn_to_t_ct_;
  T *p_host_d_errn_to_t_ot_;
  T *p_host_d_errn_to_t_ft_;
  T *p_host_d_errn_to_t_tanhcpt_;
  T *p_host_d_errn_to_t_it_;
  T *p_host_d_errn_to_t_htm1_;
  T *p_host_d_errn_to_t_ctm1_;
  T *p_host_d_errn_to_t_h_below_;

  T *p_device_d_errn_to_t_ht_;                  // lstm_size_ x minibatch_size_
  T *p_device_d_errt_ct_;                       // lstm_size_ x minibatch_size_
  T *p_device_d_errn_to_t_ct_;                  // lstm_size_ x minibatch_size_
  T *p_device_d_errn_to_t_ot_;                  // lstm_size_ x minibatch_size_
  T *p_device_d_errn_to_t_ft_;
  T *p_device_d_errn_to_t_tanhcpt_;             // lstm_size_ x minibatch_size_
  T *p_device_d_errn_to_t_it_;                  // lstm_size_ x minibatch_size_
  T *p_device_d_errn_to_t_htm1_;
  T *p_device_d_errn_to_t_ctm1_;
  T *p_device_d_errn_to_t_h_below_;

public:
  // thrust device pointers to doing parameter updates nicely (not input word embeddings though)
  thrust::device_ptr<T> p_thrust_device_w_ho_grad_;
  thrust::device_ptr<T> p_thrust_device_w_hf_grad_;
  thrust::device_ptr<T> p_thrust_device_w_hi_grad_;
  thrust::device_ptr<T> p_thrust_device_w_hc_grad_;

  thrust::device_ptr<T> p_thrust_device_m_i_grad_;
  thrust::device_ptr<T> p_thrust_device_m_f_grad_;
  thrust::device_ptr<T> p_thrust_device_m_o_grad_;
  thrust::device_ptr<T> p_thrust_device_m_c_grad_;

  thrust::device_ptr<T> p_thrust_device_b_i_grad_;
  thrust::device_ptr<T> p_thrust_device_b_f_grad_;
  thrust::device_ptr<T> p_thrust_device_b_o_grad_;
  thrust::device_ptr<T> p_thrust_device_b_c_grad_;

public:
  boost::random::mt19937 generator_;

public:
  NeuralMachineTranslation<T> *p_neural_mt_;

public:
  bool debug_mode_;                // true if want debugging printout, false otherwise
  int minibatch_size_;
  T learning_rate_;
  bool clip_gradient_mode_;
  T norm_clip_;                     // for gradient clipping
  int lstm_size_;
  int longest_sentence_;
  AttentionLayer<T> *p_attention_layer_ = NULL;

public:
  bool bi_dir_mode_ = false;        // flag for bidirectional encoder madness
  int layer_number_ = -1;           // start at 1, for indexing directly into the target layer
  

public:
  bool dropout_mode_;               // for dropout
  T dropout_rate_;
  curandGenerator_t rand_generator_;
  
public:
  UpperTransferLayer<T> upper_layer_;


public:
  HiddenToHiddenLayer() {};   // constructor

public:
  void InitHiddenToHiddenLayer(int lstm_size, int minibatch_size, int longest_sentence, bool debug_mode, \
                               T learning_rate, bool clip_gradient_mode, T norm_clip, \
                               NeuralMachineTranslation<precision> *p_neural_mt, \
                               int seed, bool dropout_mode, T dropout_rate, bool bi_dir_mode, int layer_number);

  void InitHiddenToHiddenLayerGpu(int lstm_size, int minibatch_size, int longest_sentence, bool debug_mode, \
                                  T learning_rate, bool clip_gradient_mode, T norm_clip, \
                                  NeuralMachineTranslation<precision> *p_neural_mt, int seed);

public:
  // clear the previous gradients
  void ClearGradients(bool init);
  void ClearGradientsGpu(bool init);

public:
  // update the weights of the model
  void UpdateWeights();
  void UpdateWeightsGpu();

public:
  void CalculateGlobalNorm();

public:
  void UpdateGlobalParams();

public:
  void CheckAllGradients(T epsilon);
  void CheckAllGradientsGpu(T epsilon);

public:
  void DumpWeights(std::ofstream &output);
  void DumpWeightsGpu(std::ofstream &output);

public:
  void LoadWeights(std::ifstream &input_stream);
  void LoadWeightsGpu(std::ifstream &input_stream);

public:
  void CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols);

public:
  // convert to 0/1's and to indices where there are no -1's
  void PreprocessGpuVocabIndices(int *p_host_input_vocab_indices, int current_length);


public:
  template <typename Derived>
  void SwapStatesDecoding(const Eigen::MatrixBase<Derived> &eigen_indices, int index, T *p_device_tmp_swap_vals);

public:
  void TransferDecodingStatesGpu(T *p_device_h_t, T *p_device_c_t);

public:
  void InitAttention(int device_number, int d, bool feed_input_mode, NeuralMachineTranslation<T> *p_neural_mt, GlobalConfiguration &config);
  
public:
  void ZeroAttentionError();

public:
  void ScaleGradients();
  
public:
  void UpdateParams();
};



template <typename T>
void HiddenToHiddenLayer<T>::InitHiddenToHiddenLayer(int lstm_size, int minibatch_size, int longest_sentence, bool debug_mode, \
                                                     T learning_rate, bool clip_gradient_mode, T norm_clip, \
                                                     NeuralMachineTranslation<precision> *p_neural_mt, \
                                                     int seed, bool dropout_mode, T dropout_rate, bool bi_dir_mode, int layer_number) {

  debug_mode_ = debug_mode;
  minibatch_size_ = minibatch_size;
  learning_rate_ = learning_rate;
  clip_gradient_mode_ = clip_gradient_mode;
  norm_clip_ = norm_clip;
  p_neural_mt_ = p_neural_mt;
  
  lstm_size_ = lstm_size;
  longest_sentence_ = longest_sentence;
  dropout_mode_ = dropout_mode;
  dropout_rate_ = dropout_rate;

  bi_dir_mode_ = bi_dir_mode;
  layer_number_ = layer_number;

  generator_.seed(seed);

  InitHiddenToHiddenLayerGpu(lstm_size, minibatch_size, longest_sentence, debug_mode, learning_rate, clip_gradient_mode, norm_clip, p_neural_mt, seed);

  // Initialize the vector of LSTM nodes to longest sentence
  v_nodes_.clear();
  for (int i = 0; i < longest_sentence; ++i) {
    LstmHiddenHiddenNode<T> lstm_hidden_hidden_node;
    lstm_hidden_hidden_node.Init(lstm_size, minibatch_size, this, i, p_device_zeros_, dropout_mode, dropout_rate);
    v_nodes_.push_back(lstm_hidden_hidden_node);
  }
}



template <typename T>
void HiddenToHiddenLayer<T>::InitHiddenToHiddenLayerGpu(int lstm_size, int minibatch_size, int longest_sentence, bool debug_mode, \
                                                        T learning_rate, bool clip_gradient_mode, T norm_clip, \
                                                        NeuralMachineTranslation<precision> *p_neural_mt, int seed) {

  cudaSetDevice(hidden_hidden_layer_information_.device_number_);

  FullMatrixSetup(&p_host_w_ho_, &p_device_w_ho_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hf_, &p_device_w_hf_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hi_, &p_device_w_hi_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hc_, &p_device_w_hc_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_w_ho_grad_, &p_device_w_ho_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hf_grad_, &p_device_w_hf_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hi_grad_, &p_device_w_hi_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hc_grad_, &p_device_w_hc_grad_, lstm_size, lstm_size);


  FullMatrixSetup(&p_host_m_i_, &p_device_m_i_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_f_, &p_device_m_f_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_o_, &p_device_m_o_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_c_, &p_device_m_c_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_m_i_grad_, &p_device_m_i_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_f_grad_, &p_device_m_f_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_o_grad_, &p_device_m_o_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_c_grad_, &p_device_m_c_grad_, lstm_size, lstm_size);


  FullMatrixSetup(&p_host_b_i_, &p_device_b_i_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_f_, &p_device_b_f_, lstm_size, 1);

  thrust::device_ptr<T> bias_ptr = thrust::device_pointer_cast(p_device_b_f_);
  for (int i = 0; i < lstm_size; ++i) {
    bias_ptr[i] = 1;
  }

  FullMatrixSetup(&p_host_b_c_, &p_device_b_c_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_o_, &p_device_b_o_, lstm_size, 1);

  FullMatrixSetup(&p_host_b_i_grad_, &p_device_b_i_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_f_grad_, &p_device_b_f_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_c_grad_, &p_device_b_c_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_o_grad_, &p_device_b_o_grad_, lstm_size, 1);


  FullMatrixSetupZeros(&p_host_init_hidden_vector_, &p_device_init_hidden_vector_, lstm_size, minibatch_size);
  FullMatrixSetupZeros(&p_host_init_cell_vector_, &p_device_init_cell_vector_, lstm_size, minibatch_size);
  FullMatrixSetupZeros(&p_host_init_d_errn_to_tp1_ht_, &p_device_init_d_errn_to_tp1_ht_, lstm_size, minibatch_size);
  FullMatrixSetupZeros(&p_host_init_d_errn_to_tp1_ct_, &p_device_init_d_errn_to_tp1_ct_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_tmp1_, &p_device_tmp1_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp2_, &p_device_tmp2_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp3_, &p_device_tmp3_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp4_, &p_device_tmp4_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp5_, &p_device_tmp5_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp6_, &p_device_tmp6_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp7_, &p_device_tmp7_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp8_, &p_device_tmp8_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_h_t_below_, &p_device_h_t_below_, lstm_size, minibatch_size);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_, &p_device_input_vocab_indices_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_01_full_, &p_device_input_vocab_indices_01_full_, minibatch_size, longest_sentence);

  // Set to zeros
  CudaErrorWrapper(cudaMalloc((void **)&p_device_zeros_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed zeros\n");
  cudaMemset(p_device_zeros_, 0, lstm_size * minibatch_size * sizeof(T));

  // Set to ones
  FullVectorSetupOnes(&p_host_ones_minibatch_, &p_device_ones_minibatch_, minibatch_size);

  // Get device pointers
  p_thrust_device_w_ho_grad_ = thrust::device_pointer_cast(p_device_w_ho_grad_);
  p_thrust_device_w_hf_grad_ = thrust::device_pointer_cast(p_device_w_hf_grad_);
  p_thrust_device_w_hi_grad_ = thrust::device_pointer_cast(p_device_w_hi_grad_);
  p_thrust_device_w_hc_grad_ = thrust::device_pointer_cast(p_device_w_hc_grad_);

  p_thrust_device_m_i_grad_ = thrust::device_pointer_cast(p_device_m_i_grad_);
  p_thrust_device_m_f_grad_ = thrust::device_pointer_cast(p_device_m_f_grad_);
  p_thrust_device_m_o_grad_ = thrust::device_pointer_cast(p_device_m_o_grad_);
  p_thrust_device_m_c_grad_ = thrust::device_pointer_cast(p_device_m_c_grad_);

  p_thrust_device_b_i_grad_ = thrust::device_pointer_cast(p_device_b_i_grad_);
  p_thrust_device_b_f_grad_ = thrust::device_pointer_cast(p_device_b_f_grad_);
  p_thrust_device_b_o_grad_ = thrust::device_pointer_cast(p_device_b_o_grad_);
  p_thrust_device_b_c_grad_ = thrust::device_pointer_cast(p_device_b_c_grad_);

  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_, 1 * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_tmp_, NORM_THREADS * sizeof(T)), "GPU memory allocation failed\n");

  // Save space in the LSTM
  FullMatrixSetup(&p_host_d_errn_to_t_ht_, &p_device_d_errn_to_t_ht_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errt_ct_, &p_device_d_errt_ct_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ct_, &p_device_d_errn_to_t_ct_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ot_, &p_device_d_errn_to_t_ot_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ft_, &p_device_d_errn_to_t_ft_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_tanhcpt_, &p_device_d_errn_to_t_tanhcpt_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_it_, &p_device_d_errn_to_t_it_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_d_errn_to_t_htm1_, &p_device_d_errn_to_t_htm1_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ctm1_, &p_device_d_errn_to_t_ctm1_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_h_below_, &p_device_d_errn_to_t_h_below_, lstm_size, minibatch_size);

  curandCreateGenerator(&rand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
  //boost::uniform_int<> unif_boost(1, 1000000);
  //curandSetPseudoRandomGeneratorSeed(rand_generator_, unif_boost(generator__));
  curandSetPseudoRandomGeneratorSeed(rand_generator_, curr_seed__);
  curr_seed__ += 7;

  ClearGradients(true);

  cudaSetDevice(hidden_hidden_layer_information_.device_number_);
  cudaDeviceSynchronize();
  //cudaSetDevice(0);
}


template <typename T>
void HiddenToHiddenLayer<T>::InitAttention(int device_number, int d, bool feed_input_mode, NeuralMachineTranslation<T> *p_neural_mt, GlobalConfiguration &config) {

  p_attention_layer_ = new AttentionLayer<T>(lstm_size_, minibatch_size_, hidden_hidden_layer_information_.device_number_, d, longest_sentence_, hidden_hidden_layer_information_.handle_, p_neural_mt, feed_input_mode, clip_gradient_mode_, norm_clip_, dropout_mode_, dropout_rate_, config, false);

  // multi_attention is not written

  // Now switch on the attention flag in the attention nodes
  for (int i = 0; i < v_nodes_.size(); ++i) {
    v_nodes_[i].attention_model_mode_ = true;
    // multi_attention is not written
  }
}


template <typename T>
void HiddenToHiddenLayer<T>::ZeroAttentionError() {
  cudaSetDevice(hidden_hidden_layer_information_.device_number_);
  for (int i = 0; i < v_nodes_.size(); ++i) {
    cudaMemset(v_nodes_[i].p_device_d_errt_ht_, 0, lstm_size_ * minibatch_size_ * sizeof(T));
  }
}


// CHECK DECODER: OK //
// CHECK: OK //
template <typename T>
void HiddenToHiddenLayer<T>::ClearGradients(bool init) {
  ClearGradientsGpu(init);
}



// CHECK: OK //
template <typename T>
void HiddenToHiddenLayer<T>::ClearGradientsGpu(bool init) {
  cudaSetDevice(hidden_hidden_layer_information_.device_number_);
  cudaDeviceSynchronize();
  
  cudaMemsetAsync(p_device_w_hi_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s00_);
  cudaMemsetAsync(p_device_b_i_grad_, 0, lstm_size_ * 1 * sizeof(T), hidden_hidden_layer_information_.s01_);

  cudaMemsetAsync(p_device_w_hf_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s02_);
  cudaMemsetAsync(p_device_b_f_grad_, 0, lstm_size_ * 1 * sizeof(T), hidden_hidden_layer_information_.s03_);

  cudaMemsetAsync(p_device_w_hc_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s04_);
  cudaMemsetAsync(p_device_b_c_grad_, 0, lstm_size_ * 1 * sizeof(T), hidden_hidden_layer_information_.s05_);

  cudaMemsetAsync(p_device_w_ho_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s06_);
  cudaMemsetAsync(p_device_b_o_grad_, 0, lstm_size_ * 1 * sizeof(T), hidden_hidden_layer_information_.s07_);

  cudaMemsetAsync(p_device_m_i_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s09_);
  cudaMemsetAsync(p_device_m_f_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s10_);
  cudaMemsetAsync(p_device_m_o_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s11_);
  cudaMemsetAsync(p_device_m_c_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), hidden_hidden_layer_information_.s12_);

  if (p_attention_layer_ != NULL) {
    p_attention_layer_->ClearGradients();
    // multi_source_attention_mode is not written
  }

  DeviceSyncAll();
}



template <typename T>
void HiddenToHiddenLayer<T>::PreprocessGpuVocabIndices(int *p_host_input_vocab_indices, int current_length) {

  cudaSetDevice(hidden_hidden_layer_information_.device_number_);

  p_host_input_vocab_indices_ = p_host_input_vocab_indices;
  current_length_ = current_length;

  // transfer to the GPU
  cudaMemcpy(p_device_input_vocab_indices_, p_host_input_vocab_indices, minibatch_size_ * current_length * sizeof(int), cudaMemcpyHostToDevice);
  CudaGetLastError("HiddenToHiddenLayer::p_device_input_vocab_indices_ preprocess");

  // launch kernel to turn into 0/1's and indices with no -1's
  int threads_per_block = 128;
  int blocks_per_grid = 128;
  VocabTo01Kernel<<<blocks_per_grid, threads_per_block>>>(p_device_input_vocab_indices_01_full_, p_device_input_vocab_indices_, current_length * minibatch_size_);
  CudaGetLastError("HiddenToHiddenLayer::p_device_input_vocab_indices_01_full_ preprocess");

  if (NULL != p_attention_layer_) {
    p_attention_layer_->transfer_done_ = false;

    // multi_source_attention is not written
  }
}



template <typename T>
template <typename Derived>
void HiddenToHiddenLayer<T>::SwapStatesDecoding(const Eigen::MatrixBase<Derived> &eigen_indices, int index, T *p_device_tmp_swap_vals) {

  index = 0;

  for (int i = 0; i < eigen_indices.rows(); ++i) {
    cudaMemcpy(p_device_tmp_swap_vals + i * lstm_size_, v_nodes_[index].p_device_h_t_ + eigen_indices(i) * lstm_size_, lstm_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
  }

  cudaMemcpy(v_nodes_[index].p_device_h_t_, p_device_tmp_swap_vals, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);

  for (int i = 0; i < eigen_indices.rows(); ++i) {
    cudaMemcpy(p_device_tmp_swap_vals + i * lstm_size_, v_nodes_[index].p_device_c_t_ + eigen_indices(i) * lstm_size_, lstm_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
  }

  cudaMemcpy(v_nodes_[index].p_device_c_t_, p_device_tmp_swap_vals, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
}




template <typename T>
void HiddenToHiddenLayer<T>::TransferDecodingStatesGpu(T *p_device_h_t, T *p_device_c_t) {

  for (int i = 0; i < minibatch_size_; ++i) {
    int step = i * lstm_size_;
    CudaErrorWrapper(cudaMemcpy(p_device_init_hidden_vector_ + step, p_device_h_t, lstm_size_ * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "transfer decoding states h_t memcpy failed\n");
    CudaErrorWrapper(cudaMemcpy(p_device_init_cell_vector_ + step, p_device_c_t, lstm_size_ * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "transfer decoding states c_t memcpy failed\n");
  }

  v_nodes_[0].p_device_h_t_prev_ = p_device_init_hidden_vector_;
  v_nodes_[0].p_device_c_t_prev_ = p_device_init_cell_vector_;
}


template <typename T>
void HiddenToHiddenLayer<T>::CheckAllGradients(T epsilon) {
  CheckAllGradientsGpu(epsilon);
  return;
}


template <typename T>
void HiddenToHiddenLayer<T>::CheckAllGradientsGpu(T epsilon) {
  cudaSetDevice(hidden_hidden_layer_information_.device_number_);

  std::cerr<<" #### GRADIENT CHECKING FOR HIDDEN LAYER GPU ####\n"
           <<" GRADIENT CHECKING FOR p_device_w_hi_\n";
  CheckGradientGpu(epsilon, p_device_w_hi_, p_device_w_hi_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_w_hf_\n";
  CheckGradientGpu(epsilon, p_device_w_hf_, p_device_w_hf_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_w_ho_\n";
  CheckGradientGpu(epsilon, p_device_w_ho_, p_device_w_ho_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_w_hc_\n";
  CheckGradientGpu(epsilon, p_device_w_hc_, p_device_w_hc_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_b_i_\n";
  CheckGradientGpu(epsilon, p_device_b_i_, p_device_b_i_grad_, lstm_size_, 1);

  std::cerr<<" GRADIENT CHECKING FOR p_device_b_f_\n";
  CheckGradientGpu(epsilon, p_device_b_f_, p_device_b_f_grad_, lstm_size_, 1);

  std::cerr<<" GRADIENT CHECKING FOR p_device_b_c_\n";
  CheckGradientGpu(epsilon, p_device_b_c_, p_device_b_c_grad_, lstm_size_, 1);

  std::cerr<<" GRADIENT CHECKING FOR p_device_b_o_\n";
  CheckGradientGpu(epsilon, p_device_b_o_, p_device_b_o_grad_, lstm_size_, 1);

  std::cerr<<" GRADIENT CHECKING FOR p_device_m_i_\n";
  CheckGradientGpu(epsilon, p_device_m_i_, p_device_m_i_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_m_f_\n";
  CheckGradientGpu(epsilon, p_device_m_f_, p_device_m_f_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_m_o_\n";
  CheckGradientGpu(epsilon, p_device_m_o_, p_device_m_o_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_m_c_\n";
  CheckGradientGpu(epsilon, p_device_m_c_, p_device_m_c_grad_, lstm_size_, lstm_size_);

  if (NULL != p_attention_layer_) {
    p_attention_layer_->CheckGradients(epsilon);
  }

  return;
}

template <typename T>
void HiddenToHiddenLayer<T>::CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols) {
  cudaSetDevice(hidden_hidden_layer_information_.device_number_);
  thrust::device_ptr<T> p_thrust_device_mat = thrust::device_pointer_cast(p_device_mat);
  thrust::device_ptr<T> p_thrust_device_grad = thrust::device_pointer_cast(p_device_grad);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      T loss = 0;
      p_thrust_device_mat[IDX2C(i, j, rows)] += epsilon;
      loss = p_neural_mt_->GetError(true);
      cudaSetDevice(hidden_hidden_layer_information_.device_number_);
      cudaDeviceSynchronize();

      p_thrust_device_mat[IDX2C(i, j, rows)] += -2 * epsilon;
      loss -= p_neural_mt_->GetError(true);
      cudaSetDevice(hidden_hidden_layer_information_.device_number_);
      cudaDeviceSynchronize();

      p_thrust_device_mat[IDX2C(i, j, rows)] += epsilon;
      std::cerr<<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"  my gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n";
      if((std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))) > 1 / (T)1000.0 || (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))) > 1 / 1000.0) {
        std::cerr<<"Gradient for gradient check: "<<loss / (2 * epsilon)<<"\n"
                 <<"My gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n"
                 <<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"\n"
                 <<"Gradient difference (Equation 2): "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))<<"\n\n";
      } else if (0 == p_thrust_device_grad[IDX2C(i, j, rows)] || 0 == loss / (2 * epsilon)) {
        std::cerr<<"ZERO GRADIENTS\n"
                 <<"Gradient for gradient check: "<<loss / (2 * epsilon)<<"\n"
                 <<"My gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n"
                 <<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"\n"
                 <<"Gradient difference (Equation 2): "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))<<"\n\n";
      }
    }
  }
  return;
}


template <typename T>
void HiddenToHiddenLayer<T>::CalculateGlobalNorm() {
  cudaSetDevice(hidden_hidden_layer_information_.device_number_);
  ScaleGradients();

  NormClipGpuV2P1(p_thrust_device_w_hi_grad_, p_device_w_hi_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_w_hf_grad_, p_device_w_hf_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_w_hc_grad_, p_device_w_hc_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_w_ho_grad_, p_device_w_ho_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  NormClipGpuV2P1(p_thrust_device_b_i_grad_, p_device_b_i_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_b_f_grad_, p_device_b_f_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_b_c_grad_, p_device_b_c_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_b_o_grad_, p_device_b_o_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);

  NormClipGpuV2P1(p_thrust_device_m_i_grad_, p_device_m_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_m_f_grad_, p_device_m_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_m_o_grad_, p_device_m_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_m_c_grad_, p_device_m_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  if (NULL != p_attention_layer_) {
    p_attention_layer_->NormP1();

    // multi_source_attention is not written
  }

  DeviceSyncAll();
}


template <typename T>
void HiddenToHiddenLayer<T>::UpdateGlobalParams() {

  cudaSetDevice(hidden_hidden_layer_information_.device_number_);

  NormClipGpuV2P2(p_thrust_device_w_hi_grad_, p_device_w_hi_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_hf_grad_, p_device_w_hf_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_hc_grad_, p_device_w_hc_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_ho_grad_, p_device_w_ho_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  NormClipGpuV2P2(p_thrust_device_b_i_grad_, p_device_b_i_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_b_f_grad_, p_device_b_f_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_b_c_grad_, p_device_b_c_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_b_o_grad_, p_device_b_o_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);

  NormClipGpuV2P2(p_thrust_device_m_i_grad_, p_device_m_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_m_f_grad_, p_device_m_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_m_o_grad_, p_device_m_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_m_c_grad_, p_device_m_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  if (NULL != p_attention_layer_) {
    p_attention_layer_->NormP2();
    // multi_source_attention is not written
  }

  UpdateParams();
  DeviceSyncAll();
}


template <typename T>
void HiddenToHiddenLayer<T>::UpdateParams() {

  T alpha = learning_rate_;
  T beta = 1;

  DeviceSyncAll();

  // normal matrices
  if ((source_side_mode__ && train_source_rnn_mode__) || (!source_side_mode__ && train_target_rnn_mode__)) {
    // stream 00
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s00_);
    // p_device_w_hi_ += learning_rate_ * p_device_w_hi_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_hi_grad_, lstm_size_, &beta, p_device_w_hi_, lstm_size_, p_device_w_hi_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_w_hi_ failed\n");

    // stream 02
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s02_);
    // p_device_w_hf_ += learning_rate_ * p_device_w_hf_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_hf_grad_, lstm_size_, &beta, p_device_w_hf_, lstm_size_, p_device_w_hf_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_w_hf_ failed\n");

    // stream 04
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s04_);
    // p_device_w_hc_ += learning_rate_ * p_device_w_hc_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_hc_grad_, lstm_size_, &beta, p_device_w_hc_, lstm_size_, p_device_w_hc_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_w_hc_ failed\n");

    // stream 06
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s06_);
    // p_device_w_ho_ += learning_rate_ * p_device_w_ho_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_ho_grad_, lstm_size_, &beta, p_device_w_ho_, lstm_size_, p_device_w_ho_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_w_ho_ failed\n");

    // stream 09
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s09_);
    // p_device_m_i_ += learning_rate_ * p_device_m_i_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_i_grad_, lstm_size_, &beta, p_device_m_i_, lstm_size_, p_device_m_i_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_m_i_ failed\n");

    // stream 10
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s10_);
    // p_device_m_f_ += learning_rate_ * p_device_m_f_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_f_grad_, lstm_size_, &beta, p_device_m_f_, lstm_size_, p_device_m_f_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_m_f_ failed\n");

    // stream 11
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s11_);
    // p_device_m_o_ += learning_rate_ * p_device_m_o_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_o_grad_, lstm_size_, &beta, p_device_m_o_, lstm_size_, p_device_m_o_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_m_o_ failed\n");

    // stream 12
    cublasSetStream(hidden_hidden_layer_information_.handle_, hidden_hidden_layer_information_.s12_);
    // p_device_m_c_ += learning_rate_ * p_device_m_c_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_c_grad_, lstm_size_, &beta, p_device_m_c_, lstm_size_, p_device_m_c_, lstm_size_), "HiddenToHiddenLayer::UpdateParams p_device_m_c_ failed\n");

    // p_device_b_i_ += learning_rate_ * p_device_b_i_grad_
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, hidden_hidden_layer_information_.s01_>>>(p_device_b_i_, p_device_b_i_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();

    // p_device_b_f_ += learning_rate_ * p_device_b_f_grad_
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, hidden_hidden_layer_information_.s03_>>>(p_device_b_f_, p_device_b_f_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();

    // p_device_b_c_ += learning_rate_ * p_device_b_c_grad_
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, hidden_hidden_layer_information_.s05_>>>(p_device_b_c_, p_device_b_c_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();

    // p_device_b_o_ += learning_rate_ * p_device_b_o_grad_
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, hidden_hidden_layer_information_.s07_>>>(p_device_b_o_, p_device_b_o_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();
  }

  if (train_attention_target_rnn_mode__) {
    if (NULL != p_attention_layer_) {
      p_attention_layer_->UpdateParams();
      // multi_source_attention is not written
    }
  }
  DeviceSyncAll();
}


template <typename T>
void HiddenToHiddenLayer<T>::ScaleGradients() {

  cudaSetDevice(hidden_hidden_layer_information_.device_number_);
  
  ScaleFunctor unary_op(minibatch_size_);

  thrust::for_each(p_thrust_device_w_hi_grad_, p_thrust_device_w_hi_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_i_grad_, p_thrust_device_b_i_grad_ + lstm_size_ * 1, unary_op);

  thrust::for_each(p_thrust_device_w_hf_grad_, p_thrust_device_w_hf_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_f_grad_, p_thrust_device_b_f_grad_ + lstm_size_ * 1, unary_op);

  thrust::for_each(p_thrust_device_w_hc_grad_, p_thrust_device_w_hc_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_c_grad_, p_thrust_device_b_c_grad_ + lstm_size_ * 1, unary_op);

  thrust::for_each(p_thrust_device_w_ho_grad_, p_thrust_device_w_ho_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_o_grad_, p_thrust_device_b_o_grad_ + lstm_size_ * 1, unary_op);

  thrust::for_each(p_thrust_device_m_i_grad_, p_thrust_device_m_i_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_m_f_grad_, p_thrust_device_m_f_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_m_o_grad_, p_thrust_device_m_o_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_m_c_grad_, p_thrust_device_m_c_grad_ + lstm_size_ * lstm_size_, unary_op);

  if (NULL != p_attention_layer_) {
    p_attention_layer_->ScaleGradients();

    // multi_source_attention is not written
  }

  DeviceSyncAll();
}


template <typename T>
void HiddenToHiddenLayer<T>::UpdateWeights() {
  UpdateWeightsGpu();
}


template <typename T>
void HiddenToHiddenLayer<T>::UpdateWeightsGpu() {

  cudaSetDevice(hidden_hidden_layer_information_.device_number_);

  ScaleGradients();

#ifdef DEBUG_DROPOUT_4
  std::cerr<<"   HiddenToHiddenLayer::individual_grad_clip_mode__: "<<individual_grad_clip_mode__<<"\n"<<std::flush;
#endif

  if (individual_grad_clip_mode__) {
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_w_hi_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_w_hf_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_w_hc_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_w_ho_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);

    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_b_i_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_b_f_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_b_c_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_b_o_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);

    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_m_i_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_m_f_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_m_o_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, hidden_hidden_layer_information_.s00_>>>(p_device_m_c_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);

    if (NULL != p_attention_layer_) {
      p_attention_layer_->ClipIndividual();
      // multi_source_attention is not written
    }

    DeviceSyncAll();
  }

#ifdef DEBUG_DROPOUT_4
  std::cerr<<"   HiddenToHiddenLayer::clip_gradient_mode_: "<<clip_gradient_mode_<<"\n"<<std::flush;
#endif


  if (clip_gradient_mode_) {
    NormClipGpuV2(p_thrust_device_w_hi_grad_, p_device_w_hi_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_w_hf_grad_, p_device_w_hf_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_w_hc_grad_, p_device_w_hc_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_w_ho_grad_, p_device_w_ho_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

    NormClipGpuV2(p_thrust_device_b_i_grad_, p_device_b_i_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_b_f_grad_, p_device_b_f_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_b_c_grad_, p_device_b_c_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_b_o_grad_, p_device_b_o_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);

    NormClipGpuV2(p_thrust_device_m_i_grad_, p_device_m_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_m_f_grad_, p_device_m_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_m_o_grad_, p_device_m_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_m_c_grad_, p_device_m_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

    if (NULL != p_attention_layer_) {
      p_attention_layer_->ClipGradientsFunc();
      // multi_source_attention is not written
    }
  }

  UpdateParams();
}


template <typename T>
void HiddenToHiddenLayer<T>::LoadWeights(std::ifstream &input_stream) {
  LoadWeightsGpu(input_stream);
}

template <typename T>
void HiddenToHiddenLayer<T>::LoadWeightsGpu(std::ifstream &input_stream) {
  cudaSetDevice(hidden_hidden_layer_information_.device_number_);
  
  ReadMatrixGpu(p_device_w_hi_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_i_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_w_hf_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_f_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_w_hc_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_c_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_w_ho_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_o_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_m_i_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_f_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_o_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_c_, lstm_size_, lstm_size_, input_stream);

  if (NULL != p_attention_layer_) {
    p_attention_layer_->LoadWeights(input_stream);

    // multi_source_attention_mode is not written
  }
  //cudaSetDevice(0);
}


template <typename T>
void HiddenToHiddenLayer<T>::DumpWeights(std::ofstream &output) {
  DumpWeightsGpu(output);
}


template <typename T>
void HiddenToHiddenLayer<T>::DumpWeightsGpu(std::ofstream &output) {
  cudaSetDevice(hidden_hidden_layer_information_.device_number_);

  WriteMatrixGpu(p_device_w_hi_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_i_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_w_hf_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_f_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_w_hc_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_c_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_w_ho_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_o_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_m_i_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_m_f_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_m_o_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_m_c_, lstm_size_, lstm_size_, output);

  if (NULL != p_attention_layer_) {
    p_attention_layer_->DumpWeights(output);
    // multi_source_attention is not written
  }
}



}

#endif






