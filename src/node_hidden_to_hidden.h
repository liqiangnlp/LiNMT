/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef LSTM_HIDDEN_TO_HIDDEN_H_
#define LSTM_HIDDEN_TO_HIDDEN_H_


#include "deep_rnn.h"
#include "layer_hidden_to_hidden.h"

//#define DEBUG_LSTM_HIDDEN_TO_HIDDEN



namespace neural_machine_translation {


template <typename T>
class NeuralMachineTranslation;


template <typename T>
class HiddenToHiddenLayer;


template <typename T>
class LstmHiddenHiddenNode {

public:
  // GPU parameters
  int minibatch_size_;
  int lstm_size_;
  int index_;                             // what node is this
  bool attention_model_mode_ = false;     // this will only be true for the upper layer on the target side of the LSTM

public:
  bool dropout_mode_;
  T dropout_rate_;
  T *p_device_dropout_mask_;              // lstm_size_ x minibatch_size_

public:
  HiddenToHiddenLayer<T> *p_hidden_to_hidden_layer_;  // Pointer to the model struct, so it can access all of the weight matrices

public:
  // host pointers 
  T *p_host_d_errt_ht_;
  T *p_host_o_t_;
  T *p_host_c_t_;
  int *p_host_input_vocab_indices_01_;
  T *p_host_f_t_;
  T *p_host_c_t_prev_;
  T *p_host_c_prime_t_tanh_;
  T *p_host_i_t_;
  T *p_host_h_t_prev_;
  T *p_host_h_t_;

  // device pointers 
  T *p_device_d_errn_to_tp1_ht_;           // lstm_size_ x minibatch_size_, init by HiddenToHiddenLayer::p_device_init_d_errn_to_tp1_ht_
  T *p_device_d_errn_to_tp1_ct_;           // init by HiddenToHiddenLayer::p_device_init_d_errn_to_tp1_ct_
  T *p_device_d_errt_ht_;                  // lstm_size_ x minibatch_size_
  T *p_device_o_t_;                        // lstm_size_ x minibatch_size_, this[i] = sigmoid(p_hidden_to_hidden_layer_->p_device_tmp7_[i] +
                                           //                                                 p_hidden_to_hidden_layer_->p_device_tmp8_[i] +
                                           //                                                 p_hidden_to_hidden_layer_->p_device_b_o_[i'])
  T *p_device_c_t_;                        // lstm_size_ x minibatch_size_, this[i] = p_device_f_t_[i] * p_device_c_t_prev_[i] + 
                                           //                                         p_device_i_t_[i] * p_device_c_prime_t_tanh_[i]
  int *p_device_input_vocab_indices_01_;   // init with v_hidden_layers_{source,target}_[i].p_device_vocab_indices_01_full_ for v_nodes_[0]
  T *p_device_f_t_;                        // lstm_size_ x minibatch_size_, this[i] = sigmoid(p_hidden_to_hidden_layer_->p_device_tmp3_[i] +
                                           //                                                 p_hidden_to_hidden_layer_->p_device_tmp4_[i] +
                                           //                                                 p_hidden_to_hidden_layer_->p_device_b_f_[i'])
  T *p_device_c_t_prev_;                   // init with v_hidden_layers_{source,target}_[i].p_device_init_cell_vector for v_nodes_[0]
  T *p_device_c_prime_t_tanh_;             // lstm_size_ x minibatch_size_, this[i] = tanh(p_hidden_to_hidden_layer_->p_device_tmp5_[i] +
                                           //                                              p_hidden_to_hidden_layer_->p_device_tmp6_[i] + 
                                           //                                              p_hidden_to_hidden_layer_->p_device_b_c_[i'])
  T *p_device_i_t_;                        // lstm_size_ x minibatch_size_, this[i] = sigmoid(p_hidden_to_hidden_layer_->p_device_tmp1_[i] +
                                           //                                                 p_hidden_to_hidden_layer_->p_device_tmp2_[i] +
                                           //                                                 p_hidden_to_hidden_layer_->p_device_b_i_[i'])
  T *p_device_h_t_prev_;                   // init with v_hidden_layers_{source,target}_[i].p_device_init_hidden_vector for v_nodes_[0]
  T *p_device_h_t_;                        // lstm_size_ x minibatch_size_, this[i] = p_device_o_t_[i] * tanhf(p_device_c_t_[i])

public:
  T *p_host_h_t_below_;
  T *p_device_h_t_below_;                         // lstm_size_ x minibatch_size_, copy p_device_h_t_ from layer below for the same node index

public:
  T *p_device_zeros_;


public:
    LstmHiddenHiddenNode() {}

public:
  void Init(int lstm_size, int minibatch_size, HiddenToHiddenLayer<T> *p_hidden_to_hidden_layer, int index, \
            T *p_device_zero, bool dropout_mode, T dropout_rate);    // Constructor
  
public:
  void InitLstmGpu(int lstm_size, int minibatch_size, HiddenToHiddenLayer<T> *p_hidden_to_hidden_layer);

public:
  void UpdateVectorsForwardGpu(int *p_device_input_vocab_indices_01, T *p_device_h_t_prev, T *p_device_c_t_prev);

public:
  // Compute the forward values for the LSTM node
  // This is after the node has recieved the previous hidden and cell state values
  void ForwardProp();

private:
  void ForwardPropGpu();
  void ForwardPropSync(cudaStream_t &current_stream);

public:
  void BackPropGpu(int index);
  void BackPropPreprocessGpu(T *p_device_d_errn_to_tp1_ht, T *p_device_d_errn_to_tp1_ct);


public:
  void UpdateVectorsForwardDecoder(int *p_device_input_vocab_indices_01);


public:
  void ComputeGradientsGpu();    // update the gradient matrices

private:
    void SendHTAbove();
};



template <typename T>
void LstmHiddenHiddenNode<T>::Init(int lstm_size, int minibatch_size, HiddenToHiddenLayer<T> *p_hidden_to_hidden_layer, int index, T *p_device_zeros, bool dropout_mode, T dropout_rate) {
  
  p_hidden_to_hidden_layer_ = p_hidden_to_hidden_layer;
  dropout_mode_ = dropout_mode;
  dropout_rate_ = dropout_rate;

  InitLstmGpu(lstm_size, minibatch_size, p_hidden_to_hidden_layer);

  minibatch_size_ = minibatch_size;
  lstm_size_ = lstm_size;
  index_ = index;
  p_device_zeros_ = p_device_zeros;
}


template <typename T>
void LstmHiddenHiddenNode<T>::InitLstmGpu(int lstm_size, int minibatch_size, HiddenToHiddenLayer<T> *p_hidden_to_hidden_layer) {

  cudaSetDevice(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.device_number_);

  FullMatrixSetup(&p_host_o_t_, &p_device_o_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_c_t_, &p_device_c_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_f_t_, &p_device_f_t_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_c_prime_t_tanh_, &p_device_c_prime_t_tanh_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_i_t_, &p_device_i_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_h_t_, &p_device_h_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_h_t_below_, &p_device_h_t_below_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errt_ht_, &p_device_d_errt_ht_, lstm_size, minibatch_size);

  cudaMemset(p_device_d_errt_ht_, 0, lstm_size * minibatch_size * sizeof(T));

  if (dropout_mode_) {
    CudaErrorWrapper(cudaMalloc((void **)&p_device_dropout_mask_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  }
}


// CHECK: OK //
template <typename T>
void LstmHiddenHiddenNode<T>::UpdateVectorsForwardGpu(int *p_device_input_vocab_indices_01, T *p_device_h_t_prev, T *p_device_c_t_prev) {
  p_device_h_t_prev_ = p_device_h_t_prev;
  p_device_c_t_prev_ = p_device_c_t_prev;
  p_device_input_vocab_indices_01_ = p_device_input_vocab_indices_01;    
}


// CHECK: OK //
// Forward Prop for Hidden to Hidden Layer
template <typename T>
void LstmHiddenHiddenNode<T>::ForwardProp() {
  ForwardPropGpu();
}


// Forward Prop Gpu for Hidden to Hidden Layer
template <typename T>
void LstmHiddenHiddenNode<T>::ForwardPropGpu() {

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaSetDevice(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.device_number_);

#ifdef DEBUG_DROPOUT
  std::cerr<<"   copy_d_err_ht_mode_: "<<p_hidden_to_hidden_layer_->lower_layer_.copy_d_err_ht_mode_<<"\n"
           <<"   lower_input_mode_: "<<p_hidden_to_hidden_layer_->lower_layer_.lower_input_mode_<<"\n"
           <<std::flush;
#endif
  // dropout
  if (dropout_mode_ && p_hidden_to_hidden_layer_->p_neural_mt_->train_mode_) {
    cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.h_t_below_transfer_, 0);
    if (p_hidden_to_hidden_layer_->lower_layer_.copy_d_err_ht_mode_ && p_hidden_to_hidden_layer_->lower_layer_.lower_input_mode_) {
      cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, \
                          p_hidden_to_hidden_layer_->lower_layer_.p_input_layer_->input_hidden_layer_information_.h_t_below_transfer_, 0);
    } else {
      cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, \
                          p_hidden_to_hidden_layer_->lower_layer_.p_hidden_layer_->hidden_hidden_layer_information_.h_t_below_transfer_, 0);
    }

    if (!p_hidden_to_hidden_layer_->p_neural_mt_->grad_check_flag_) {
      curandSetStream(p_hidden_to_hidden_layer_->rand_generator_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
      CurandGenerateUniformWrapper(p_device_dropout_mask_, lstm_size_ * minibatch_size_, p_hidden_to_hidden_layer_->rand_generator_);
    }
    // dropout for p_device_h_t_below_
    // p_device_h_t_below_[i] = (p_device_dropout_mask_[i] < dropout_rate_) * (1 / dropout_rate_) * p_device_h_t_below_[i]
    DropoutKernel<<<256, 256, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(p_device_dropout_mask_, dropout_rate_, p_device_h_t_below_, lstm_size_ * minibatch_size_);
    cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.dropout_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
  }

  // Operation, using streams 1 and 2
  T alpha = 1;
  T beta = 0;

  int threads_per_block = 128;
  int num_block = (lstm_size_ + threads_per_block - 1)/threads_per_block;
  dim3 kernel(minibatch_size_, num_block, 1);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu CHECKPOINT 0");

  // stream 01
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s01_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s01_);
  
  // p_hidden_to_hidden_layer_->p_device_tmp1_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_m_i_ (lstm_size_ x lstm_size_)
  //                                                                            p_device_h_t_below_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_m_i_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp1_, lstm_size_), "LstmHiddenHiddenNode<T>::ForwardPropGpu (i_t) p_device_tmp1_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.i_t_part1_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s01_);

  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu: CHECKPOINT 1");
#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 02
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_);

  // p_hidden_to_hidden_layer_->p_device_tmp2_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_w_hi_ (lstm_size_ x lstm_size_) *
  //                                                                            p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_w_hi_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp2_, lstm_size_), "LstmHiddenHiddenNode<T>::ForwardPropGpu (i_t) p_device_tmp2_ failed\n");

  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu CHECKPOINT 2");
#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.i_t_part1_, 0);

  // tmp_value = p_hidden_to_hidden_layer_->p_device_tmp1_[i] + 
  //             p_hidden_to_hidden_layer_->p_device_tmp2_[i] + 
  //             p_hidden_to_hidden_layer_->p_device_b_i_[i']
  // p_device_i_t_[i] (lstm_size_ x minibatch_size) = sigmoid(tmp_value)
  ForwardSigmoidKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_>>>(p_device_i_t_, p_hidden_to_hidden_layer_->p_device_tmp1_, p_hidden_to_hidden_layer_->p_device_tmp2_, p_hidden_to_hidden_layer_->p_device_b_i_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu p_device_i_t_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.i_t_full_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_);

  // Operation, using streams 3 and 4
  alpha = 1;
  beta = 0;

  // stream 03
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s03_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s03_);

  // p_hidden_to_hidden_layer_->p_device_tmp3_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_m_f_ (lstm_size_ x lstm_size_) *
  //                                                                            p_device_h_t_below_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_m_f_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp3_, lstm_size_), "LstmHiddenHiddenNode::ForwardPropGpu (f_t) p_device_tmp3_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.f_t_part1_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s03_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 04
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_);

  // p_hidden_to_hidden_layer_->p_device_tmp4_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_w_hf_ (lstm_size_ x lstm_size_) *
  //                                                                            p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_w_hf_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp4_, lstm_size_), "LstmHiddenHiddenNode::ForwardPropGpu (f_t) p_device_tmp4_ failed\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.f_t_part1_, 0);

  // tmp_value = p_hidden_to_hidden_layer_->p_device_tmp3_[i] + 
  //             p_hidden_to_hidden_layer_->p_device_tmp4_[i] + 
  //             p_hidden_to_hidden_layer_->p_device_b_f_[i']
  // p_device_f_t_[i] (lstm_size_ x minibatch_size) = sigmoid(tmp_value)
  ForwardSigmoidKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_>>>(p_device_f_t_, p_hidden_to_hidden_layer_->p_device_tmp3_, p_hidden_to_hidden_layer_->p_device_tmp4_, p_hidden_to_hidden_layer_->p_device_b_f_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu p_device_f_t_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.f_t_full_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // Operation, using streams 5 and 6
  alpha = 1;
  beta = 0;

  // stream 05
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_);

  // p_hidden_to_hidden_layer_->p_device_tmp5_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_m_c_ (lstm_size_ x lstm_size_) *
  //                                                                            p_device_h_t_below_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_m_c_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp5_, lstm_size_), "LstmHiddenHiddenNode::ForwardPropGpu (c_prime_t_tanh) p_device_tmp5_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.c_prime_t_tanh_part1_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 06
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_);

  // p_hidden_to_hidden_layer_->p_device_tmp6_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_w_hc_ (lstm_size_ x lstm_size_) *
  //                                                                            p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_w_hc_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp6_, lstm_size_), "LstmHiddenHiddenNode::ForwardPropGpu (c_prime_t_tanh) p_device_tmp6_ failed\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.c_prime_t_tanh_part1_, 0);

  // tmp_value = p_hidden_to_hidden_layer_->p_device_tmp5_[i] + p_hidden_to_hidden_layer_->p_device_tmp6_[i] + p_hidden_to_hidden_layer_->p_device_b_c_[i']
  // p_device_c_prime_t_tanh_[i] (lstm_size_ x minibatch_size_) = tanh(tmp_value)
  ForwardTanhKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_>>>(p_device_c_prime_t_tanh_, p_hidden_to_hidden_layer_->p_device_tmp5_, p_hidden_to_hidden_layer_->p_device_tmp6_, p_hidden_to_hidden_layer_->p_device_b_c_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu p_device_c_prime_t_tanh_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.c_prime_t_tanh_full_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // Operation, using streams 7 and 8
  alpha = 1;
  beta = 0;

  // stream 07
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_);

  // p_hidden_to_hidden_layer_->p_device_tmp7_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_m_o_ (lstm_size_ x lstm_size_) *
  //                                                                            p_device_h_t_below_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_m_o_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp7_, lstm_size_), "LstmHiddenHiddenNode::ForwardPropGpu (o_t) p_device_tmp7_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.o_t_part1_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 08
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_);
  ForwardPropSync(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_);

  // p_hidden_to_hidden_layer_->p_device_tmp8_ (lstm_size_ x minibatch_size_) = p_hidden_to_hidden_layer_->p_device_w_ho_ (lstm_size_ x lstm_size_) *
  //                                                                            p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_hidden_to_hidden_layer_->p_device_w_ho_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_tmp8_, lstm_size_), "LstmHiddenHiddenNode::ForwardPropGpu (o_t) p_device_tmp8_ failed\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.o_t_part1_, 0);

  // tmp_value = p_hidden_to_hidden_layer_->p_device_tmp7_[i] +
  //             p_hidden_to_hidden_layer_->p_device_tmp8_[i] +
  //             p_hidden_to_hidden_layer_->p_device_b_o_[i']
  // p_device_o_t_[i] = sigmoid(tmp_value)
  ForwardSigmoidKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_>>>(p_device_o_t_, p_hidden_to_hidden_layer_->p_device_tmp7_, p_hidden_to_hidden_layer_->p_device_tmp8_, p_hidden_to_hidden_layer_->p_device_b_o_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu p_device_o_t_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.o_t_full_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation, for now the rest are using the default stream
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.i_t_full_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.f_t_full_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.c_prime_t_tanh_full_, 0);

  // p_device_c_t_[i] (lstm_size_ x minibatch_size_) = p_device_f_t_[i] * p_device_c_t_prev_[i] + p_device_i_t_[i] * p_device_c_prime_t_tanh_[i]
  ForwardCTKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(p_device_c_t_, p_device_f_t_, p_device_c_t_prev_, p_device_i_t_, p_device_c_prime_t_tanh_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu p_device_c_t_");

  if (cell_clip_mode__) {
    // restrict the value of p_device_c_t_ to [-cell_clip_threshold__, cell_clip_threshold__]  
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(p_device_c_t_, cell_clip_threshold__, lstm_size_ * minibatch_size_);
  }

  // Operation
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.o_t_full_, 0);

  // p_device_h_t_[i] (lstm_size_ x minibatch_size_) = p_device_o_t_[i] * tanhf(p_device_c_t_[i])
  ForwardHTKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(p_device_h_t_, p_device_o_t_, p_device_c_t_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu p_device_h_t_");

  // p_device_h_t_[i] = p_device_h_t_[i] * p_device_input_vocab_indices_01_[j]
  // p_device_c_t_[i] = p_device_c_t_[i] * p_device_input_vocab_indices_01_[j]
  ZeroCTAndHT<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(p_device_h_t_, p_device_c_t_, p_device_input_vocab_indices_01_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::ForwardPropGpu ZeroCTAndHT");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // This is for the attention model forward prop testing
  SendHTAbove();

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

}


// Forward Prop Sync for Hidden to Hidden Layer
template <typename T>
void LstmHiddenHiddenNode<T>::ForwardPropSync(cudaStream_t &current_stream) {
  if (p_hidden_to_hidden_layer_->lower_layer_.copy_d_err_ht_mode_) {
    if (p_hidden_to_hidden_layer_->lower_layer_.lower_input_mode_) {
      cudaStreamWaitEvent(current_stream, p_hidden_to_hidden_layer_->lower_layer_.p_input_layer_->input_hidden_layer_information_.h_t_below_transfer_, 0);  
    } else {
      cudaStreamWaitEvent(current_stream, p_hidden_to_hidden_layer_->lower_layer_.p_hidden_layer_->hidden_hidden_layer_information_.h_t_below_transfer_, 0);
    }
  }
  cudaStreamWaitEvent(current_stream, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.h_t_below_transfer_, 0);
  cudaStreamWaitEvent(current_stream, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.dropout_done_, 0);
}



template <typename T>
void LstmHiddenHiddenNode<T>::SendHTAbove() {

  // run forward prop for attention model
  if (attention_model_mode_) {
    cudaEventRecord(p_hidden_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.start_forward_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
    p_hidden_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].ForwardProp();
    cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.forward_prop_done_, 0);

    // multi_attention is not written
  }

  // Send the finished h_t to the above layer
  // the multigpu synchronization structure
  if (p_hidden_to_hidden_layer_->upper_layer_.copy_h_t_mode_) {
    // transfer h_t to the layer above
    if (p_hidden_to_hidden_layer_->upper_layer_.upper_softmax_mode_) {
      // upper layer is softmax layer
      if (!p_hidden_to_hidden_layer_->upper_layer_.source_side_mode_) {
        // target side
        if (!attention_model_mode_) {
          // no attention layer
          cudaMemcpyAsync(p_hidden_to_hidden_layer_->upper_layer_.p_softmax_->GetHTPtr(index_), p_device_h_t_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
        } else {
          // attention
          
          // multi_attention is not written

          cudaMemcpyAsync(p_hidden_to_hidden_layer_->upper_layer_.p_softmax_->GetHTPtr(index_), p_hidden_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
        }
      }

      // bi_dir is not written

    } else {
      // upper layer is hidden layer

      if (!attention_model_mode_) {
        cudaMemcpyAsync(p_hidden_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_, p_device_h_t_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
      } else {
        // never go here, as upper layer is hidden layer, so this layer does not have attention
        // multi_attention is not written
        cudaMemcpyAsync(p_hidden_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_, p_hidden_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
      }
    }
  } else {
    if (p_hidden_to_hidden_layer_->upper_layer_.upper_softmax_mode_) {
      // upper layer is softmax
      if (!p_hidden_to_hidden_layer_->upper_layer_.source_side_mode_) {
        // target side
        if (!attention_model_mode_) {
          // no attention
          p_hidden_to_hidden_layer_->upper_layer_.p_softmax_->SetHTPtr(index_, p_device_h_t_);
        } else {
          // have attention
          p_hidden_to_hidden_layer_->upper_layer_.p_softmax_->SetHTPtr(index_, p_hidden_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_);
        }
      }
    } else {
      // upper layer is hidden layer
      if (!attention_model_mode_) {
        p_hidden_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_ = p_device_h_t_;
      } else {
        // never go here, as upper layer is hidden layer, so this layer does not have attention
        p_hidden_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_ = p_hidden_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_;
      }
    }
  }

  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.h_t_below_transfer_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);

}


template <typename T>
void LstmHiddenHiddenNode<T>::BackPropGpu(int index) {
  
#ifdef DEBUG_DROPOUT_3
  std::cerr<<"   index_: "<<index_<<"\n"<<std::flush;
  std::cerr<<"   index: "<<index<<"\n"<<std::flush;
#endif

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif 

  cudaSetDevice(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.device_number_);

  // back prop node starting
  if (p_hidden_to_hidden_layer_->upper_layer_.upper_softmax_mode_) {
    cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->upper_layer_.p_softmax_->GetErrHTEvent(), 0);
  } else {
    cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->upper_layer_.p_hidden_layer_->hidden_hidden_layer_information_.htm1_done_, 0);
  }

  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_grad_full_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.ctm1_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_hi_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_hf_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_ho_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_hc_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_i_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_f_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_o_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_c_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_i_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_f_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_o_grad_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_c_grad_done_, 0);

  if (attention_model_mode_) {
    // here is attention mode backprop
    // multi_attention is not written

    cudaEventRecord(p_hidden_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.start_backward_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
    p_hidden_to_hidden_layer_->p_attention_layer_->v_nodes_[index].BackProp();
    cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_, p_hidden_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.backward_prop_done_, 0);

    // multi_attention is not written
  }

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  T alpha = 1;
  T beta = 1;

  // using stream 00
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_[i] = p_device_d_errn_to_tp1_ht_[i] (lstm_size_ x minibatch_size_) +
  // (lstm_size_ x minibatch_size_)                           p_device_d_errt_ht_[i] (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGeamWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_device_d_errn_to_tp1_ht_, lstm_size_, &beta, p_device_d_errt_ht_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_ht_\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // nonrev_bi_dir is not written

  // operation
  int threads_per_block = 128;
  int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel(minibatch_size_, num_block, 1);
  // p_hidden_to_hidden_layer_->p_device_d_errt_ct_[index] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_[index] *
  // lstm_size_ x minibatch_size_                            p_device_o_t_[index] *
  //                                                         (1.0f - (tanhf(p_device_c_t[index]))^2)
  DErrtCTKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(p_hidden_to_hidden_layer_->p_device_d_errt_ct_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_, p_device_o_t_, p_device_c_t_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errt_ct_");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 00
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_[i] = p_device_d_errn_to_tp1_ct_[i] +
  // lstm_size_ x minibatch_size_                             p_hidden_to_hidden_layer_->p_device_d_errt_ct_[i]
  CublasErrorWrapper(CublasGeamWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_device_d_errn_to_tp1_ct_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_d_errt_ct_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_ct_\n");

  // nonrev_bi_dir is not written

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // Zero out columns of p_device_d_errn_to_t_ht_ and p_device_errn_to_t_ct_
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_[i][j] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_[i][j] * 
  // lstm_size_ x minibatch_size_                                p_device_input_vocab_indices_01_[j]
  ZeroColumnsKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_, p_device_input_vocab_indices_01_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_ht_ ZeroColumnsKernel");

  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_[i][j] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_[i][j] * 
  // lstm_size_ x minibatch_size_                                p_device_input_vocab_indices_01_[j]
  ZeroColumnsKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_>>>(lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_input_vocab_indices_01_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_ct_ ZeroColumnsKernel");

  // Event for finishing the first stuff
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.backprop_init_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s00_);

  // Starting from this point streams will be used
  // operation
  // stream 01
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s01_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.backprop_init_, 0);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_[index] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_[index] * 
  //                                                              tanhf(p_device_c_t_[index]) * 
  //                                                              p_device_o_t_[index] * (1 - p_device_o_t_[index])
  DErrnToTOTKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s01_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ht_, p_device_o_t_, p_device_c_t_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_ot_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ot_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s01_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 02
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.backprop_init_, 0);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_[index] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_[index] * 
  //                                                              p_device_c_t_prev_[index] * 
  //                                                              p_device_f_t_[index] * (1 - p_device_f_t_[index])
  DErrnToTFTITKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_c_t_prev_, p_device_f_t_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_ft_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ft_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s02_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // using stream 03
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s03_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.backprop_init_, 0);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_[index] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_[index] * 
  //                                                                   p_device_i_t_[index] * 
  //                                                                   (1 - p_device_c_prime_t_tanh_[index] * p_device_c_prime_t_tanh_[index])
  DErrnToTTanhcptKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s03_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_i_t_, p_device_c_prime_t_tanh_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_tanhcpt_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_tanhcpt_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s03_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 04
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.backprop_init_, 0);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_[index] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_[index] * 
  //                                                              p_device_c_prime_t_tanh_[index] * 
  //                                                              p_device_i_t_[index] * (1 - p_device_i_t_[index]);
  DErrnToTFTITKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_c_prime_t_tanh_, p_device_i_t_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_it_");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_it_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s04_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  T alpha2 = 1;
  T beta2 = 0;
  
  // operation
  // using stream 5, 6, 7, 8, 9
  // this is for the error being passed to the lower lstm layer

  // stream 05
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ot_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp1_  = p_hidden_to_hidden_layer_->p_device_m_o_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)               p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_m_o_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp1_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp1_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p1_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 06
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ft_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp2_ = p_hidden_to_hidden_layer_->p_device_m_f_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)              p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_m_f_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp2_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp2_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p2_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 07
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_it_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp3_ = p_hidden_to_hidden_layer_->p_device_m_i_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)              p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_m_i_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp3_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp3_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p3_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 08
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp4_ = p_hidden_to_hidden_layer_->p_device_m_c_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)              p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_m_c_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp4_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp4_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p4_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p1_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p2_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p3_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p4_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_[index] = p_hidden_to_hidden_layer_->p_device_tmp1_[index] + 
  //                                                                   p_hidden_to_hidden_layer_->p_device_tmp2_[index] + 
  //                                                                   p_hidden_to_hidden_layer_->p_device_tmp3_[index] + 
  //                                                                   p_hidden_to_hidden_layer_->p_device_tmp4_[index]
  AddFourMatricesKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_, p_hidden_to_hidden_layer_->p_device_tmp1_, p_hidden_to_hidden_layer_->p_device_tmp2_, p_hidden_to_hidden_layer_->p_device_tmp3_, p_hidden_to_hidden_layer_->p_device_tmp4_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_h_below_");

#ifdef DEBUG_DROPOUT_3
  std::cerr<<"   cell_clip_mode__: "<<cell_clip_mode__<<"\n"<<std::flush;
#endif

  if (cell_clip_mode__) {
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_, error_clip_threshold__, lstm_size_ * minibatch_size_);
  }

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  if (dropout_mode_) {
    DropoutKernel<<<256, 256, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_>>>(p_device_dropout_mask_, dropout_rate_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_, lstm_size_ * minibatch_size_);
  }

#ifdef DEBUG_DROPOUT_3
  std::cerr<<"   copy_d_err_ht_mode_: "<<p_hidden_to_hidden_layer_->lower_layer_.copy_d_err_ht_mode_<<"\n"
           <<"   lower_input_mode_: "<<p_hidden_to_hidden_layer_->lower_layer_.lower_input_mode_<<"\n"<<std::flush;
#endif

  if (p_hidden_to_hidden_layer_->lower_layer_.copy_d_err_ht_mode_) {
    if (p_hidden_to_hidden_layer_->lower_layer_.lower_input_mode_) {
      cudaMemcpyAsync(p_hidden_to_hidden_layer_->lower_layer_.p_input_layer_->v_nodes_[index].p_device_d_errt_ht_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_);
    } else {
      cudaMemcpyAsync(p_hidden_to_hidden_layer_->lower_layer_.p_hidden_layer_->v_nodes_[index].p_device_d_errt_ht_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_);
    }
  } else {
    if (p_hidden_to_hidden_layer_->lower_layer_.lower_input_mode_) {
      p_hidden_to_hidden_layer_->lower_layer_.p_input_layer_->v_nodes_[index].p_device_d_errt_ht_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_;
    } else {
      p_hidden_to_hidden_layer_->lower_layer_.p_hidden_layer_->v_nodes_[index].p_device_d_errt_ht_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_h_below_;
    }
  }
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.d_error_ht_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // using stream 5, 6, 7, 8, 9
  /* p_device_errn_to_t_htm1_.transpose() = (w_ho_.transpose() * ((p_device_errn_to_t_ot_.transpose().array() * o_t_.array() * (1 - o_t_.array())).matrix())) */
  /*                                      + (w_hf_.transpose() * ((p_device_errn_to_t_ft_.transpose().array() * f_t_.array() * (1 - f_t_.array())).matrix())) */
  /*                                      + (w_hi_.transpose() * ((p_device_errn_to_t_it_.transpose().array() * i_t_.array() * (1 - i_t_.array())).matrix())) */
  /*                                      + (w_hc_.transpose() * ((p_device_errn_to_t_tanhcpt_.transpose().array() * (1 - c_prime_t_tanh.array())).matrix())) */
  
  // stream 05
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.d_error_ht_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp1_ = p_hidden_to_hidden_layer_->p_device_w_ho_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)              p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_w_ho_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp1_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp1_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p1_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s05_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 06
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.d_error_ht_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp2_ = p_hidden_to_hidden_layer_->p_device_w_hf_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)              p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_w_hf_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp2_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp2_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p2_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s06_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 07
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.d_error_ht_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp3_ = p_hidden_to_hidden_layer_->p_device_w_hi_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)              p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_w_hi_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp3_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp3_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p3_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s07_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 08
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.d_error_ht_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_tmp4_ = p_hidden_to_hidden_layer_->p_device_w_hc_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)              p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_hidden_to_hidden_layer_->p_device_w_hc_, lstm_size_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, &beta2, p_hidden_to_hidden_layer_->p_device_tmp4_, lstm_size_), "LstmHiddenHiddenNode::BackPropGpu p_device_tmp4_\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p4_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s08_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p1_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p2_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p3_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_p4_done_, 0);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.d_error_ht_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_htm1_[index] = p_hidden_to_hidden_layer_->p_device_tmp1_[index] + 
  //                                                                p_hidden_to_hidden_layer_->p_device_tmp2_[index] + 
  //                                                                p_hidden_to_hidden_layer_->p_device_tmp3_[index] + 
  //                                                                p_hidden_to_hidden_layer_->p_device_tmp4_[index]
  AddFourMatricesKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_htm1_, p_hidden_to_hidden_layer_->p_device_tmp1_, p_hidden_to_hidden_layer_->p_device_tmp2_, p_hidden_to_hidden_layer_->p_device_tmp3_, p_hidden_to_hidden_layer_->p_device_tmp4_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_htm1_");

  if (cell_clip_mode__) {
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_htm1_, error_clip_threshold__, lstm_size_ * minibatch_size_);
  }

  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.htm1_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s09_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 10
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s10_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.backprop_init_, 0);
  // p_device_errn_to_t_ctm1_.transpose() = (p_device_errn_to_t_ct_.transpose().array() * f_t_.array())
  // p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ctm1_[i][j] = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_[i][j] * 
  // (lstm_size_ x minibatch_size_)                                p_device_f_t_[i][j]
  ElementwiseMultKernel<<<kernel, threads_per_block, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s10_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_f_t_, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ctm1_, lstm_size_);
  CudaGetLastError("LstmHiddenHiddenNode::BackPropGpu p_device_d_errn_to_t_ctm1_");

  if (cell_clip_mode__) {
    // bug?????????? should be p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ctm1_
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s10_>>>(p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ct_, error_clip_threshold__, lstm_size_ * minibatch_size_);
  }

  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.ctm1_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s10_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  ComputeGradientsGpu();
  cudaSetDevice(0);
}


template <typename T>
void LstmHiddenHiddenNode<T>::ComputeGradientsGpu() {

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // using streams 11, 12, 13, 14
  // p_hidden_to_hidden_layer_->w_hi_grad_.noalias() += (h_t_prev_ * (p_device_errn_to_t_it_.array() * i_t_.transpose().array() * (1 - i_t_.transpose().array())).matrix()).transpose()
  // p_hidden_to_hidden_layer_->w_hf_grad_.noalias() += (h_t_prev_ * (p_device_errn_to_t_ft_.array() * f_t_.transpose().array() * (1 - f_t_.transpose().array())).matrix()).transpose()
  // p_hidden_to_hidden_layer_->w_hc_grad_.noalias() += (h_t_prev_ * (p_device_errn_to_t_ct_.array() * i_t_.transpose().array() * (1 - c_prime_t_tanh_.transpose().array().square())).matrix()).transpose()
  // p_hidden_to_hidden_layer_->w_ho_grad_.noalias() += (h_t_prev_ * (p_device_errn_to_t_ot_.array() * o_t_.transpose().array() * (1 - o_t_.transpose().array())).matrix()).transpose()

  T alpha = 1;
  T beta = 1;

  // stream 11
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s11_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s11_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_it_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_w_hi_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                        p_device_h_t_prev_^T (minibatch_size_ x lstm_size_) +
  //                                                  p_hidden_to_hidden_layer_->p_device_w_hi_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_w_hi_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_w_hi_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_hi_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s11_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 12
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s12_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s12_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ft_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_w_hf_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                        p_device_h_t_prev_^T (minibatch_size_ x lstm_size_) +
  //                                                  p_hidden_to_hidden_layer_->p_device_w_hf_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_w_hf_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_w_hf_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_hf_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s12_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  // stream 13
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s13_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s13_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_w_hc_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                        p_device_h_t_prev_^T (minibatch_size_ x lstm_size_) +
  //                                                  p_hidden_to_hidden_layer_->p_device_w_hc_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_w_hc_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_w_hc_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_hc_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s13_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 14
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s14_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s14_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ot_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_w_ho_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                        p_device_h_t_prev_^T (minibatch_size_ x lstm_size_) +
  //                                                  p_hidden_to_hidden_layer_->p_device_w_ho_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_w_ho_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_w_ho_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.w_ho_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s14_);


  // operation
  // using stream 15, 16 ,17, 18
  // ComputeTmpMat(p_hidden_to_hidden_layer_->w)
  // p_hidden_to_hidden_layer_->m_i_grad_.noalias() += (p_device_errn_to_t_it_.transpose().array() * i_t_.array() * (1 - i_t_.array())).matrix() * tmp_mat_.transpose()
  // p_hidden_to_hidden_layer_->m_f_grad_.noalias() += (p_device_errn_to_t_ft_.transpose().array() * f_t_.array() * (1 - f_t_.array())).matrix() * tmp_mat_.transpose()
  // p_hidden_to_hidden_layer_->m_o_grad_.noalias() += (p_device_errn_to_t_ot_.transpose().array() * o_t_.array() * (1 - o_t_.array())).matrix() * tmp_mat_.transpose()
  // p_hidden_to_hidden_layer_->m_c_grad_.noalias() += (p_device_errn_to_t_tanhcpt_.transpose().array() * (1 - c_prime_t_tanh_.array().square())).matrix() * tmp_mat_.transpose()

  alpha = 1;
  beta = 1;

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  // stream 15
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s15_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s15_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_it_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_m_i_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_)
  // (lstm_size_ x lstm_size_)                       p_device_h_t_below_^T (minibatch_size_ x lstm_size_) +
  //                                                 p_hidden_to_hidden_layer_->p_device_m_i_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_m_i_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_m_i_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_i_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s15_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 16
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s16_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s16_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ft_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_m_f_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_h_t_below_^T (minibatch_size_ x lstm_size_) +
  //                                                 p_hidden_to_hidden_layer_->p_device_m_f_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_m_f_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_m_f_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_f_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s16_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 17
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s17_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s17_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ot_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_m_o_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_h_t_below_^T (minibatch_size_ x lstm_size_) +
  //                                                 p_hidden_to_hidden_layer_->p_device_m_o_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_m_o_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_m_o_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_o_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s17_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 18
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s18_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s18_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_m_c_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_h_t_below_^T (minibatch_size_ x lstm_size_) +
  //                                                 p_hidden_to_hidden_layer_->p_device_m_c_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, p_device_h_t_below_, lstm_size_, &beta, p_hidden_to_hidden_layer_->p_device_m_c_grad_, lstm_size_), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_m_c_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.m_c_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s18_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // using streams 19, 20, 21, 22
  // b_i_grad_.noalias() += ((p_device_errn_to_t_it_.array() * (i_t_.array() * (1 - i_t_.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose()
  // b_f_grad_.noalias() += ((p_device_errn_to_t_ft_.array() * (f_t_.array() * (1 - f_t_.array())).matrix().transpose().array()).clowise().sum()).matrix().transpose()
  // b_c_grad_.noalias() += ((p_device_errn_to_t_tanhcpt_.array() * (1 - c_prime_t_tanh_.array().square()).matrix().transpose().array()).clowise().sum().matrix().transpose()
  // b_o_grad_.noalias() += ((p_device_errn_to_t_ot_.array() * (o_t_.array() * (1 - o_t_.array())).matrix().transpose().array()).clowise().sum()).matrix().transpose()

  // stream 19
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s19_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s19_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_it_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_b_i_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_hidden_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1) +
  //                                                 p_hidden_to_hidden_layer_->p_device_b_i_grad_
  CublasErrorWrapper(CublasGemvWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, p_hidden_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_hidden_to_hidden_layer_->p_device_b_i_grad_, 1), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_b_i_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_i_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s19_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  // stream 20
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s20_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s20_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ft_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_b_f_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_hidden_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1) +
  //                                                 p_hidden_to_hidden_layer_->p_device_b_f_grad_
  CublasErrorWrapper(CublasGemvWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, p_hidden_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_hidden_to_hidden_layer_->p_device_b_f_grad_, 1), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_b_f_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_f_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s20_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 21
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s21_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s21_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_ot_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_b_o_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_hidden_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1) +
  //                                                 p_hidden_to_hidden_layer_->p_device_b_o_grad_
  CublasErrorWrapper(CublasGemvWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, p_hidden_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_hidden_to_hidden_layer_->p_device_b_o_grad_, 1), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_b_o_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_o_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s21_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 22
  cublasSetStream(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s22_);
  cudaStreamWaitEvent(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s22_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_hidden_to_hidden_layer_->p_device_b_c_grad_ = p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_hidden_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1) +
  //                                                 p_hidden_to_hidden_layer_->p_device_b_c_grad_
  CublasErrorWrapper(CublasGemvWrapper(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_hidden_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, p_hidden_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_hidden_to_hidden_layer_->p_device_b_c_grad_, 1), "LstmHiddenHiddenNode::ComputeGradientsGpu p_device_b_c_grad_ failed\n");
  cudaEventRecord(p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.b_c_grad_done_, p_hidden_to_hidden_layer_->hidden_hidden_layer_information_.s22_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif
}




template <typename T>
void LstmHiddenHiddenNode<T>::BackPropPreprocessGpu(T *p_device_d_errn_to_tp1_ht, T *p_device_d_errn_to_tp1_ct) {
  p_device_d_errn_to_tp1_ht_ = p_device_d_errn_to_tp1_ht;
  p_device_d_errn_to_tp1_ct_ = p_device_d_errn_to_tp1_ct;
}


template <typename T>
void LstmHiddenHiddenNode<T>::UpdateVectorsForwardDecoder(int *p_device_input_vocab_indices_01) {
  // GPU stuff
  p_device_input_vocab_indices_01_ = p_device_input_vocab_indices_01;
}




}






#endif


