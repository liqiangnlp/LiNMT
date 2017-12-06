/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef LSTM_INPUT_TO_HIDDEN_H_
#define LSTM_INPUT_TO_HIDDEN_H_


#include "deep_rnn.h"
#include "deep_rnn_kernel.h"
#include "layer_input_to_hidden.h"

//#define DEBUG_LSTM_INPUT_TO_HIDDEN

namespace neural_machine_translation {

// Forward declaration 
template <typename T>
class NeuralMachineTranslation;


// Forward declaration
template <typename T>
class InputToHiddenLayer;

template <typename T>
class LstmInputHiddenNode {

public:
  InputToHiddenLayer<T> *p_input_to_hidden_layer_;    // Pointer to the model struct, so it can access all of the weight matrices

public:
  int minibatch_size_;
  int lstm_size_;
  int index_;

public:
  bool dropout_mode_;
  T dropout_rate_;
  // device pointer
  T *p_device_dropout_mask_;

public:
  bool attention_model_mode_ = false;    // this will only be true for the upper layer on the target side of the LSTM
  bool feed_input_mode_ = false;
  bool multi_attention_mode_ = false;

public:
  // host pointers
  T *p_host_o_t_;
  T *p_host_c_t_;
  T *p_host_d_errt_ht_;
  T *p_host_f_t_;
  T *p_host_c_t_prev_;
  T *p_host_c_prime_t_tanh_;
  T *p_host_i_t_;

  T *p_host_sparse_lookup_;

  T *p_host_h_t_;

public:
  // device pointers
  T *p_device_d_errn_to_tp1_ht_;
  T *p_device_d_errn_to_tp1_ct_;
  T *p_device_d_errt_ht_;


  T *p_device_o_t_;                          // lstm_size_ x minibatch_size_, o_t = sigmoid((p_input_to_hidden_layer_->p_device_m_o_ * p_device_sparse_lookup_) +
                                             //                                             (p_input_to_hidden_layer_->p_device_w_ho_ * p_device_h_t_prev_) +
                                             //                                              p_input_to_hidden_layer_->p_device_b_o_)
  T *p_device_c_t_;                          // lstm_size_ x minibatch_size_, c_t[i] = p_device_f_t_[i] * p_device_c_t_prev_[i] +
                                             //                                        p_device_i_t_[i] * p_device_c_prime_t_tanh_[i]

  int *p_device_input_vocab_indices_01_;     // this = input_layer_{source,target}_.p_device_input_vocab_indices_01_full_
                                             // (-1 -1 -1 2 179 0 10 22 0) => (0 0 0 1 1 1 1 1 1)
  int *p_device_input_vocab_indices_;        // this = input_layer_{source,target}_.p_device_input_vocab_indices_full_ 
                                             // (malloc: minibatch_size_ x longest_sent_) (use: minibatch_size_ x current_source_length_)
                                             // (-1 -1 -1 2 179 0 10 22 0) => (0 0 0 2 179 0 10 22 0)

  T *p_device_f_t_;                          // lstm_size_ x minibatch_size_, f_t = sigmoid((p_input_to_hidden_layer_->p_device_m_f_ * p_device_sparse_lookup_) +
                                             //                                             (p_input_to_hidden_layer_->p_device_w_hf_ * p_device_h_t_prev_) +
                                             //                                              p_input_to_hidden_layer_->p_device_b_f_)
  T *p_device_c_t_prev_;                     // init with input_layer_{source,target}_.p_device_init_cell_vec_, 
                                             // lstm_size_ x minibatch_size_
  T *p_device_c_prime_t_tanh_;               // lstm_size_ x minibatch_size_, this = tanh((p_input_to_hidden_layer_->p_device_m_c_ * p_device_sparse_lookup_) +
                                             //                                           (p_input_to_hidden_layer_->p_device_w_hc_ * p_device_h_t_prev_) + 
                                             //                                            p_input_to_hidden_layer_->p_device_b_c_)
  T *p_device_i_t_;                          // lstm_size_ x minibatch_size_, i_t = sigmoid((p_input_to_hidden_layer_->p_device_m_i_ * p_device_sparse_lookup_) +
                                             //                                             (p_input_to_hidden_layer_->p_device_w_hi_ * p_device_h_t_prev_) + 
                                             //                                              p_input_to_hidden_layer_->p_device_b_i_)

  T *p_device_h_t_prev_;                     // init with input_layer_{source,target}_.p_device_init_hidden_vec_ for v_nodes_[0], 
                                             // lstm_size_ x minibatch_size_, for calculting i_t, f_t, c_prime_t_tanh, o_t
  T *p_device_sparse_lookup_;                // lstm_size_ x minibatch_size_, for calculting i_t, f_t, c_prime_t_tanh, o_t
  T *p_device_h_t_;                          // lstm_size_ x minibatch_size_, h_t[i] = p_device_o_t_[i] * tanhf(p_device_c_t_[i])
  T *p_device_zeros_;                        // points to a zero matrix that can be used for p_device_d_errt_ht_ in backprop


  // for feed input
  T *p_device_errn_to_t_h_tild_;             // lstm_size_ x minibatch_size_
  T *p_device_errn_to_t_h_tild_cpy_;         // this is a pointer, this = p_attention_layer_->v_nodes_[i].p_device_errt_to_n_htild_below_ 
                                             // as current node is i + 1
  T *p_device_h_tild_;                       // lstm_size_ x minibatch_size_, for i_t, f_t, c_prime_t_tanh, o_t


public:
  LstmInputHiddenNode() {}                   // Constructor

public:
  void Init(int lstm_size, int minibatch_size, int vocab_size, InputToHiddenLayer<T> *p_input_to_hidden_layer, \
       int index, T *p_device_zero, bool dropout_mode, T dropout_rate);  

public:
  void InitLstmGpu(int lstm_size, int minibatch_size, int vocab_size, InputToHiddenLayer<T> *p_input_to_hidden_layer);

public:
  void UpdateVectorsForwardGpu(int *p_device_input_vocab_indices, int *p_device_input_vocab_indices_01, T *p_device_h_t_prev, T *p_device_c_t_prev);

public:
  // Compute the forward values for the LSTM node, this is after the node has received the previous hidden and cell state values
  void ForwardProp();
  void ForwardPropGpu();

public:
  void BackPropPreprocessGpu(T *p_device_d_errn_to_tp1_ht, T *p_device_d_errn_to_tp1_ct);
  void BackPropGpu(int index);


public:
  void UpdateVectorsForwardDecoder(int *p_device_input_vocab_indices, int *p_device_input_vocab_indices_01);

public:
  void ComputeGradientsGpu();


private:
  void SendHTAbove();

public:
  void AttentionExtra();


};


// Constructor
// CHECK: OK //
template <typename T>
void LstmInputHiddenNode<T>::Init(int lstm_size, int minibatch_size, int vocab_size, InputToHiddenLayer<T> *p_input_to_hidden_layer, \
                                  int index, T *p_device_zeros, bool dropout_mode, T dropout_rate) {
  p_input_to_hidden_layer_ = p_input_to_hidden_layer;
  dropout_mode_ = dropout_mode;
  dropout_rate_ = dropout_rate;

  InitLstmGpu(lstm_size, minibatch_size, vocab_size, p_input_to_hidden_layer);

  minibatch_size_ = minibatch_size;
  lstm_size_ = lstm_size;
  index_ = index;
  p_device_zeros_ = p_device_zeros;
}


// CHECK: OK //
template <typename T>
void LstmInputHiddenNode<T>::InitLstmGpu(int lstm_size, int minibatch_size, int vocab_size, InputToHiddenLayer<T> *p_input_to_hidden_layer) {
  cudaSetDevice(p_input_to_hidden_layer_->input_hidden_layer_information_.device_number_);

  FullMatrixSetup(&p_host_o_t_, &p_device_o_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_c_t_, &p_device_c_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_f_t_, &p_device_f_t_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_c_prime_t_tanh_, &p_device_c_prime_t_tanh_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_i_t_, &p_device_i_t_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_sparse_lookup_, &p_device_sparse_lookup_, lstm_size, minibatch_size);

  FullMatrixSetup(&p_host_h_t_, &p_device_h_t_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errt_ht_, &p_device_d_errt_ht_, lstm_size, minibatch_size);

  cudaMemset(p_device_d_errt_ht_, 0, lstm_size * minibatch_size * sizeof(T));

  // allocate a matrix that will have values between zero and one
  if (dropout_mode_) {
    CudaErrorWrapper(cudaMalloc((void **)&p_device_dropout_mask_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  }
}


// CHECK: OK //
template <typename T>
void LstmInputHiddenNode<T>::UpdateVectorsForwardGpu(int *p_device_input_vocab_indices, int *p_device_input_vocab_indices_01, T *p_device_h_t_prev, T *p_device_c_t_prev) {
  p_device_h_t_prev_ = p_device_h_t_prev;
  p_device_c_t_prev_ = p_device_c_t_prev;
  
  p_device_input_vocab_indices_ = p_device_input_vocab_indices;
  p_device_input_vocab_indices_01_ = p_device_input_vocab_indices_01;
}


// CHECK: OK //
// Forward Prop Gpu for Input to Hidden Layer
template <typename T>
void LstmInputHiddenNode<T>::ForwardProp() {
  ForwardPropGpu();
}


// Forward Prop Gpu for Input to Hidden Layer
template <typename T>
void LstmInputHiddenNode<T>::ForwardPropGpu() {

#ifdef DEBUG_CHECKPOINT_7
  std::cerr<<"\n************CP7 In *LstmInputHiddenNode* *ForwardPropGpu*\n"<<std::flush;
  std::cerr<<"   ForwardPropGpu 1\n"
           <<"   lstm_size_: "<<lstm_size_<<"\n"
           <<"   minibatch_size_: "<<minibatch_size_<<"\n"<<std::flush;
#endif

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaSetDevice(p_input_to_hidden_layer_->input_hidden_layer_information_.device_number_);

  // operation
  // using stream 0
  // compute tmp_mat(model->w)
  int threads_per_block = 128;
  int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel(minibatch_size_, num_block, 1);
  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu PRE SPARSE");

  if (!p_input_to_hidden_layer_->char_cnn_mode_) {
    // Init p_device_sparse_lookup_ (lstm_size_ x minibatch_size_) using word embedding p_input_to_hidden_layer_->p_device_w_ (lstm_size x vocab_size_)
    SparseLookupKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(p_device_sparse_lookup_, p_input_to_hidden_layer_->p_device_w_, p_device_input_vocab_indices_, minibatch_size_, lstm_size_);
  } else {
    // char_cnn_mode_ is not written
  }
  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu SPARSE");
  // ABOVE CHECK DECODER: OK //

#ifdef DEBUG_NEWCHECKPOINT_1
  DeviceSyncAll();
  CudaGetLastError("checkpoint b1");
#endif



#ifdef DEBUG_CHECKPOINT_7
  std::cerr<<"   dropout_mode_ : "<<dropout_mode_<<"\n"
           <<"   train_mode_: "<<p_input_to_hidden_layer_->p_neural_mt_->train_mode_<<"\n"
           <<std::flush;
#endif


  if (dropout_mode_ && p_input_to_hidden_layer_->p_neural_mt_->train_mode_) {
    // dropout and train
    if (!p_input_to_hidden_layer_->p_neural_mt_->grad_check_flag_) {
      curandSetStream(p_input_to_hidden_layer_->rand_generator_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
      CurandGenerateUniformWrapper(p_device_dropout_mask_, lstm_size_ * minibatch_size_, p_input_to_hidden_layer_->rand_generator_);
    }
    // p_device_sparse_lookup_ dropout
    DropoutKernel<<<256, 256, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(p_device_dropout_mask_, p_input_to_hidden_layer_->dropout_rate_, p_device_sparse_lookup_, lstm_size_ * minibatch_size_);
  }
  // ABOVE CHECK: OK //
#ifdef DEBUG_NEWCHECKPOINT_1
  DeviceSyncAll();
  CudaGetLastError("checkpoint b2");
#endif

  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  // Operation, using streams 1 and 2
  T alpha = 1;
  T beta = 0;


  // stream 1
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s01_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s01_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp1_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_m_i_ (lstm_size_ x lstm_size_) * 
  //                                                                           p_device_sparse_lookup_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_i_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp1_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (i_t) p_device_tmp1_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.i_t_part1_, p_input_to_hidden_layer_->input_hidden_layer_information_.s01_);


  // stream 2
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s02_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s02_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp2_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_w_hi_ (lstm_size_ x lstm_size_) *
  //                                                                           p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  // p_device_h_t_prev_ is initialized by input_layer_{source,target}_.p_device_init_hidden_vec_
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_w_hi_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp2_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (i_t) p_device_tmp2_ failed\n");


#ifdef REMOVE_STREAMS
  DeviceSyncAll();
  CudaGetLastError("IH: Check Here");
#endif
  // ABOVE CHECK: OK //


#ifdef DEBUG_CHECKPOINT_7
  std::cerr<<"   feed_input_mode_ : "<<feed_input_mode_<<"\n"
           <<"   index_: "<<index_<<"\n"
           <<std::flush;
#endif
  
  // use feed_input_mode_ just in attention model, target side
  if (feed_input_mode_ && 0 != index_) {

#ifdef REMOVE_STREAMS_FEED_INPUT
    DeviceSyncAll();
#endif

    // stream 2
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s02_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s02_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s02_, p_input_to_hidden_layer_->input_hidden_layer_information_.attention_forward_, 0);

    // p_input_to_hidden_layer_->p_device_tmp9_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_q_i_ (lstm_size_ x lstm_size_) * 
    //                                                                           p_device_h_tild_ (lstm_size_ * minibatch_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_q_i_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp9_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (i_t) p_device_tmp9_ failed\n");
  }
  // ABOVE CHECK: OK //

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
  CudaGetLastError("IH: Check 1234");
#endif

  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu i_t for lower level LSTM P1");
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s02_, p_input_to_hidden_layer_->input_hidden_layer_information_.i_t_part1_, 0);

  if (!feed_input_mode_ || 0 == index_) {
    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp1_ (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp2_ (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_b_i_ (lstm_size_ x 1)
    // p_device_i_t_ (lstm_size x minibatch_size_) = 1.0f / (1.0f + expf(-1.0f * tmp_value)), this is sigmoid function, value is (0,1)
    ForwardSigmoidKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s02_>>>(p_device_i_t_, p_input_to_hidden_layer_->p_device_tmp1_, p_input_to_hidden_layer_->p_device_tmp2_, p_input_to_hidden_layer_->p_device_b_i_, lstm_size_);
  } else {
    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp1_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp2_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp9_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_b_i_[i] (lstm_size_ x 1)
    // p_device_i_t_[i] (lstm_size x minibatch_size_) = 1.0f / (1.0f + exp(-1.0f * tmp_value)), this is sigmoid function, value is (0,1)
    ForwardSigmoidKernelFeed<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s02_>>>(p_device_i_t_, p_input_to_hidden_layer_->p_device_tmp1_, p_input_to_hidden_layer_->p_device_tmp2_, p_input_to_hidden_layer_->p_device_tmp9_, p_input_to_hidden_layer_->p_device_b_i_, lstm_size_);
  }


  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu i_t for lower level LSTM");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.i_t_full_, p_input_to_hidden_layer_->input_hidden_layer_information_.s02_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
  CudaGetLastError("i_t for lower level LSTM P2");
#endif


  // operation, using streams 3 and 4
  alpha = 1;
  beta = 0;

  // stream 3
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s03_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s03_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp3_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_m_f_ (lstm_size_ x lstm_size_) *
  //                                                                           p_device_sparse_lookup_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_f_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp3_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (f_t) p_device_tmp3_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.f_t_part1_, p_input_to_hidden_layer_->input_hidden_layer_information_.s03_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 4
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s04_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s04_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp4_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_w_hf_ (lstm_size_ x lstm_size_)
  //                                                                           p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_w_hf_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp4_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (f_t) p_device_tmp4_ failed\n");


#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  if (feed_input_mode_ && 0 != index_) {

#ifdef REMOVE_STREAMS_FEED_INPUT
    DeviceSyncAll();
#endif

    // stream 4
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s04_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s04_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s04_, p_input_to_hidden_layer_->input_hidden_layer_information_.attention_forward_, 0);

    // p_input_to_hidden_layer_->p_device_tmp10_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_q_f_ (lstm_size_ x lstm_size_) *
    //                                                                            p_device_h_tild_ (lstm_size_ x minibatch_size)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_q_f_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp10_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (f_t) p_device_tmp10_ failed\n");
  }

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s04_, p_input_to_hidden_layer_->input_hidden_layer_information_.f_t_part1_, 0);
  if (!feed_input_mode_ || 0 == index_) {
    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp3_ (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp4_ (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_b_f_ (lstm_size_ x 1)
    // p_device_f_t_ (lstm_size x minibatch_size_) = 1.0f / (1.0f + expf(-1.0f * tmp_value)), this is sigmoid function, value is (0,1)
    ForwardSigmoidKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s04_>>>(p_device_f_t_, p_input_to_hidden_layer_->p_device_tmp3_, p_input_to_hidden_layer_->p_device_tmp4_, p_input_to_hidden_layer_->p_device_b_f_, lstm_size_);
  } else {

    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp3_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp4_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp10_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_b_f_[i] (lstm_size_ x 1)
    // p_device_f_t_[i] (lstm_size x minibatch_size_) = 1.0f / (1.0f + exp(-1.0f * tmp_value)), this is sigmoid function, value is (0,1)
    ForwardSigmoidKernelFeed<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s04_>>>(p_device_f_t_, p_input_to_hidden_layer_->p_device_tmp3_, p_input_to_hidden_layer_->p_device_tmp4_, p_input_to_hidden_layer_->p_device_tmp10_, p_input_to_hidden_layer_->p_device_b_f_, lstm_size_);
  }

  CudaGetLastError("f_t");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.f_t_full_, p_input_to_hidden_layer_->input_hidden_layer_information_.s04_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // Operation, using streams 5 and 6
  alpha = 1;
  beta = 0;

  // stream 5
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s05_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s05_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp5_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_m_c_ (lstm_size_ x lstm_size_) *
  //                                                                           p_device_sparse_lookup_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_c_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp5_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (c_prime_t_tanh) p_device_tmp5_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.c_prime_t_tanh_part1_, p_input_to_hidden_layer_->input_hidden_layer_information_.s05_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 6
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s06_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp6_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_w_hc_ (lstm_size_ x lstm_size_)
  //                                                                           p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_w_hc_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp6_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (c_prime_t_tanh) p_device_tmp6_ failed\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  if (feed_input_mode_ && 0 != index_) {

#ifdef REMOVE_STREAMS_FEED_INPUT
    DeviceSyncAll();
#endif

    // stream 6
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s06_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s06_, p_input_to_hidden_layer_->input_hidden_layer_information_.attention_forward_, 0);

    // p_input_to_hidden_layer_->p_device_tmp11_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_q_c_ (lstm_size_ x lstm_size_) *
    //                                                                            p_device_h_tild_ (lstm_size_ x minibatch_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_q_c_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp11_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (c_prime_t_tanh) p_device_tmp11_ failed\n");
  }

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s06_, p_input_to_hidden_layer_->input_hidden_layer_information_.c_prime_t_tanh_part1_, 0);

  if (!feed_input_mode_ || 0 == index_) {
    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp5_ (lstm_size_ x minibatch_size_) +
    //                                            p_input_to_hidden_layer_->p_device_tmp6_ (lstm_size_ x minibatch_size_) +
    //                                            p_input_to_hidden_layer_->p_device_b_c_ (lstm_size_ x 1)
    // p_device_c_prime_t_tanh_ (lstm_size_ x minibatch_size_) = tanhf(tmp_value), tanh function, (-1,1)
    ForwardTanhKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_>>>(p_device_c_prime_t_tanh_, p_input_to_hidden_layer_->p_device_tmp5_, p_input_to_hidden_layer_->p_device_tmp6_, p_input_to_hidden_layer_->p_device_b_c_, lstm_size_);
  } else {
    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp5_[i] (lstm_size_ x minibatch_size_) +
    //                                            p_input_to_hidden_layer_->p_device_tmp6_[i] (lstm_size_ x minibatch_size_) +
    //                                            p_input_to_hidden_layer_->p_device_tmp11_[i] (lstm_size_ x minibatch_size_) +
    //                                            p_input_to_hidden_layer_->p_device_b_c_[i] (lstm_size_ x 1)
    // p_device_c_prime_t_tanh_[i] (lstm_size_ x minibatch_size_) = tanhf(tmp_value), tanh function, (-1,1)
    ForwardTanhKernelFeed<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_>>>(p_device_c_prime_t_tanh_, p_input_to_hidden_layer_->p_device_tmp5_, p_input_to_hidden_layer_->p_device_tmp6_, p_input_to_hidden_layer_->p_device_tmp11_, p_input_to_hidden_layer_->p_device_b_c_, lstm_size_);
  }

  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu c_prime_t_tanh");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.c_prime_t_tanh_full_, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // Operation, using streams 7 and 8
  alpha = 1;
  beta = 0;

  // stream 7
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s07_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s07_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp7_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_m_o_ (lstm_size_ x lstm_size_) *
  //                                                                           p_device_sparse_lookup_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_o_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp7_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (o_t) p_device_tmp7_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.o_t_part1_, p_input_to_hidden_layer_->input_hidden_layer_information_.s07_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 8
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s08_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);

  // p_input_to_hidden_layer_->p_device_tmp8_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_w_ho_ (lstm_size_ x lstm_size_) *
  //                                                                           p_device_h_t_prev_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_w_ho_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp8_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (o_t) p_device_tmp8_ failed\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  if (feed_input_mode_ && 0 != index_) {

#ifdef REMOVE_STREAMS_FEED_INPUT
    DeviceSyncAll();
#endif

    // stream 8
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s08_, p_input_to_hidden_layer_->input_hidden_layer_information_.sparse_forward_start_, 0);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s08_, p_input_to_hidden_layer_->input_hidden_layer_information_.attention_forward_, 0);

    // p_input_to_hidden_layer_->p_device_tmp12_ (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_q_o_ (lstm_size_ x lstm_size_) *
    //                                                                            p_device_h_tild_ (lstm_size_ x minibatch_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_q_o_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp12_, lstm_size_), "LstmInputHiddenNode::ForwardPropGpu (o_t) p_device_tmp12_ failed\n");
  }

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s08_, p_input_to_hidden_layer_->input_hidden_layer_information_.o_t_part1_, 0);

  if (!feed_input_mode_ || 0 == index_) {
    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp7_ (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp8_ (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_b_o_ (lstm_size_ x 1)
    // p_device_o_t_ (lstm_size x minibatch_size_) = 1.0f / (1.0f + expf(-1.0f * tmp_value)), this is sigmoid function, value is (0,1)
    ForwardSigmoidKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_>>>(p_device_o_t_, p_input_to_hidden_layer_->p_device_tmp7_, p_input_to_hidden_layer_->p_device_tmp8_, p_input_to_hidden_layer_->p_device_b_o_, lstm_size_);
  } else {
    // tmp_value (lstm_size_ x minibatch_size_) = p_input_to_hidden_layer_->p_device_tmp7_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp8_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_tmp12_[i] (lstm_size_ x minibatch_size) +
    //                                            p_input_to_hidden_layer_->p_device_b_o_[i] (lstm_size_ x 1)
    // p_device_o_t_[i] (lstm_size x minibatch_size_) = 1.0f / (1.0f + expf(-1.0f * tmp_value)), this is sigmoid function, value is (0,1)
    ForwardSigmoidKernelFeed<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_>>>(p_device_o_t_, p_input_to_hidden_layer_->p_device_tmp7_, p_input_to_hidden_layer_->p_device_tmp8_, p_input_to_hidden_layer_->p_device_tmp12_, p_input_to_hidden_layer_->p_device_b_o_, lstm_size_);
  }

  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu o_t");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.o_t_full_, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // Operation, for now the rest are using the default stream
  // stream 0
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.i_t_full_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.f_t_full_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.c_prime_t_tanh_full_, 0);

  // p_device_c_t_[i] (lstm_size_ x minibatch_size_) = p_device_f_t_[i] (lstm_size_ x minibatch_size_) * p_device_c_t_prev_[i] (lstm_size x minibatch_size_) +
  //                                                   p_device_i_t_[i] (lstm_size_ x minibatch_size_) * p_device_c_prime_t_tanh_[i] (lstm_size_ x minibatch_size_)
  ForwardCTKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(p_device_c_t_, p_device_f_t_, p_device_c_t_prev_, p_device_i_t_, p_device_c_prime_t_tanh_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu c_t");

#ifdef DEBUG_DROPOUT
  std::cerr<<"   cell_clip_mode__: "<<cell_clip_mode__<<"\n"<<std::flush;
#endif

  if (cell_clip_mode__) {
    // the value of p_device_c_t_[i] (lstm_size x minibatch_size) is between [-cell_clip_threshold__, cell_clip_threshold__]
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1)/256), 256, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(p_device_c_t_, cell_clip_threshold__, lstm_size_ * minibatch_size_);
  }

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  // Operation
  // stream 0
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.o_t_full_, 0);

  // p_device_h_t_[i] = p_device_o_t_[i] * tanhf(p_device_c_t_[i])
  ForwardHTKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(p_device_h_t_, p_device_o_t_, p_device_c_t_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu h_t");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif
  
  // p_device_h_t_[i] (lstm_size_ x minibatch_size_) = p_device_h_t_[i] (lstm_size_ x minibatch_size_) * p_device_input_vocab_indices_01_[j] (minibatch_size_ x ?(longest_sentence_))
  ZeroCTAndHT<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(p_device_h_t_, p_device_c_t_, p_device_input_vocab_indices_01_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::ForwardPropGpu zero");


#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  SendHTAbove();

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaSetDevice(0);
}


template <typename T>
void LstmInputHiddenNode<T>::BackPropPreprocessGpu(T *p_device_d_errn_to_tp1_ht, T *p_device_d_errn_to_tp1_ct) {
  p_device_d_errn_to_tp1_ht_ = p_device_d_errn_to_tp1_ht;
  p_device_d_errn_to_tp1_ct_ = p_device_d_errn_to_tp1_ct;
}


template <typename T>
void LstmInputHiddenNode<T>::BackPropGpu(int index) {

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaSetDevice(p_input_to_hidden_layer_->input_hidden_layer_information_.device_number_);
  
  // back prop node starting
  if (p_input_to_hidden_layer_->upper_layer_.upper_softmax_mode_) {
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->upper_layer_.p_softmax_->GetErrHTEvent(), 0);
  } else {
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->upper_layer_.p_hidden_layer_->hidden_hidden_layer_information_.d_error_ht_done_, 0);
  }

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.ctm1_done_, 0);

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_full_done_, 0);

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_hi_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_hf_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_ho_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_hc_grad_done_, 0);

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.m_i_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.m_f_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.m_o_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.m_c_grad_done_, 0);

  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.b_i_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.b_f_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.b_o_grad_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->input_hidden_layer_information_.b_c_grad_done_, 0);

  // now pass the error to the attention node if the attention model is in place
  // deal with the feed input here
  if (attention_model_mode_) {

    // multi_attention is not written

    cudaEventRecord(p_input_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.start_backward_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
    p_input_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].BackProp();
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.backward_prop_done_, 0);

    // multi_attention is not written
  }

  // operation
  T alpha = 1;
  T beta = 1;
  // stream 0
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_ = p_device_d_errn_to_tp1_ht_ + p_device_d_errt_ht_
  CublasErrorWrapper(CublasGeamWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_device_d_errn_to_tp1_ht_, lstm_size_, &beta, p_device_d_errt_ht_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ht_\n");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // nonrev_bi_dir is not written

  // operation
  int threads_per_block = 128;
  int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel(minibatch_size_, num_block, 1);
  // p_input_to_hidden_layer_->p_device_d_errt_ct_[i] = p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_[i] * 
  //                                                    p_device_o_t_[i] * 
  //                                                    (1.0f - (tanhf(p_device_c_t_[i]))^2);
  DErrtCTKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(p_input_to_hidden_layer_->p_device_d_errt_ct_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_, p_device_o_t_, p_device_c_t_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errt_ct_");

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_ = p_device_d_errn_to_tp1_ct_ + p_input_to_hidden_layer_->p_device_d_errt_ct_
  CublasErrorWrapper(CublasGeamWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_device_d_errn_to_tp1_ct_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_d_errt_ct_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ct_\n");

  // nonrev_bi_dir is not written

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // zero out columns of p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_
  ZeroColumnsKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_, p_device_input_vocab_indices_01_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ht_ ZeroColumnsKernel");

  // zero out columns of p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_
  ZeroColumnsKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_>>>(lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_input_vocab_indices_01_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ct_ ZeroColumnsKernel");

  // event for finishing the first stuff
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.backprop_init_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);


  // starting from this point streams will be used
  // operation
  // stream 1
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s01_, p_input_to_hidden_layer_->input_hidden_layer_information_.backprop_init_, 0);
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_[i] = p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_[i] * 
  //                                                         tanhf(p_device_c_t_[i]) * 
  //                                                         p_device_o_t_[i] * (1 - p_device_o_t_[i])
  DErrnToTOTKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s01_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_, p_device_o_t_, p_device_c_t_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ot_");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.error_ot_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s01_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 02
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s02_, p_input_to_hidden_layer_->input_hidden_layer_information_.backprop_init_, 0);
  // p_device_d_errn_to_t_ft_.transpose() = p_device_d_errn_to_t_ct_.transpose().array() * (c_t_prev_.array()) * f_t_ * (1 - f_t_)
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_[i] = p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_[i] * 
  //                                                         p_device_c_t_prev_[i] * 
  //                                                         p_device_f_t_[i] * (1 - p_device_f_t_[i])
  DErrnToTFTITKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s02_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_c_t_prev_, p_device_f_t_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ft_");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.error_ft_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s02_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 03
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s03_, p_input_to_hidden_layer_->input_hidden_layer_information_.backprop_init_, 0);
  // p_device_errn_to_tanhcpt_.transpose() = p_device_errn_to_t_ct_.transpose().array() * (i_t_.array())
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_[i] = p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_[i] * 
  //                                                              p_device_i_t_[i] * 
  //                                                              (1 - p_device_c_prime_t_tanh_[i]^2)
  DErrnToTTanhcptKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s03_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_i_t_, p_device_c_prime_t_tanh_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_tanhcpt_");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.error_tanhcpt_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s03_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 04
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s04_, p_input_to_hidden_layer_->input_hidden_layer_information_.backprop_init_, 0);
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_it_[i] = p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_[i] * 
  //                                                         p_device_c_prime_t_tanh_[i] * 
  //                                                         p_device_i_t_[i] * (1 - p_device_i_t_[i])
  DErrnToTFTITKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s04_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_c_prime_t_tanh_, p_device_i_t_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_it_");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.error_it_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s04_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // using streams 5, 6, 7, 8, 9
  // p_device_errn_to_t_htm1_.transpose() = (w_ho_.transpose() * ((p_device_errn_to_t_ot_.transpose().array() * o_t_.array() * (1 - o_t_.array())).matrix())) \
  //                                      + (w_hf_.transpose() * ((p_device_errn_to_t_ft_.transpose().array() * f_t_.array() * (1 - f_t_.array())).matrix())) \
  //                                      + (w_hi_.transpose() * ((p_device_errn_to_t_it_.transpose().array() * i_t_.array() * (1 - i_t_.array())).matrix())) \
  //                                      + (w_hc_.transpose() * ((p_device_errn_to_t_tanhcpt_.transpose().array() * (1 - c_prime_t_tanh_.array().square())).matrix()))

  T alpha2 = 1;
  T beta2 = 0;

  // stream 05
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s05_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s05_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ot_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp1_ = p_input_to_hidden_layer_->p_device_w_ho_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_w_ho_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp1_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp1_ htm1\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p1_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s05_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 06
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s06_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ft_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp2_ = p_input_to_hidden_layer_->p_device_w_hf_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_w_hf_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp2_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp2_ htm1\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p2_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 07
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s07_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s07_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_it_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp3_ = p_input_to_hidden_layer_->p_device_w_hi_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_w_hi_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp3_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp3_ htm1\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p3_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s07_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 08
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s08_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp4_ = p_input_to_hidden_layer_->p_device_w_hc_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_w_hc_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp4_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp4_ htm1\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p4_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 09
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p1_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p2_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p3_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p4_done_, 0);
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_m1_[i] = p_input_to_hidden_layer_->p_device_tmp1_[i] + 
  //                                                            p_input_to_hidden_layer_->p_device_tmp2_[i] + 
  //                                                            p_input_to_hidden_layer_->p_device_tmp3_[i] + 
  //                                                            p_input_to_hidden_layer_->p_device_tmp4_[i]
  AddFourMatricesKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_m1_, p_input_to_hidden_layer_->p_device_tmp1_, p_input_to_hidden_layer_->p_device_tmp2_, p_input_to_hidden_layer_->p_device_tmp3_, p_input_to_hidden_layer_->p_device_tmp4_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ht_m1_");

  if (cell_clip_mode__) {
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_ht_m1_, error_clip_threshold__, lstm_size_ * minibatch_size_);
  }

  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_done_tmp_, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_);

  // operation
  // send error to the attention model
  if (feed_input_mode_ && 0 != index_) {

#ifdef REMOVE_STREAMS_FEED_INPUT
    DeviceSyncAll();
#endif

    // stream 05
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s05_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s05_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_done_tmp_, 0);
    // p_input_to_hidden_layer_->p_device_tmp1_ = p_input_to_hidden_layer_->p_device_q_o_^T (lstm_size_ x lstm_size_) * 
    // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_q_o_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp1_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp1_ feedinput\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p1_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s05_);

#ifdef REMOVE_STREAMS
    DeviceSyncAll();
#endif

    // stream 06
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s06_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_done_tmp_, 0);
    // p_input_to_hidden_layer_->p_device_tmp2_ = p_input_to_hidden_layer_->p_device_q_f_^T (lstm_size_ x lstm_size_)
    // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_q_f_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp2_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp2_ feedinput\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p2_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s06_);

#ifdef REMOVE_STREAMS
    DeviceSyncAll();
#endif

    // stream 07
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s07_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s07_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_done_tmp_, 0);
    // p_input_to_hidden_layer_->p_device_tmp3_ = p_input_to_hidden_layer_->p_device_q_i_^T (lstm_size_ x lstm_size_) *
    // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_q_i_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp3_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp3_ feedinput\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p3_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s07_);

#ifdef REMOVE_STREAMS
    DeviceSyncAll();
#endif

    // stream 8
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s08_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_done_tmp_, 0);
    // p_input_to_hidden_layer_->p_device_tmp4_ = p_input_to_hidden_layer_->p_device_q_c_^T (lstm_size_ x lstm_size_) *
    // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha2, p_input_to_hidden_layer_->p_device_q_c_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, &beta2, p_input_to_hidden_layer_->p_device_tmp4_, lstm_size_), "LstmInputHiddenNode::BackPropGpu p_device_tmp4_ feedinput\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p4_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s08_);

#ifdef REMOVE_STREAMS
    DeviceSyncAll();
#endif

    // stream 09
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p1_done_, 0);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p2_done_, 0);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p3_done_, 0);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s09_, p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_p4_done_, 0);
    // p_device_errn_to_t_h_tild_ = p_input_to_hidden_layer_->p_device_tmp1_ +
    //                              p_input_to_hidden_layer_->p_device_tmp2_ +
    //                              p_input_to_hidden_layer_->p_device_tmp3_ +
    //                              p_input_to_hidden_layer_->p_device_tmp4_
    AddFourMatricesKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_>>>(p_device_errn_to_t_h_tild_, p_input_to_hidden_layer_->p_device_tmp1_, p_input_to_hidden_layer_->p_device_tmp2_, p_input_to_hidden_layer_->p_device_tmp3_, p_input_to_hidden_layer_->p_device_tmp4_, lstm_size_);
    CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_errn_to_t_h_tild_ feedinput");

    if (cell_clip_mode__) {
      ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_>>>(p_device_errn_to_t_h_tild_, error_clip_threshold__, lstm_size_ * minibatch_size_);
    }

    cudaMemcpyAsync(p_device_errn_to_t_h_tild_cpy_, p_device_errn_to_t_h_tild_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_);
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.error_htild_below_, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_);
  }

  // don't record this event until after the feed input
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.htm1_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s09_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // stream 10
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s10_, p_input_to_hidden_layer_->input_hidden_layer_information_.backprop_init_, 0);
  // p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_m1_[i][j] = p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_[i][j] * 
  //                                                               p_device_f_t_[i][j]
  ElementwiseMultKernel<<<kernel, threads_per_block, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s10_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_, p_device_f_t_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_m1_, lstm_size_);
  CudaGetLastError("LstmInputHiddenNode::BackPropGpu p_device_d_errn_to_t_ct_m1_");

  if (cell_clip_mode__) {
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s10_>>>(p_input_to_hidden_layer_->p_device_d_errn_to_t_ct_, error_clip_threshold__, lstm_size_ * minibatch_size_);
  }
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.ctm1_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s10_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  ComputeGradientsGpu();
}


template <typename T>
void LstmInputHiddenNode<T>::UpdateVectorsForwardDecoder(int *p_device_input_vocab_indices, int *p_device_input_vocab_indices_01) {
  // GPU stuff
  p_device_input_vocab_indices_ = p_device_input_vocab_indices;
  p_device_input_vocab_indices_01_ = p_device_input_vocab_indices_01;
}




template <typename T>
void LstmInputHiddenNode<T>::ComputeGradientsGpu() {

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // streams 11, 12, 13, 14

  T alpha = 1;
  T beta = 1;

  // stream 11
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s11_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s11_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_it_done_, 0);
  // p_input_to_hidden_layer_->p_device_w_hi_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                        p_device_h_t_prev_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_w_hi_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_w_hi_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_hi_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s11_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 12
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s12_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s12_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ft_done_, 0);
  // p_input_to_hidden_layer_->p_device_w_hf_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                        p_device_h_t_prev_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_w_hf_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_w_hf_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_hf_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s12_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 13
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s13_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s13_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_input_to_hidden_layer_->p_device_w_hc_grad_ = p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_h_t_prev_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_w_hc_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_w_hc_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_hc_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s13_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 14
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s14_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s14_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ot_done_, 0);
  // p_input_to_hidden_layer_->p_device_w_ho_grad_ = p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_h_t_prev_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, p_device_h_t_prev_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_w_ho_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_w_ho_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_ho_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s14_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // operation
  // streams 15, 16, 17, 18
  // compute_temp_mat(p_input_to_hidden_layer_->w)

  alpha = 1;
  beta = 1;

  // stream 15
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s15_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s15_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_it_done_, 0);
  // p_input_to_hidden_layer_->p_device_m_i_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x minibatch_size_)                  p_device_sparse_lookup_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_m_i_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_m_i_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_i_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s15_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 16
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s16_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s16_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ft_done_, 0);
  // p_input_to_hidden_layer_->p_device_m_f_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_sparse_lookup_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_m_f_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_m_f_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_f_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s16_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 17
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s17_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s17_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ot_done_, 0);
  // p_input_to_hidden_layer_->p_device_m_o_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_sparse_lookup_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_m_o_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_m_o_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_o_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s17_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 18
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s18_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s18_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_input_to_hidden_layer_->p_device_m_c_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x lstm_size_)                       p_device_sparse_lookup_^T (minibatch_size_ x lstm_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, p_device_sparse_lookup_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_m_c_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_m_c_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_c_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s18_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // now do all of the q's
  // reuse all events from m_i
  if (feed_input_mode_ && 0 != index_) {
    alpha = 1;
    beta = 1;

#ifdef REMOVE_STREAMS_FEED_INPUT
    DeviceSyncAll();
#endif

    // stream 15
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s15_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s15_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_it_done_, 0);
    // p_input_to_hidden_layer_->p_device_q_i_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_) *
    // (lstm_size_ x lstm_size_)                       p_device_h_tild_^T (minibatch_size_ x lstm_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_q_i_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_q_i_grad_ failed\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_i_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s15_);

#ifdef REMOVE_STREAMS
    DeviceSyncAll();
#endif

    // stream 16
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s16_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s16_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ft_done_, 0);
    // p_input_to_hidden_layer_->p_device_q_f_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_) *
    // (lstm_size_ x lstm_size_)                       p_device_h_tild_^T (minibatch_size_ x lstm_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_q_f_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_q_f_grad_ failed\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_f_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s16_);

#ifdef REMOVE_STREAMS
    DeviceSyncAll();
#endif

    // stream 17
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s17_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s17_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ot_done_, 0);
    // p_input_to_hidden_layer_->p_device_q_o_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_) *
    // (lstm_size_ x lstm_size_)                       p_device_h_tild_^T (minibatch_size_ x lstm_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_q_o_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_q_o_grad_ failed\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_o_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s17_);

#ifdef REMOVE_STREAMS
    DeviceSyncAll();
#endif

    // stream 18
    cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s18_);
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s18_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_tanhcpt_done_, 0);
    // p_input_to_hidden_layer_->p_device_q_c_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_) *
    // (lstm_size_ x lstm_size_)                       p_device_h_tild_^T (minibatch_size_ x lstm_size_)
    CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, p_device_h_tild_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_q_c_grad_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_q_c_grad_ failed\n");
    cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.m_c_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s18_);
  }

  // operation
  // stream 19
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s19_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s19_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_it_done_, 0);
  // p_input_to_hidden_layer_->p_device_b_i_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_input_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1)
  CublasErrorWrapper(CublasGemvWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, p_input_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_input_to_hidden_layer_->p_device_b_i_grad_, 1), "LstmInputHiddenNode::ComputeGradientsGpu p_device_b_i_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.b_i_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s19_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 20
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s20_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s20_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ft_done_, 0);
  // p_input_to_hidden_layer_->p_device_b_f_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_input_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1)
  CublasErrorWrapper(CublasGemvWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, p_input_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_input_to_hidden_layer_->p_device_b_f_grad_, 1), "LstmInputHiddenNode::ComputeGradientsGpu p_device_b_f_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.b_f_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s20_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 21
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s21_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s21_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ot_done_, 0);
  // p_input_to_hidden_layer_->p_device_b_o_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_input_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1)
  CublasErrorWrapper(CublasGemvWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, p_input_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_input_to_hidden_layer_->p_device_b_o_grad_, 1), "LstmInputHiddenNode::ComputeGradientsGpu p_device_b_o_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.b_o_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s21_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 22
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s22_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s22_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_input_to_hidden_layer_->p_device_b_c_grad_ += p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_) *
  // (lstm_size_ x 1)                                p_input_to_hidden_layer_->p_device_ones_minibatch_ (minibatch_size_ x 1)
  CublasErrorWrapper(CublasGemvWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, p_input_to_hidden_layer_->p_device_ones_minibatch_, 1, &beta, p_input_to_hidden_layer_->p_device_b_c_grad_, 1), "LstmInputHiddenNode::ComputeGradientsGpu p_device_b_c_grad_ failed\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.b_c_grad_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s22_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

#ifdef DEBUG_DROPOUT_3
  std::cerr<<"   char_cnn_mode_: "<<p_input_to_hidden_layer_->char_cnn_mode_<<"\n"<<std::flush;
#endif

  // operation
  alpha = 1;
  beta = 0;

  // stream 23
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s23_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s23_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_it_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp5_ = p_input_to_hidden_layer_->p_device_m_i_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_it_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_i_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_it_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp5_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_tmp5_ w_grad\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p1_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s23_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 24
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s24_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s24_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ft_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp6_ = p_input_to_hidden_layer_->p_device_m_f_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_f_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ft_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp6_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_tmp6_ w_grad\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p2_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s24_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 25
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s25_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s25_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_ot_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp7_ = p_input_to_hidden_layer_->p_device_m_o_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_o_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_ot_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp7_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_tmp7_ w_grad\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p3_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s25_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 26
  cublasSetStream(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, p_input_to_hidden_layer_->input_hidden_layer_information_.s26_);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s26_, p_input_to_hidden_layer_->input_hidden_layer_information_.error_tanhcpt_done_, 0);
  // p_input_to_hidden_layer_->p_device_tmp8_ = p_input_to_hidden_layer_->p_device_m_c_^T (lstm_size_ x lstm_size_) *
  // (lstm_size_ x minibatch_size_)             p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_input_to_hidden_layer_->input_hidden_layer_information_.handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_input_to_hidden_layer_->p_device_m_c_, lstm_size_, p_input_to_hidden_layer_->p_device_d_errn_to_t_tanhcpt_, lstm_size_, &beta, p_input_to_hidden_layer_->p_device_tmp8_, lstm_size_), "LstmInputHiddenNode::ComputeGradientsGpu p_device_tmp8_ w_grad\n");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p4_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s26_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  // stream 27
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s27_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p1_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s27_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p2_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s27_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p3_done_, 0);
  cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s27_, p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_p4_done_, 0);


  int threads_per_block = 128;
  int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel(minibatch_size_, num_block, 1);

  if (!dropout_mode_) {
    // p_input_to_hidden_layer_->p_device_small_w_grad_ (lstm_size_ x (minibatch_size_ * longest_sent_)) is aligned to 
    // p_device_input_vocab_indices_wgrad_
    WSmallGradientKernel<<<256, 256, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s27_>>>(p_input_to_hidden_layer_->p_device_small_w_grad_, p_input_to_hidden_layer_->p_device_reverse_unique_indices_, p_input_to_hidden_layer_->p_device_tmp5_, p_input_to_hidden_layer_->p_device_tmp6_, p_input_to_hidden_layer_->p_device_tmp7_, p_input_to_hidden_layer_->p_device_tmp8_, p_device_input_vocab_indices_, lstm_size_, minibatch_size_);

  } else {

    WSmallGradientKernelDropout<<<256, 256, 0, p_input_to_hidden_layer_->input_hidden_layer_information_.s27_>>>(p_input_to_hidden_layer_->p_device_small_w_grad_, p_input_to_hidden_layer_->p_device_reverse_unique_indices_, p_input_to_hidden_layer_->p_device_tmp5_, p_input_to_hidden_layer_->p_device_tmp6_, p_input_to_hidden_layer_->p_device_tmp7_, p_input_to_hidden_layer_->p_device_tmp8_, p_device_input_vocab_indices_, lstm_size_, minibatch_size_, p_device_dropout_mask_, dropout_rate_);

  }

  CudaGetLastError("LstmInputHiddenNode::ComputeGradientsGpu p_device_small_w_grad_");
  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.w_grad_full_done_, p_input_to_hidden_layer_->input_hidden_layer_information_.s27_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif
}


// CHECK: OK //
// this is to be called if feed input is true
template <typename T>
void LstmInputHiddenNode<T>::AttentionExtra() {

  cudaSetDevice(p_input_to_hidden_layer_->input_hidden_layer_information_.device_number_);
  T *p_host_tmp;
  FullMatrixSetup(&p_host_tmp, &p_device_errn_to_t_h_tild_, lstm_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_h_tild_, lstm_size_, minibatch_size_);
  feed_input_mode_ = true;

}


// CHECK: OK //
template <typename T>
void LstmInputHiddenNode<T>::SendHTAbove() {

  if (p_input_to_hidden_layer_->p_neural_mt_->decode_mode_) {
    index_ = 0;
  }

  // run forward prop for attention model
  if (attention_model_mode_) {
    cudaEventRecord(p_input_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.start_forward_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
    p_input_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].ForwardProp();
    cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->p_attention_layer_->attention_layer_gpu_information_.forward_prop_done_, 0);

    if (multi_attention_mode_) {
      cudaEventRecord(p_input_to_hidden_layer_->p_attention_layer_bi_->attention_layer_gpu_information_.start_forward_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
      p_input_to_hidden_layer_->p_attention_layer_bi_->v_nodes_[index_].ForwardProp();
      cudaStreamWaitEvent(p_input_to_hidden_layer_->input_hidden_layer_information_.s00_, p_input_to_hidden_layer_->p_attention_layer_bi_->attention_layer_gpu_information_.forward_prop_done_, 0);
    }

    if (multi_attention_mode_) {
      // combiner layer is not written
    }
  }

#ifdef DEBUG_DROPOUT
  std::cerr<<"\n"
           <<"   copy_h_t_mode_: "<<p_input_to_hidden_layer_->upper_layer_.copy_h_t_mode_<<"\n"
           <<"   upper_softmax_mode_: "<<p_input_to_hidden_layer_->upper_layer_.upper_softmax_mode_<<"\n"
           <<"   source_side_mode_: "<<p_input_to_hidden_layer_->upper_layer_.source_side_mode_<<"\n"
           <<"   attention_model_mode_: "<<attention_model_mode_<<"\n"
           <<std::flush;
#endif
  // send the finished p_device_h_t_ to the above layer
  if (p_input_to_hidden_layer_->upper_layer_.copy_h_t_mode_) {
    // transfer h_t to the layer above
    if (p_input_to_hidden_layer_->upper_layer_.upper_softmax_mode_) {
      // upper layer is softmax
      if (!p_input_to_hidden_layer_->upper_layer_.source_side_mode_) {
        // target side
        if (!attention_model_mode_) {
          cudaMemcpyAsync(p_input_to_hidden_layer_->upper_layer_.p_softmax_->GetHTPtr(index_), p_device_h_t_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
        } else {
          // multi_attention is not written
          cudaMemcpyAsync(p_input_to_hidden_layer_->upper_layer_.p_softmax_->GetHTPtr(index_), p_input_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
        }
      }

      // bi_dir is not written 

    }else {
      // upper layer is hidden

#ifdef DEBUG_DROPOUT
      std::cerr<<"\n\n\n where?";
#endif
        
        
      if (!attention_model_mode_) {

#ifdef DEBUG_DROPOUT
          std::cerr<<"go here!!!\n\n\n";
#endif

        cudaMemcpyAsync(p_input_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_, p_device_h_t_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);  
      } else {
        // never go here, upper layer is hidden layer, so input layer does not have attention
#ifdef DEBUG_DROPOUT
        std::cerr<<"never go here!!!\n\n\n";
#endif
        cudaMemcpyAsync(p_input_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_, p_input_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
      }
    }
  } else {
    if (p_input_to_hidden_layer_->upper_layer_.upper_softmax_mode_) {
      // upper layer is softmax
      if (!p_input_to_hidden_layer_->upper_layer_.source_side_mode_) {
        // target side
        if (!attention_model_mode_) {
          p_input_to_hidden_layer_->upper_layer_.p_softmax_->SetHTPtr(index_, p_device_h_t_);  
        } else {
          p_input_to_hidden_layer_->upper_layer_.p_softmax_->SetHTPtr(index_, p_input_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_);
        }
      }
    } else {
      // upper layer is hidden
      if (!attention_model_mode_) {
        p_input_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_ = p_device_h_t_;
      } else {
        // never go here, upper layer is hidden layer, so input layer does not have attention
        p_input_to_hidden_layer_->upper_layer_.p_hidden_layer_->v_nodes_[index_].p_device_h_t_below_ = p_input_to_hidden_layer_->p_attention_layer_->v_nodes_[index_].p_device_final_tmp_2_;
      }
    }
  }

  cudaEventRecord(p_input_to_hidden_layer_->input_hidden_layer_information_.h_t_below_transfer_, p_input_to_hidden_layer_->input_hidden_layer_information_.s00_);
}



}






#endif


