/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef ATTENTION_LAYER_H_
#define ATTENTION_LAYER_H_


#include "deep_rnn.h"
#include "layer_gpu.h"


//#define DEBUG_ATTENTION_LAYER


namespace neural_machine_translation {

template <typename T>
class NeuralMachineTranslation;

template <typename T>
class AttentionLayer;


////////////////////////////////////// AttentionNode //////////////////////////////////////
////////////////////////////////////// AttentionNode //////////////////////////////////////
////////////////////////////////////// AttentionNode //////////////////////////////////////
template <typename T>
class AttentionNode {

public:
  AttentionLayer<T> *p_attention_layer_;

public:
  int lstm_size_;
  int minibatch_size_;
  int device_number_;
  int d_;                     // alignment width
  bool dropout_mode_;
  T dropout_rate_;
  T *p_device_dropout_mask_;

public:
  bool multi_attention_mode_ = false;
  bool multi_attention_v2_mode_ = false;

public:
  // device pointers
  T *p_device_tanh_1_;                   // lstm_size_ x minibatch_size_, tanh_1 = tanh(p_attention_layer_->p_device_w_p_ * p_device_h_t_attention_)
  T *p_device_tanh_1_v2_;
  T *p_device_sigma_1_;                  // 1 x minibatch_size_, sigma_1 = sigmoid(p_attention_layer_->p_device_v_p_ * p_device_tanh_1_)
  T *p_device_sigma_1_v2_;

  T *p_device_p_t_;                      // 1 x minibatch_size_, p_t[i] = p_device_sigma_1_[i] * p_attention_layer_->p_device_batch_information_[i], 
                                         // 0 =< i < minibatch_size
  T *p_device_p_t_v2_;

  T *p_device_alignments_;               // minibatch size x (2 * d_ + 1)
  T *p_device_alignments_v2_;
  T *p_device_h_t_ = NULL;               // lstm_size_ x minibatch_size_, initialized by {top layer of target}.v_nodes_[i].p_device_h_t_
  T *p_device_c_t_;                      // lstm_size_ x minibatch_size_
  T *p_device_c_t_v2_;
  T *p_device_exped_scored_;             // multiply these with a binary mask, so if alignments go off edge then just set to zero
  T *p_device_final_tmp_1_;              // lstm_size_ x minibatch_size_, p_attention_layer_->p_device_w_c_p1_ * p_device_c_t_
  T *p_device_final_tmp_2_;              // lstm_size_ x minibatch_size_, p_attention_layer_->p_device_w_c_p2_ * p_device_h_t_attention_, 
                                         // also reuse to add the bias and tanh

  int *p_device_lower_upper_;            // 2 x minibatch_size_
                                         // p_device_lower_upper_[IDX2C(0,i,2)] is from 0 to p_device_p_t_[i] - d_
                                         // p_device_lower_upper_[IDX2C(1,i,2)] is from p_device_p_t_[i] + d_ to p_attention_layer_->p_device_batch_information_[i] - 1
  int *p_device_lower_upper_v2_;
  int *p_device_indices_;                // minibatch_size_ x (2 * d_ + 1)
  int *p_device_indices_v2_;
  T sigma_sq_;                           // standard deviation, (d_/2.0)^2 = (d_ * d_) / 4.0
  T *p_device_h_t_attention_;            // lstm_size_ x minibatch_size_

public:
  // device pointers
  int **p_device_indices_mask_;           // minibatch_size_ x 1 for node i, points to the lstm node for this information for zeroing out forward and back prop
                                          // p_device_indices_mask_ = &input_layer_target_.v_nodes_[i].p_device_input_vocab_indices_01_ 
                                          //                        =  input_layer_target_.p_device_input_vocab_indices_01_full_ + step 
                                          // (value is 0 0 1 1 1)

  T *p_device_cached_exp_;                // (2 * d_ + 1) x minibatch_size_, stores the exp(-(s - p_t)^2/2*sigma_sq)
  T *p_device_cached_exp_v2_;
  T *p_device_h_t_wa_cache_;              // lstm_size_ x minibatch_size_, precompute h_t multiplied by w_a
  T *p_device_h_t_wa_cache_v2_;


public:
  // device pointers
  T *p_device_hs_mat_;                    // lstm_size_ x (minibatch_size_ * (2 * d_ + 1)) 
  T *p_device_hs_mat_v2_;
  T *p_device_d_errt_ht_tild_;            // this is the error passwd back from the softmax, lstm_size_ x minibatch_size_
                                          // this = v_hidden_layers_target_[layers_number - 2].v_nodes_[i].p_device_d_errt_ht_
                                          //      = SoftmaxLayer::v_nodes_[i].p_device_d_errt_ht_
  T *p_device_d_errt_ht_input_;           // if feed input, then this error will be added in place to p_device_d_errt_ht_p_
  T *p_device_err_above_;                 // what the lstm get passwd from the above layer or softmax

  T *p_device_lower_htild_;               // lstm_size_ x minibatch_size_, for feed input, send htild to this location
                                          // this = p_device_final_tmp_2_
  T *p_device_errt_to_n_htild_below_;     // this is from the previous lower lstm for feed input, lstm_size_ x minibatch_size_
  
public:
  bool feed_input_mode_ = false;          // get rid of most parallelism
  int index_;

public:
  AttentionNode() {}

public:
  void Init(int lstm_size, int minibatch_size, int device_number, int d, bool feed_input_mode, \
            AttentionLayer<T> *p_attention_layer, int index, bool dropout_mode, T dropout_rate, \
            bool multi_attention_mode, bool multi_attention_v2_mode);

public:
  void ForwardProp();

public:
  void BackProp();

public:
  void InitFeedInput(T *p_device_ptr_htild);


};

/*
// CHECK: OK //
template <typename T>
AttentionNode<T>::AttentionNode(int lstm_size, int minibatch_size, int device_number, int d, bool feed_input_mode, \
                                AttentionLayer<T> *p_attention_layer, int index) {

  lstm_size_ = lstm_size;
  minibatch_size_ = minibatch_size;
  device_number_ = device_number;
  d_ = d;
  feed_input_mode_ = feed_input_mode;
  p_attention_layer_ = p_attention_layer;
  index_ = index;

  cudaSetDevice(device_number);

  CudaErrorWrapper(cudaMalloc((void **)&p_device_p_t_, 1 * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_sigma_1_, 1 * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_lower_upper_, 2 * minibatch_size * sizeof(int)), "GPU memory allocation failed\n");

  CudaErrorWrapper(cudaMalloc((void **)&p_device_alignments_, (2 * d + 1) * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_tanh_1_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_hs_mat_, (2 * d + 1) * lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

  CudaErrorWrapper(cudaMalloc((void **)&p_device_c_t_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_indices_, (2 * d + 1) * minibatch_size * sizeof(int)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_final_tmp_1_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

  CudaErrorWrapper(cudaMalloc((void **)&p_device_final_tmp_2_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_h_t_wa_cache_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_cached_exp_, (2 * d + 1) * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

  sigma_sq_ = (d * d) / 4.0;
  index_ = index;
}
*/


template <typename T>
void AttentionNode<T>::Init(int lstm_size, int minibatch_size, int device_number, int d, bool feed_input_mode, \
                            AttentionLayer<T> *p_attention_layer, int index, bool dropout_mode, T dropout_rate, \
                            bool multi_attention_mode, bool multi_attention_v2_mode) {

  lstm_size_ = lstm_size;
  minibatch_size_ = minibatch_size;
  device_number_ = device_number;
  d_ = d;
  feed_input_mode_ = feed_input_mode;
  p_attention_layer_ = p_attention_layer;
  index_ = index;
  dropout_mode_ = dropout_mode;
  dropout_rate_ = dropout_rate;
  multi_attention_mode_ = multi_attention_mode;
  multi_attention_v2_mode_ = multi_attention_v2_mode;

  cudaSetDevice(device_number);
  CudaErrorWrapper(cudaMalloc((void **)&p_device_p_t_, 1 * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_sigma_1_, 1 * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_lower_upper_, 2 * minibatch_size * sizeof(int)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_alignments_, (2 * d + 1) * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

  CudaErrorWrapper(cudaMalloc((void **)&p_device_tanh_1_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_hs_mat_, (2 * d + 1) * lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_c_t_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_indices_, (2 * d + 1) * minibatch_size * sizeof(int)), "GPU memory allocation failed\n");

  CudaErrorWrapper(cudaMalloc((void **)&p_device_final_tmp_1_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_final_tmp_2_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_h_t_wa_cache_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_h_t_attention_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

  CudaErrorWrapper(cudaMalloc((void **)&p_device_cached_exp_, (2 * d + 1) * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

  if (dropout_mode) {
    CudaErrorWrapper(cudaMalloc((void **)&p_device_dropout_mask_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  }

  if (multi_attention_v2_mode) {
    CudaErrorWrapper(cudaMalloc((void**)&p_device_p_t_v2_, 1 * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_sigma_1_v2_, 1 * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_lower_upper_v2_, 2 * minibatch_size * sizeof(int)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_alignments_v2_, (2 * d + 1) * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

    CudaErrorWrapper(cudaMalloc((void**)&p_device_tanh_1_v2_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_hs_mat_v2_, (2 * d + 1) * lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_c_t_v2_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_indices_v2_, (2 * d + 1) * minibatch_size * sizeof(int)), "GPU memory allocation failed\n");

    CudaErrorWrapper(cudaMalloc((void**)&p_device_h_t_wa_cache_v2_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_cached_exp_v2_, (2 * d + 1) * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  }

  sigma_sq_ = (d * d) / 4.0;
  index_ = index;

}


// CHECK: OK //
template <typename T>
void AttentionNode<T>::InitFeedInput(T *p_device_ptr_htild) {
  cudaSetDevice(device_number_);
  p_device_lower_htild_ = p_device_ptr_htild;
  CudaErrorWrapper(cudaMalloc((void **)&p_device_errt_to_n_htild_below_, lstm_size_ * minibatch_size_ * sizeof(T)), "GPU memory allocation failed");
}



template <typename T>
void AttentionNode<T>::ForwardProp() {

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif


  cudaSetDevice(device_number_);
  cudaStreamWaitEvent(p_attention_layer_->attention_layer_gpu_information_.s00_, p_attention_layer_->attention_layer_gpu_information_.start_forward_, 0);

  // dropout, if using dropout
  cudaMemcpyAsync(p_device_h_t_attention_, p_device_h_t_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_attention_layer_->attention_layer_gpu_information_.s00_);

  if (dropout_mode_ && p_attention_layer_->p_neural_mt_->train_mode_) {
    if (!p_attention_layer_->p_neural_mt_->grad_check_flag_) {
      curandSetStream(p_attention_layer_->random_generator_, p_attention_layer_->attention_layer_gpu_information_.s00_);
      CurandGenerateUniformWrapper(p_device_dropout_mask_, lstm_size_ * minibatch_size_, p_attention_layer_->random_generator_);
    }
    DropoutKernel<<<256, 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_dropout_mask_, dropout_rate_, p_device_h_t_attention_, lstm_size_ * minibatch_size_);
  }

  // Event wait on stream zero
  T alpha = 1;
  T beta = 0;

  // stream 0
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // p_device_tanh_1 (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_p_ (lstm_size_ x lstm_size_) * 
  //                                                  p_device_h_t_attention_ (lstm_size_ x minibatch_size)
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_p_, lstm_size_, p_device_h_t_attention_, lstm_size_, &beta, p_device_tanh_1_, lstm_size_), "AttentionNode::ForwardProp p_device_tanh_1_ (p_t) part 1\n");

  // p_device_tanh_1_ (lstm_size_ x minibatch_size_) = tanh(p_device_tanh_1_)  
  TanhKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1)/256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_tanh_1_, p_device_tanh_1_, lstm_size_ * minibatch_size_);
  CudaGetLastError("AttentionNode::ForwardProp p_device_tanh_1_");

  
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // p_device_sigma_1_ (1 x minibatch_size_) = p_attention_layer_->p_device_v_p_ (1 x lstm_size_) *
  //                                           p_device_tanh_1_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, 1, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_v_p_, 1, p_device_tanh_1_, lstm_size_, &beta, p_device_sigma_1_, 1), "AttentionNode::ForwardProp p_device_sigma_1_ (p_t) part 2\n");


  // p_device_sigma_1_ (1 x minibatch_size_) = sigmoid(p_device_sigma_1_) 
  SigmoidKernel<<<std::min(256, (minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_sigma_1_, p_device_sigma_1_, minibatch_size_);
  CudaGetLastError("AttentionNode::ForwardProp sigmoid");


  // Compute p_device_p_t_ for the entire minibatch
  // p_device_p_t_[i] = p_device_sigma_1_[i] * p_attention_layer_->p_device_batch_information_[i], 0 =< i < minibatch_size_ 
  AlignmentPosKernel<<<std::min(256, (minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_sigma_1_, p_device_p_t_, minibatch_size_, p_attention_layer_->p_device_batch_information_);
  CudaGetLastError("AttentionNode::ForwardProp p_device_p_t_");


  // Get the lower and upper ranges for the alignments
  // p_device_lower_upper_ is column major
  // p_device_lower_upper_[IDX2C(0,i,2)] is from 0 to p_device_p_t_[i] - d_
  // p_device_lower_upper_[IDX2C(1,i,2)] is from p_device_p_t_[i] + d_ to p_attention_layer_->p_device_batch_information_[i] - 1
  LowerUpperKernel<<<std::min(256, (2 * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_p_t_, p_device_lower_upper_, d_, p_attention_layer_->p_device_batch_information_, minibatch_size_);
  CudaGetLastError("AttentionNode::ForwardProp p_device_lower_upper_");


  // Get a padding vector, so the scores after exping can be zeroed
  // create p_device_indices_
  // p_device_indices_mask_ = &input_layer_target_.v_nodes_[i].p_device_input_vocab_indices_01_ 
  //                        =  input_layer_target_.p_device_input_vocab_indices_01_full_ (value is 0 0 1 1 1)
  // p_device_indices_ (minibatch_size_ x (2 * d_ + 1))
  CreateIndicesKernel<<<1, 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_indices_, d_, minibatch_size_, p_device_lower_upper_, *p_device_indices_mask_);
  CudaGetLastError("AttentionNode::ForwardProp p_device_indices_");


  // Load in h_s vectors for the minibatch, fill with zeros if off one edge
  // Get all the p_device_hs_mat_ loaded in, also load in the w_a * h_s??????, could be a speedup
  // initialize p_device_hs_mat_ (lstm_size_ x (minibatch_size_ * (2 * d_ + 1))) with p_attention_layer_->p_device_total_hs_mat_
  // p_attention_layer_->p_device_total_hs_mat_[i] = top_source.v_nodes_[i].p_device_h_t_
  LoadInHSKernel<<<std::min(256, (2 * d_ + 1) * minibatch_size_), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_total_hs_mat_, d_, p_device_hs_mat_, p_device_indices_, minibatch_size_, lstm_size_, p_attention_layer_->p_device_batch_information_);
  CudaGetLastError("AttentionNode::ForwardProp load in p_device_hs_mat_");


  // precompute h_t multiplied by w_a
  // stream 0
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);

  // p_device_h_t_wa_cache_ (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_a_^T (lstm_size_ x lstm_size_) * 
  //                                                         p_device_h_t_attention_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_a_, lstm_size_, p_device_h_t_attention_, lstm_size_, &beta, p_device_h_t_wa_cache_, lstm_size_), "AttentionNode::ForwardProp p_device_h_t_wa_cache_ = p_device_w_a_^T * p_device_h_t_attention_\n");


  // do one big reduction for the reduce
  // p_device_alignments_[i] = sum_j(p_device_hs_mat_[j][i] * p_device_h_t_wa_cache_[j][i % minibatch_size_])
  // p_device_alignments_ (minibatch_size * (2 * d_ + 1))
  ElemReduceKernelLarge<<<std::min(minibatch_size_ * (2 * d_ + 1), 256), NUM_ATTENTION_THREADS, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_hs_mat_, p_device_h_t_wa_cache_, p_device_alignments_, lstm_size_, minibatch_size_, d_);


  // normalize the alignments
  // p_device_alignments_ = align(h_t, h_s) * exp(-(p_device_indices_ - p_device_p_t_)^2/2 * sigma_sq_)
  // align(h_t, h_s) = (exp(p_device_alignments_ - max_value)) / sum
  // store exp(-(p_device_indices_ - p_device_p_t_)^2/2 * sigma_sq_) in p_device_cached_exp_ (2 * d_ + 1) x minibatch_size_
  AlignmentReductionKernel<<<1, minibatch_size_, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_alignments_, lstm_size_, minibatch_size_, d_, sigma_sq_, p_device_p_t_, p_device_indices_, p_device_cached_exp_);
  CudaGetLastError("AttentionNode::ForwardProp p_device_alignments_");


  if (unk_replacement_mode__) {
    // find max for each minibatch and store them in the global vector
    DeviceSyncAll();

    GetViterbiAlignmentKernel<<<1, minibatch_size_, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_alignments_, p_device_indices_, d_, minibatch_size_, p_attention_layer_->p_device_viterbi_alignments_);


    DeviceSyncAll();


    thrust::device_ptr<int> thrust_viterbi = thrust::device_pointer_cast(p_attention_layer_->p_device_viterbi_alignments_);
    for (int i = 0; i < minibatch_size_; ++i) {
      viterbi_alignments__[i] = thrust_viterbi[i];
    }


    // now fill in the alignment scores
    // set alignment values to zero
    for (int i = 0; i < alignment_scores__.size(); ++i) {
      alignment_scores__[i] = 0;
    }


    // copy over indices 
    cudaMemcpy(p_host_align_indices__, p_device_indices_, minibatch_size_ * (2 * d_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // copy over alignment values
    cudaMemcpy(p_host_alignment_values__, p_device_alignments_, minibatch_size_ * (2 * d_ + 1) * sizeof(T), cudaMemcpyDeviceToHost);


    for (int i = 0; i < (2 * d_ + 1); ++i) {
      for (int j = 0; j < minibatch_size_; ++j) {
        int curr_index = p_host_align_indices__[IDX2C(j, i, minibatch_size_)];
        
        if(curr_index == -1) {
          continue;
        }

        T curr_val = p_host_alignment_values__[IDX2C(j, i, minibatch_size_)];
        alignment_scores__[IDX2C(curr_index, j, p_attention_layer_->longest_sentence_)] = curr_val;
      }
    }
  }


  // Create the c_t vector
  // p_device_c_t_[i] (lstm_size x minibatch_size) = p_device_alignments[minibatch_index + minibatch_size * j] * 
  //                                              p_device_hs_mat[i + lstm_size * minibatch_size * j]
  CreateCTKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_alignments_, p_device_hs_mat_, p_device_c_t_, lstm_size_, minibatch_size_, d_);
  CudaGetLastError("AttentionNode::ForwardProp p_device_c_t_");


  if (multi_attention_v2_mode_) {
    cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
    CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_p_v2_, lstm_size_, \
                       p_device_h_t_attention_, lstm_size_, &beta, p_device_tanh_1_v2_, lstm_size_), "attention forward p_t part 1\n");

    TanhKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1)/256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_tanh_1_v2_, p_device_tanh_1_v2_, lstm_size_ * minibatch_size_);
    CudaGetLastError("attention tanh1");

    cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
    CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, 1, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_v_p_v2_, 1, \
                       p_device_tanh_1_v2_, lstm_size_, &beta, p_device_sigma_1_v2_, 1), "attention forward p_t part 2\n");
  
    SigmoidKernel<<<std::min(256, (minibatch_size_ + 256 - 1)/256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_sigma_1_v2_, p_device_sigma_1_v2_, minibatch_size_);
    CudaGetLastError("attention sigmoid");

    AlignmentPosKernel<<<std::min(256, (minibatch_size_ + 256 - 1)/256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_sigma_1_v2_, p_device_p_t_v2_, minibatch_size_, p_attention_layer_->p_device_batch_information_v2_);
    CudaGetLastError("attention sigmoid 2");

    LowerUpperKernel<<<std::min(256, (2 * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_p_t_v2_, p_device_lower_upper_v2_, d_, p_attention_layer_->p_device_batch_information_v2_, minibatch_size_);
    CudaGetLastError("attention lower upper");

    CreateIndicesKernel<<<1, 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_indices_v2_, d_, minibatch_size_, p_device_lower_upper_v2_, *p_device_indices_mask_);
    CudaGetLastError("attention create indicies");

    LoadInHSKernel<<<std::min(256, (2 * d_ + 1) * minibatch_size_), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_total_hs_mat_v2_, d_, p_device_hs_mat_v2_, p_device_indices_v2_, minibatch_size_, lstm_size_, p_attention_layer_->p_device_batch_information_v2_);
    CudaGetLastError("attention load in hs");

    cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
    CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_a_v2_, lstm_size_, \
                       p_device_h_t_attention_, lstm_size_, &beta, p_device_h_t_wa_cache_v2_, lstm_size_), "attention forward h_t * w_a\n");

    ElemReduceKernelLarge<<<std::min(minibatch_size_ * (2 * d_ + 1), 256), NUM_ATTENTION_THREADS, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_hs_mat_v2_, p_device_h_t_wa_cache_v2_, p_device_alignments_v2_, lstm_size_, minibatch_size_, d_);

    AlignmentReductionKernel<<<1, minibatch_size_, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_alignments_v2_, lstm_size_, minibatch_size_, d_, sigma_sq_, p_device_p_t_v2_, p_device_indices_v2_, p_device_cached_exp_v2_);
    CudaGetLastError("attention alignment reduction");

    CreateCTKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_alignments_v2_, p_device_hs_mat_v2_, p_device_c_t_v2_, lstm_size_, minibatch_size_, d_);
    CudaGetLastError("attention create ct");
  }


  // stream 0
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // p_device_final_tmp_1_ (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_c_p1_ (lstm_size_ x lstm_size_) *
  //                                                        p_device_c_t_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_c_p1_, lstm_size_, p_device_c_t_, lstm_size_, &beta, p_device_final_tmp_1_, lstm_size_), "AttentionNode::ForwardProp p_device_final_tmp_1_\n");


  // stream 0
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // p_device_final_tmp_2_ (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_c_p2_ (lstm_size_ x lstm_size_) *
  //                                                        p_device_h_t_attention_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_c_p2_, lstm_size_, p_device_h_t_attention_, lstm_size_, &beta, p_device_final_tmp_2_, lstm_size_), "AttentionNode::ForwardProp p_device_final_tmp_2_\n");


  if (multi_attention_v2_mode_) {
    beta = 1;
    cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
    CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_c_p3_v2_, lstm_size_, \
                       p_device_c_t_v2_, lstm_size_, &beta, p_device_final_tmp_2_, lstm_size_), "attention forward p_t part 2\n");
  }


  // Add in the bias and tanh
  // p_device_final_tmp_2_[i] = tanh(p_device_final_tmp_1_[i] + p_device_final_tmp_2_[i] + p_attention_layer_->p_device_output_bias_[i%lstm_size_])
  TanhAttentionForwardKernel<<<std::min(256,(lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_final_tmp_2_, p_device_final_tmp_1_, p_device_final_tmp_2_, p_attention_layer_->p_device_output_bias_, lstm_size_, minibatch_size_);
  CudaGetLastError("AttentionNode::ForwardProp p_device_final_tmp_2_");


  // Zero out cols based on 0 and 1 indices
  // p_device_final_tmp_2_[i] *= *p_device_indices_mask_[i / lstm_size]
  ZeroHT<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_final_tmp_2_, *p_device_indices_mask_, lstm_size_, minibatch_size_);


  // Send h_tild to the lowest level
  // if last index, then there is nothing to copy to
  if (feed_input_mode_ && (p_attention_layer_->longest_sentence_ - 1) != index_ && !multi_attention_mode_) {
    cudaMemcpyAsync(p_device_lower_htild_, p_device_final_tmp_2_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDefault, p_attention_layer_->attention_layer_gpu_information_.s00_);
  }

  cudaEventRecord(p_attention_layer_->attention_layer_gpu_information_.forward_prop_done_, p_attention_layer_->attention_layer_gpu_information_.s00_);


#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

}


template <typename T>
void AttentionNode<T>::BackProp() {

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

  cudaSetDevice(device_number_);
  cudaStreamWaitEvent(p_attention_layer_->attention_layer_gpu_information_.s00_, p_attention_layer_->attention_layer_gpu_information_.start_backward_, 0);
  
  if (feed_input_mode_ && p_attention_layer_->transfer_done_) {
#ifdef REMOVE_STREAMS_FEED_INPUT
    DeviceSyncAll();
#endif

    cudaStreamWaitEvent(p_attention_layer_->attention_layer_gpu_information_.s00_, p_attention_layer_->attention_layer_gpu_information_.error_htild_below_, 0);
    // p_device_d_errt_ht_tild_[i] = p_device_d_errt_ht_tild_[i] + p_device_errt_to_n_htild_below_[i]
    AddTwoMatsKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_d_errt_ht_tild_, p_device_errt_to_n_htild_below_, lstm_size_ * minibatch_size_);
  }

  // for feed input errors, the first error we don't want to add
  p_attention_layer_->transfer_done_ = true;

  T alpha = 1;
  T beta = 1;

  // test this for gradients
  // p_device_d_errt_ht_tild_[i] (lstm_size_ x minibatch_size) *= p_deivce_01_mask[i / lstm_size] (minibatch_size_ x 1)
  // i = (0 -> (lstm_size_ x minibatch_size_ - 1))
  ZeroHT<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_d_errt_ht_tild_, *p_device_indices_mask_, lstm_size_, minibatch_size_);
  CudaGetLastError("AttentionNode::BackProp zero p_device_d_errt_ht_tild_ failed");

  // multiply the gradient coming down by 1 - tanh()^2
  // p_device_d_errt_ht_tild_[i] = p_device_d_errt_ht_tild_[i] * (1 - p_device_final_tmp_2_[i] * p_device_final_tmp_2_[i]
  // tanh(x)' = 1 - tanh(x)^2
  // lstm_size_ x minibatch_size_
  TanhGradKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_d_errt_ht_tild_, p_device_d_errt_ht_tild_, p_device_final_tmp_2_, lstm_size_ * minibatch_size_);
  CudaGetLastError("AttentionNode::BackProp tanh grad p_device_d_errt_ht_tild_");

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // calculate gradient with respect to p_device_w_cp1_
  // p_attention_layer_->p_device_w_c_p1_grad_ (lstm_size_ x lstm_size_) = p_device_d_errt_ht_tild_ (lstm_size_ x minibatch_size_) *
  //                                                                       p_device_c_t_^T (minibatch_size_ x lstm_size_) +
  //                                                                       p_attention_layer_->p_device_w_c_p1_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_device_d_errt_ht_tild_, lstm_size_, p_device_c_t_, lstm_size_, &beta, p_attention_layer_->p_device_w_c_p1_grad_, lstm_size_), "AttentionNode::BackProp p_device_w_c_p1_grad_\n");

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // calculate gradient with respect to p_device_w_cp2_
  // p_attention_layer_->p_device_w_c_p2_grad_ (lstm_size_ x lstm_size_) = p_device_d_errt_ht_tild_ (lstm_size_ x minibatch_size_) * 
  //                                                                       p_device_h_t_attention_^T (minibatch_size_ x lstm_size_) +
  //                                                                       p_attention_layer_->p_device_w_c_p2_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_device_d_errt_ht_tild_, lstm_size_, p_device_h_t_attention_, lstm_size_, &beta, p_attention_layer_->p_device_w_c_p2_grad_, lstm_size_), "AttentionNode::BackProp p_device_w_c_p2_grad_\n");

  // multi_attention_v2 is not written

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // calculate gradient with respect to p_device_output_bias_
  // p_attention_layer_->p_device_output_bias_grad_ (lstm_size_) = p_device_d_errt_ht_tild_ (lstm_size_ x minibatch_size_) *
  //                                                               p_attention_layer_->p_device_ones_minibatch_ (minibatch_size_) +
  //                                                               p_attention_layer_->p_device_output_bias_grad_
  CublasErrorWrapper(CublasGemvWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_device_d_errt_ht_tild_, lstm_size_, p_attention_layer_->p_device_ones_minibatch_, 1, &beta, p_attention_layer_->p_device_output_bias_grad_, 1), "AttentionNode::BackProp p_device_output_bias_grad_ failed\n");

  alpha = 1;
  beta = 0;

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // calculate error with respect to p_device_c_t_
  // p_attention_layer_->p_device_errn_to_t_ct_ (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_c_p1_^T (lstm_size_ x lstm_size_) *
  //                                                                             p_device_d_errt_ht_tild_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_c_p1_, lstm_size_, p_device_d_errt_ht_tild_, lstm_size_, &beta, p_attention_layer_->p_device_errn_to_t_ct_, lstm_size_), "AttentionNode::BackProp p_device_errn_to_t_ct_\n");

  // multi_attention_v2 is not written

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // calculate first part of error with respect to p_device_h_t
  // p_attention_layer_->p_device_errn_to_t_ht_p1_ (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_c_p2_^T (lstm_size_ x lstm_size_) *
  //                                                                                p_device_d_errt_ht_tild_ (lstm_size_ x minibatch_size_)
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_c_p2_, lstm_size_, p_device_d_errt_ht_tild_, lstm_size_, &beta, p_attention_layer_->p_device_errn_to_t_ht_p1_, lstm_size_), "AttentionNode::BackProp p_device_errn_to_t_ht_p1_\n");

  // more efficent version of the code below with less kernel lanches
  // calculate p_attention_layer_->p_device_errn_to_t_as_ ((2 * d_ + 1) x minibatch_size_)
  ErrorAlignmentsKernelLarge<<<std::min(minibatch_size_ * (2 * d_ + 1), 256), NUM_ATTENTION_THREADS, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_errn_to_t_ct_, p_device_hs_mat_, p_attention_layer_->p_device_errn_to_t_as_, lstm_size_, minibatch_size_, d_);
  CudaGetLastError("AttentionNode::BackProp p_device_errn_to_t_as_ failed");

  // get the error for h_s from c_t
  // More efficent version of the code below with less kernel lanches
  ErrorHSAndCTKernelLarge<<<std::min(256, minibatch_size_ * (2 * d_ + 1)), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_errn_to_t_ct_, p_device_alignments_, p_device_indices_, p_attention_layer_->p_device_batch_information_, p_attention_layer_->p_device_total_hs_error_, lstm_size_, minibatch_size_, d_);

  // multi_attention_v2 is not written

  // calculate the error with respect to p_device_p_t_
  ErrorPTKernel<<<minibatch_size_, NUM_ATTENTION_THREADS, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_errn_to_t_pt_, p_attention_layer_->p_device_errn_to_t_as_, d_, sigma_sq_, p_device_indices_, minibatch_size_, p_device_p_t_, p_device_alignments_);
  CudaGetLastError("AttentionNode::BackProp p_device_errn_to_t_pt_");

  // calculate the error with respect to p_device_v_p_
  AttentionVPErrorKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_sigma_1_, p_device_tanh_1_, p_attention_layer_->p_device_tmp_1_, p_attention_layer_->p_device_errn_to_t_pt_, p_attention_layer_->p_device_batch_information_, lstm_size_, minibatch_size_);
  CudaGetLastError("AttentionNode::BackProp p_device_tmp_1_");

  // multi_attention_v2 is not written

  alpha = 1;
  beta = 1;

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // calculate the gradient with respect to p_device_v_p_
  // p_attention_layer_->p_device_v_p_grad_ (lstm_size_ x 1) = p_attention_layer_->p_device_tmp_1_ (lstm_size_ x minibatch_size_) *
  //                                                           p_attention_layer_->p_device_ones_minibatch_ (minibatch_size_) + 
  //                                                           p_attention_layer_->p_device_v_p_grad_
  CublasErrorWrapper(CublasGemvWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, lstm_size_, minibatch_size_, &alpha, p_attention_layer_->p_device_tmp_1_, lstm_size_, p_attention_layer_->p_device_ones_minibatch_, 1, &beta, p_attention_layer_->p_device_v_p_grad_, 1), "AttentionNode::BackProp p_device_v_p_grad_ failed\n");

  // calculate the error with respect to p_device_w_p_
  // p_attention_layer_->p_device_tmp_1_[i] = p_attention_layer_->p_device_errn_to_t_pt_[minibatch_index] * 
  //                                          p_attention_layer_->p_device_batch_information_[minibatch_index] * 
  //                                          p_attention_layer_->p_device_v_p_[lstm_index] * 
  //                                          p_device_sigma_1_[minibatch_index] * (1 - p_device_sigma_1_[minibatch_index]) * 
  //                                          (1 - p_device_tanh_1_[i] * p_device_tanh_1_[i])
  GradWPKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_v_p_, p_attention_layer_->p_device_tmp_1_, p_device_sigma_1_, p_device_tanh_1_, p_attention_layer_->p_device_errn_to_t_pt_, p_attention_layer_->p_device_batch_information_, lstm_size_, minibatch_size_);
  CudaGetLastError("AttentionNode::BackProp p_device_tmp_1_");

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // finish the gradient calculation of p_device_w_p_ with outer product
  // p_attention_layer_->p_device_w_p_grad_ (lstm_size_ x lstm_size_) = p_attention_layer_->p_device_tmp_1_ (lstm_size_ x minibatch_size_) *
  //                                                                    p_device_h_t_attention_ (minibatch_size_ x lstm_size_) + 
  //                                                                    p_attention_layer_->p_device_w_p_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_attention_layer_->p_device_tmp_1_, lstm_size_, p_device_h_t_attention_, lstm_size_, &beta, p_attention_layer_->p_device_w_p_grad_, lstm_size_), "AttentionNode::BackProp p_device_w_p_grad_\n");

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // now get the second part of the error with respect to p_device_h_t_ and add it to the first part
  // stuff is already stored in the above tmp matrix
  // p_attention_layer_->p_device_errn_to_t_ht_p1_ (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_p_^T (lstm_size_ x lstm_size_) *
  //                                                                                p_attention_layer_->p_device_tmp_1_ (lstm_size_ x minibatch_size_) +
  //                                                                                + p_attention_layer_->p_device_errn_to_t_ht_p1_
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_T, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_p_, lstm_size_, p_attention_layer_->p_device_tmp_1_, lstm_size_, &beta, p_attention_layer_->p_device_errn_to_t_ht_p1_, lstm_size_), "AttentionNode::BackProp p_device_errn_to_t_ht_p1_\n");

  // multi_attention_v2 is not written

  cudaMemsetAsync(p_attention_layer_->p_device_h_s_sum_, 0, lstm_size_ * minibatch_size_ * sizeof(T), p_attention_layer_->attention_layer_gpu_information_.s00_);

  // multi_attention_v2 is not written

  alpha = 1;
  beta = 1;
  // get p_attention_layer_->p_device_h_t_wa_factor_
  GetHTScalingsWaGradKernel<<<std::min(256, ((2 * d_ + 1) * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_h_t_wa_factor_, p_attention_layer_->p_device_errn_to_t_as_, p_device_alignments_, p_device_cached_exp_, d_, minibatch_size_);
  CudaGetLastError("AttentionNode::BackProp p_device_h_t_wa_factor_");

  for (int i = 0; i < 2 * d_ + 1; ++i) {
    // for w_a gradient
    // p_attention_layer_->p_device_tmp_1_[i] = (p_device_hs_mat_ + i * (lstm_size_ * minibatch_size_))[i] * 
    //                                          p_attention_layer_->p_device_h_t_wa_factor_[alignment_index][minibatch_index]
    ScaleHTKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_h_t_wa_factor_, p_attention_layer_->p_device_tmp_1_, p_device_hs_mat_ + i * (lstm_size_ * minibatch_size_), lstm_size_, minibatch_size_, i, d_);
    CudaGetLastError("AttentionNode::BackProp p_device_tmp_1_");

    // p_attention_layer_->p_device_h_s_sum_[i] = p_attention_layer_->p_device_h_s_sum_[i] + p_attention_layer_->p_device_tmp_1_[i]
    AddTwoMatsKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_h_s_sum_, p_attention_layer_->p_device_tmp_1_, lstm_size_ * minibatch_size_);
    CudaGetLastError("AttentionNode::BackProp p_device_h_s_sum_");

    // for h_t and h_s gradient
    beta = 0;
    // p_attention_layer_->p_device_tmp_1_[i] = p_device_h_t_wa_cache_[i] * 
    //                                          p_attention_layer_->p_device_h_t_wa_factor_[alignment_index][minibatch_index]
    ScaleHTKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_h_t_wa_factor_, p_attention_layer_->p_device_tmp_1_, p_device_h_t_wa_cache_, lstm_size_, minibatch_size_, i, d_);
    CudaGetLastError("AttentionNode::BackProp p_device_tmp_1_");

    // copy errors to p_attention_layer_->p_device_total_hs_error_
    CopyErrorsSource<<<std::min(256, minibatch_size_), 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_attention_layer_->p_device_total_hs_error_, p_attention_layer_->p_device_tmp_1_, p_device_indices_, lstm_size_, minibatch_size_, d_, i, p_attention_layer_->p_device_batch_information_);
    CudaGetLastError("AttentionNode::BackProp p_device_total_hs_error_");
  }

  // multi_attention_v2 is not written

  beta = 1;
  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // calculate p_device_w_a_grad_
  // p_attention_layer_->p_device_w_a_grad_ (lstm_size_ x lstm_size_) = p_device_h_t_attention_ (lstm_size_ x minibatch_size_) *
  //                                                                    p_attention_layer_->p_device_h_s_sum_^T (minibatch_size_ x lstm_size_) +
  //                                                                    p_attention_layer_->p_device_w_a_grad_
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_T, lstm_size_, lstm_size_, minibatch_size_, &alpha, p_device_h_t_attention_, lstm_size_, p_attention_layer_->p_device_h_s_sum_, lstm_size_, &beta, p_attention_layer_->p_device_w_a_grad_, lstm_size_), "AttentionNode::BackProp p_device_w_a_grad_\n");

  // stream 00
  cublasSetStream(p_attention_layer_->handle_, p_attention_layer_->attention_layer_gpu_information_.s00_);
  // p_attention_layer_->p_device_errn_to_t_ht_p1_ (lstm_size_ x minibatch_size_) = p_attention_layer_->p_device_w_a_ (lstm_size_ x lstm_size_) *
  //                                                                                p_attention_layer_->p_device_h_s_sum_ (lstm_size_ x minibatch_size_) +
  //                                                                                p_attention_layer_->p_device_errn_to_t_ht_p1_
  CublasErrorWrapper(CublasGemmWrapper(p_attention_layer_->handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_attention_layer_->p_device_w_a_, lstm_size_, p_attention_layer_->p_device_h_s_sum_, lstm_size_, &beta, p_attention_layer_->p_device_errn_to_t_ht_p1_, lstm_size_), "AttentionNode::BackProp p_device_errn_to_t_ht_p1_\n");

  // multi_attention_v2 is not written

  if (dropout_mode_) {
    // dropout for p_device_errn_to_t_ht_p1_
    DropoutKernel<<<256, 256, 0, p_attention_layer_->attention_layer_gpu_information_.s00_>>>(p_device_dropout_mask_, dropout_rate_, p_attention_layer_->p_device_errn_to_t_ht_p1_, lstm_size_ * minibatch_size_);
  }

  // reset p_device_d_errt_ht_tild_ to p_attention_layer_->p_device_errn_to_t_ht_p1_
  cudaMemcpyAsync(p_device_d_errt_ht_tild_, p_attention_layer_->p_device_errn_to_t_ht_p1_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice, p_attention_layer_->attention_layer_gpu_information_.s00_);
  CudaGetLastError("AttentionNode::BackProp p_device_d_errt_ht_tild_");

  cudaEventRecord(p_attention_layer_->attention_layer_gpu_information_.backward_prop_done_, p_attention_layer_->attention_layer_gpu_information_.s00_);

#ifdef REMOVE_STREAMS
  DeviceSyncAll();
#endif

}



////////////////////////////////////// AttentionLayer //////////////////////////////////////
////////////////////////////////////// AttentionLayer //////////////////////////////////////
////////////////////////////////////// AttentionLayer //////////////////////////////////////
template <typename T>
class AttentionLayer {

public:
  cublasHandle_t handle_;
  int device_number_;
  int lstm_size_;
  int minibatch_size_;
  bool clip_gradient_mode_;        // for gradient clipping
  T norm_clip_;
  bool feed_input_mode_ = false;
  int longest_sentence_;
  bool transfer_done_ = false;     // if true then take the copied matrix
  bool multi_attention_v2_mode_ = false;
  
public:
  // device pointers
  T *p_device_w_a_;                // lstm_size_ x lstm_size_, for the score function
  T *p_device_v_p_;                // 1 x lstm_size_
  T *p_device_w_p_;                // lstm_size_ x lstm_size_
  T *p_device_w_c_p1_;             // lstm_size_ x lstm_size_,
  T *p_device_w_c_p2_;             // lstm_size_ x lstm_size_, 
  T *p_device_output_bias_;        // lstm_size_ x 1

  T *p_device_w_a_v2_;
  T *p_device_w_p_v2_;
  T *p_device_v_p_v2_;
  T *p_device_w_c_p3_v2_;
  T *p_device_tmp_1_v2_;
  T *p_device_h_t_wa_factor_v2_;
  T *p_device_h_t_sum_v2_;
  T *p_device_h_s_sum_v2_;
  

  int *p_device_viterbi_alignments_; // minibatch_size, for decoding unk replacement

  T *p_device_w_a_grad_;
  T *p_device_v_p_grad_;                        // lstm_size_
  T *p_device_w_p_grad_;                        // lstm_size_ x lstm_size_
  T *p_device_w_c_p1_grad_;                     // lstm_size_ x lstm_size_
  T *p_device_w_c_p2_grad_;                     // lstm_size_ x lstm_size_
  T *p_device_output_bias_grad_;                // lstm_size_


  thrust::device_ptr<T> p_thrust_device_w_a_grad_;
  thrust::device_ptr<T> p_thrust_device_v_p_grad_;
  thrust::device_ptr<T> p_thrust_device_w_p_grad_;
  thrust::device_ptr<T> p_thrust_device_w_c_p1_grad_;
  thrust::device_ptr<T> p_thrust_device_w_c_p2_grad_;
  thrust::device_ptr<T> p_thrust_device_output_bias_grad_;

public:
  // for gradient clipping
  T *p_device_result_;
  T *p_device_result_tmp_;
  
public:
  // device pointers
  T *p_device_errn_to_t_ht_p1_;              // lstm_size_ x minibatch_size_
  T *p_device_errn_to_t_tan_htild_;
  T *p_device_errn_to_t_ct_;                 // lstm_size_ x minibatch_size_

  T **p_device_total_hs_mat_;                // longest_sent_ x (lstm_size_ x minibatch_size_), hs_mat[i] = top_source.v_nodes_[i].p_device_h_t_
  T **p_device_total_hs_mat_v2_;
  T **p_device_total_hs_error_;              // longest_sent_ x (lstm_size_ x minibatch_size_), hs_error[i] = top_source.v_nodes_[i].p_device_d_errt_ht_

  T *p_device_ones_minibatch_;               // minibatch_size_, init by all 1

  T *p_device_errn_to_t_as_;            // (2 * d_ + 1) x minibatch_size_
  T *p_device_errn_to_t_pt_;            // minibatch_size_

  T *p_device_tmp_1_;                   // lstm_size_ x minibatch_size_, lstm by minibatch size
  T *p_device_tmp_wa_grad_;
  T *p_device_h_t_wa_factor_;           // (2*d_ + 1) x minibatch_size_, for w_a gradient
  T *p_device_h_t_sum_;                 // for summing weighted h_t's
  T *p_device_h_s_sum_;                 // lstm_size_ x minibatch_size_, for summing weighted h_s

  T *p_device_errn_to_t_htild_below_;   // from the input layer

public:
  AttentionLayerGpuInformation attention_layer_gpu_information_;     // stores the gpu information for the attention model
  curandGenerator_t random_generator_;

public:
  int *p_device_batch_information_;                                  // minibatch_size_ * 2, length of minibatches, then offsets
  int *p_device_batch_information_v2_;

public:
  int *p_device_ones_minibatch_int_;
  
public:
  std::vector<AttentionNode<T>> v_nodes_;

public:
  NeuralMachineTranslation<T> *p_neural_mt_;
  
public:
  AttentionLayer() {};

public:
  AttentionLayer(int lstm_size, int minibatch_size, int device_number, int d, int longest_sentence, cublasHandle_t &handle, \
                 NeuralMachineTranslation<T> *p_neural_mt, bool feed_input_mode, bool clip_gradient_mode, T norm_clip, \
                 bool dropout_mode, T dropout_rate, GlobalConfiguration &config, bool bi_side_mode);   // constructor

public:
  void CheckGradients(T epsilon);
  void CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols);

public:
  void ClearGradients();

public:
  void PreprocessMinibatchInformation(int *p_host_batch_info);

public:
  void DumpWeights(std::ofstream &output);

public:
  void LoadWeights(std::ifstream &input_stream);

public:
  void ClipGradientsFunc();

public:
  void ScaleGradients();

public:
  void UpdateParams();

public:
  void NormP1();
  void NormP2();

public:
  void ClipIndividual();

public:
  void InitAttentionDecoder(int lstm_size, int beam_size, int device_number, int d, int longest_sentence, cublasHandle_t &handle, \
                            NeuralMachineTranslation<T> *p_neural_mt, bool feed_input_mode, std::vector<T*> &v_top_source_states, \
                            bool multi_attention_v2, std::vector<T*> &v_top_source_states_v2);
};


// CHECK: OK //
template <typename T>
AttentionLayer<T>::AttentionLayer(int lstm_size, int minibatch_size, int device_number, int d, int longest_sentence, cublasHandle_t &handle, \
                                  NeuralMachineTranslation<T> *p_neural_mt, bool feed_input_mode, bool gradient_clip_mode, T norm_clip_threshold, \
                                  bool dropout_mode, T dropout_rate, GlobalConfiguration &config, bool bi_side_mode) {

#ifdef DEBUG_DROPOUT
  std::cerr<<"\n************CP3 In *AttentionLayer* *Constructor*\n"
           <<"   lstm_size: "<<lstm_size<<"\n"
           <<"   minibatch_size: "<<minibatch_size<<"\n"
           <<"   device_number: "<<device_number<<"\n"
           <<"   d: "<<d<<"\n"
           <<"   longest_sentence: "<<longest_sentence<<"\n"
           <<"   feed_input_mode: "<<feed_input_mode<<"\n"
           <<"   gradient_clip_mode: "<<gradient_clip_mode<<"\n"
           <<"   norm_clip_threshold: "<<norm_clip_threshold<<"\n"
           <<"   dropout_mode: "<<dropout_mode<<"\n"
           <<"   dropout_rate: "<<dropout_rate<<"\n"
           <<"   bi_side_mode: "<<bi_side_mode<<"\n"
           <<"\n"<<std::flush;
#endif

  handle_ = handle;
  p_neural_mt_ = p_neural_mt;
  device_number_ = device_number;
  lstm_size_ = lstm_size;
  minibatch_size_ = minibatch_size;
  clip_gradient_mode_ = gradient_clip_mode;
  norm_clip_ = norm_clip_threshold;
  feed_input_mode_ = feed_input_mode;
  longest_sentence_ = longest_sentence;

  cudaSetDevice(device_number);
  attention_layer_gpu_information_.Init(device_number, d);

  T *p_host_tmp;
  FullMatrixSetup(&p_host_tmp, &p_device_w_a_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_w_p_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_v_p_, 1, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_output_bias_, lstm_size, 1);
  FullMatrixSetup(&p_host_tmp, &p_device_w_c_p1_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_w_c_p2_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_tmp, &p_device_w_a_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_w_p_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_v_p_grad_, 1, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_output_bias_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_tmp, &p_device_w_c_p1_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_w_c_p2_grad_, lstm_size, lstm_size);

  // multi_attention_v2 is not written

  FullMatrixSetup(&p_host_tmp, &p_device_errn_to_t_tan_htild_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp, &p_device_errn_to_t_ct_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp, &p_device_errn_to_t_ht_p1_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp, &p_device_errn_to_t_as_, 2 * d + 1, minibatch_size);
  FullMatrixSetup(&p_host_tmp, &p_device_errn_to_t_pt_, 1, minibatch_size);

  FullMatrixSetup(&p_host_tmp, &p_device_tmp_1_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_tmp, &p_device_h_t_wa_factor_, 2 * d + 1, minibatch_size);

  // multi_attention_v2 is not written

  FullVectorSetupOnes(&p_host_tmp, &p_device_ones_minibatch_, minibatch_size);

  curandCreateGenerator(&random_generator_, CURAND_RNG_PSEUDO_DEFAULT);
  boost::uniform_int<> unif_boost(1, 1000000);
  curandSetPseudoRandomGeneratorSeed(random_generator_, curr_seed__);
  curr_seed__ += 7;

  p_thrust_device_w_a_grad_ = thrust::device_pointer_cast(p_device_w_a_grad_);
  p_thrust_device_v_p_grad_ = thrust::device_pointer_cast(p_device_v_p_grad_);
  p_thrust_device_w_p_grad_ = thrust::device_pointer_cast(p_device_w_p_grad_);
  p_thrust_device_w_c_p1_grad_ = thrust::device_pointer_cast(p_device_w_c_p1_grad_);
  p_thrust_device_w_c_p2_grad_ = thrust::device_pointer_cast(p_device_w_c_p2_grad_);
  p_thrust_device_output_bias_grad_ = thrust::device_pointer_cast(p_device_output_bias_grad_);

  // multi_attention_v2 is not written


  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_, 1 * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_tmp_, NORM_THREADS * sizeof(T)), "GPU memory allocation failed\n");

  ClearGradients();

  CudaErrorWrapper(cudaMalloc((void **)&p_device_h_t_sum_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_h_s_sum_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed\n");

  // multi_attention_v2 is not written

  for (int i = 0; i < longest_sentence; ++i) {
    AttentionNode<T> attention_node;
    attention_node.Init(lstm_size, minibatch_size, device_number, d, feed_input_mode, this, i, dropout_mode, dropout_rate, \
                        config.multi_source_params_.multi_attention_mode_, multi_attention_v2_mode_);
    v_nodes_.push_back(attention_node);
  }

  // Now construct p_device_total_hs_mat_
  T **p_host_total_hs_mat = (T **)malloc(longest_sentence * sizeof(T*));
  T **p_host_total_hs_error = (T **)malloc(longest_sentence * sizeof(T*));

  for (int i = 0; i < longest_sentence; ++i) {
    
    // bi_dir is not written

    if (p_neural_mt->v_hidden_layers_source_.size() == 0) {
      p_host_total_hs_mat[i] = p_neural_mt->input_layer_source_.v_nodes_[i].p_device_h_t_;
      p_host_total_hs_error[i] = p_neural_mt->input_layer_source_.v_nodes_[i].p_device_d_errt_ht_;
    } else {
      p_host_total_hs_mat[i] = p_neural_mt->v_hidden_layers_source_[p_neural_mt->v_hidden_layers_source_.size() - 1].v_nodes_[i].p_device_h_t_;
      p_host_total_hs_error[i] = p_neural_mt->v_hidden_layers_source_[p_neural_mt->v_hidden_layers_source_.size() - 1].v_nodes_[i].p_device_d_errt_ht_;
    }
  }

  CudaErrorWrapper(cudaMalloc((void **)&p_device_total_hs_mat_, longest_sentence * sizeof(T*)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_total_hs_error_, longest_sentence * sizeof(T*)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_batch_information_, 2 * minibatch_size * sizeof(int)), "GPU memory allocation failed\n");

  cudaMemcpy(p_device_total_hs_mat_, p_host_total_hs_mat, longest_sentence * sizeof(T*), cudaMemcpyHostToDevice);
  cudaMemcpy(p_device_total_hs_error_, p_host_total_hs_error, longest_sentence * sizeof(T*), cudaMemcpyHostToDevice);

  free(p_host_total_hs_mat);

  // multi_attention_v2 is not written
}


// CHECK: OK //
template <typename T>
void AttentionLayer<T>::PreprocessMinibatchInformation(int *p_host_batch_info) {
  cudaSetDevice(device_number_);
  cudaMemcpy(p_device_batch_information_, p_host_batch_info, 2 * minibatch_size_ * sizeof(int), cudaMemcpyHostToDevice);
}


template <typename T>
void AttentionLayer<T>::CheckGradients(T epsilon) {
  std::cerr<<" #### GRADIENT CHECKING FOR ATTENTION LAYER GPU ####\n";
  std::cerr<<" GRADIENT CHECKING FOR p_device_w_c_p1_\n";
  CheckGradientGpu(epsilon, p_device_w_c_p1_, p_device_w_c_p1_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_w_c_p2_\n";
  CheckGradientGpu(epsilon, p_device_w_c_p2_, p_device_w_c_p2_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR OUTPUT BIAS\n";
  CheckGradientGpu(epsilon, p_device_output_bias_, p_device_output_bias_grad_, lstm_size_, 1);

  std::cerr<<" GRADIENT CHECKING FOR p_device_v_p_\n";
  CheckGradientGpu(epsilon, p_device_v_p_, p_device_v_p_grad_, lstm_size_, 1);

  std::cerr<<" GRADIENT CHECKING FOR p_device_w_p_\n";
  CheckGradientGpu(epsilon, p_device_w_p_, p_device_w_p_grad_, lstm_size_, lstm_size_);

  std::cerr<<" GRADIENT CHECKING FOR p_device_w_a_\n";
  CheckGradientGpu(epsilon, p_device_w_a_, p_device_w_a_grad_, lstm_size_, lstm_size_);

  return;
}

template <typename T>
void AttentionLayer<T>::CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols) {
  cudaSetDevice(device_number_);
  thrust::device_ptr<T> p_thrust_device_mat = thrust::device_pointer_cast(p_device_mat);
  thrust::device_ptr<T> p_thrust_device_grad = thrust::device_pointer_cast(p_device_grad);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      T loss = 0;
      p_thrust_device_mat[IDX2C(i, j, rows)] += epsilon;
      loss = p_neural_mt_->GetError(true);
      cudaSetDevice(device_number_);
      cudaDeviceSynchronize();

      p_thrust_device_mat[IDX2C(i, j, rows)] += -2 * epsilon;
      loss -= p_neural_mt_->GetError(true);
      cudaSetDevice(device_number_);
      cudaDeviceSynchronize();

      p_thrust_device_mat[IDX2C(i, j, rows)] += epsilon;
      std::cerr<<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"  my gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n";
      if ((std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))) > 1 / (T)1000.0 || (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))) > 1 / 1000.0) {
        std::cerr<<"Gradient for gradient check: "<<loss / (2 * epsilon)<<"\n";
        std::cerr<<"My gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n";
        std::cerr<<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"\n";
        std::cerr<<"Gradient difference (Equation 2): "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))<<"\n\n";
      } else if (0 == p_thrust_device_grad[IDX2C(i, j, rows)] || 0 == loss / (2 * epsilon)) {
        std::cerr<<"ZERO GRADIENTS\n";
        std::cerr<<"Gradient for gradient check: "<<loss / (2 * epsilon)<<"\n";
        std::cerr<<"My gradient: "<<p_thrust_device_grad[IDX2C(i, j, rows)]<<"\n";
        std::cerr<<"Gradient difference: "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon))<<"\n";
        std::cerr<<"Gradient difference (Equation 2): "<<std::abs(p_thrust_device_grad[IDX2C(i, j, rows)] - loss / (2 * epsilon)) / (std::abs(p_thrust_device_grad[IDX2C(i, j, rows)]) + std::abs(loss / (2 * epsilon)))<<"\n\n";
      }
    }
  }
  return;
}


// CHECK DECODER: OK //
// CHECK: OK //
template <typename T>
void AttentionLayer<T>::ClearGradients() {
  cudaSetDevice(device_number_);
  cudaMemsetAsync(p_device_w_a_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), attention_layer_gpu_information_.s00_);
  cudaMemsetAsync(p_device_w_p_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), attention_layer_gpu_information_.s00_);
  cudaMemsetAsync(p_device_v_p_grad_, 0, lstm_size_ * 1 * sizeof(T), attention_layer_gpu_information_.s00_);
  cudaMemsetAsync(p_device_output_bias_grad_, 0, lstm_size_ * 1 * sizeof(T), attention_layer_gpu_information_.s00_);
  cudaMemsetAsync(p_device_w_c_p1_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), attention_layer_gpu_information_.s00_);
  cudaMemsetAsync(p_device_w_c_p2_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), attention_layer_gpu_information_.s00_);

  // multi_attention_v2 is not written //

  DeviceSyncAll();
}

template <typename T>
void AttentionLayer<T>::ScaleGradients() {
  ScaleFunctor unary_op(minibatch_size_);
  thrust::for_each(p_thrust_device_w_a_grad_, p_thrust_device_w_a_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_w_p_grad_, p_thrust_device_w_p_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_v_p_grad_, p_thrust_device_v_p_grad_ + lstm_size_ * 1, unary_op);
  thrust::for_each(p_thrust_device_output_bias_grad_, p_thrust_device_output_bias_grad_ + lstm_size_ * 1, unary_op);
  thrust::for_each(p_thrust_device_w_c_p1_grad_, p_thrust_device_w_c_p1_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_w_c_p2_grad_, p_thrust_device_w_c_p2_grad_ + lstm_size_ * lstm_size_, unary_op);

  // multi_attention_v2 is not written
}

template <typename T>
void AttentionLayer<T>::UpdateParams() {
  // p_device_w_a_ += p_neural_mt_->input_layer_target_.learning_rate_ * p_device_w_a_grad_
  GradientUpdateMats<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_a_, p_device_w_a_grad_, p_neural_mt_->input_layer_target_.learning_rate_, lstm_size_ * lstm_size_);

  // p_device_w_p_ += p_neural_mt_->input_layer_target_.learning_rate_ * p_device_w_p_grad_
  GradientUpdateMats<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_p_, p_device_w_p_grad_, p_neural_mt_->input_layer_target_.learning_rate_, lstm_size_ * lstm_size_);

  // p_device_v_p_ += p_neural_mt_->input_layer_target_.learning_rate_ * p_device_v_p_grad_
  GradientUpdateMats<<<std::min(256, (lstm_size_ * 1 + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_v_p_, p_device_v_p_grad_, p_neural_mt_->input_layer_target_.learning_rate_, lstm_size_ * 1);

  // p_device_output_bias_ += p_neural_mt_->input_layer_target_.learning_rate_ * p_device_output_bias_grad_
  GradientUpdateMats<<<std::min(256, (lstm_size_ * 1 + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_output_bias_, p_device_output_bias_grad_, p_neural_mt_->input_layer_target_.learning_rate_, lstm_size_ * 1);

  // p_device_w_c_p1_ += p_neural_mt_->input_layer_target_.learning_rate_ * p_device_w_c_p1_grad_
  GradientUpdateMats<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_c_p1_, p_device_w_c_p1_grad_, p_neural_mt_->input_layer_target_.learning_rate_, lstm_size_ * lstm_size_);

  // p_device_w_c_p2_ += p_neural_mt_->input_layer_target_.learning_rate_ * p_device_w_c_p2_grad_
  GradientUpdateMats<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_c_p2_, p_device_w_c_p2_grad_, p_neural_mt_->input_layer_target_.learning_rate_, lstm_size_ * lstm_size_);

  // multi_attention_v2 is not written
}


template <typename T>
void AttentionLayer<T>::NormP1() {
  NormClipGpuV2P1(p_thrust_device_w_a_grad_, p_device_w_a_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_w_p_grad_, p_device_w_p_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_v_p_grad_, p_device_v_p_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_output_bias_grad_, p_device_output_bias_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_w_c_p1_grad_, p_device_w_c_p1_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P1(p_thrust_device_w_c_p2_grad_, p_device_w_c_p2_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  // multi_attention_v2 is not written
}

template <typename T>
void AttentionLayer<T>::NormP2() {
  NormClipGpuV2P2(p_thrust_device_w_a_grad_, p_device_w_a_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_p_grad_, p_device_w_p_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_v_p_grad_, p_device_v_p_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_output_bias_grad_, p_device_output_bias_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_c_p1_grad_, p_device_w_c_p1_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_c_p2_grad_, p_device_w_c_p2_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  // multi_attention_v2 is not written
}


template <typename T>
void AttentionLayer<T>::ClipIndividual() {
  ClipMatKernel<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_a_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
  ClipMatKernel<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_p_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
  ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_v_p_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
  ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_output_bias_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
  ClipMatKernel<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_c_p1_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
  ClipMatKernel<<<std::min(256, (lstm_size_ * lstm_size_ + 256 - 1) / 256), 256, 0, attention_layer_gpu_information_.s00_>>>(p_device_w_c_p2_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);

  // multi_attention_v2 is not written

  DeviceSyncAll();
}


template <typename T>
void AttentionLayer<T>::InitAttentionDecoder(int lstm_size, int beam_size, int device_number, int d, int longest_sentence, \
                                             cublasHandle_t &handle, NeuralMachineTranslation<T> *p_neural_mt, bool feed_input_mode, \
                                             std::vector<T*> &v_top_source_states, bool multi_attention_v2_mode, \
                                             std::vector<T*> &v_top_source_states_v2) {

  handle_ = handle;
  p_neural_mt_ = p_neural_mt;
  device_number_ = device_number;
  lstm_size_ = lstm_size;
  minibatch_size_ = beam_size;
  longest_sentence_ = longest_sentence;
  multi_attention_v2_mode_ = multi_attention_v2_mode;

  cudaSetDevice(device_number);
  attention_layer_gpu_information_.Init(device_number, d);

  T *p_host_tmp;
  FullMatrixSetup(&p_host_tmp, &p_device_w_a_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_w_p_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_v_p_, 1, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_output_bias_, lstm_size, 1);
  FullMatrixSetup(&p_host_tmp, &p_device_w_c_p1_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_w_c_p2_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_1_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_h_t_wa_factor_, 2 * d + 1, minibatch_size_);
  FullVectorSetupOnes(&p_host_tmp, &p_device_ones_minibatch_, minibatch_size_);

  if (multi_attention_v2_mode) {
    FullMatrixSetup(&p_host_tmp, &p_device_w_a_v2_, lstm_size, lstm_size);
    FullMatrixSetup(&p_host_tmp, &p_device_w_p_v2_, lstm_size, lstm_size);
    FullMatrixSetup(&p_host_tmp, &p_device_v_p_v2_, 1, lstm_size);
    FullMatrixSetup(&p_host_tmp, &p_device_w_c_p3_v2_, lstm_size, lstm_size);
    FullMatrixSetup(&p_host_tmp, &p_device_tmp_1_v2_, lstm_size, minibatch_size_);
    FullMatrixSetup(&p_host_tmp, &p_device_h_t_wa_factor_v2_, 2 * d + 1, minibatch_size_);
    CudaErrorWrapper(cudaMalloc((void**)&p_device_h_t_sum_v2_, lstm_size * minibatch_size_ * sizeof(T)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void**)&p_device_h_s_sum_v2_, lstm_size * minibatch_size_ * sizeof(T)), "GPU memory allocation failed\n");

  }

  CudaErrorWrapper(cudaMalloc((void **)&p_device_h_t_sum_, lstm_size * minibatch_size_ * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_h_s_sum_, lstm_size * minibatch_size_ * sizeof(T)), "GPU memory allocation failed\n");

  for (int i = 0; i < 1; ++i) {
    AttentionNode<T> attention_node;
    attention_node.Init(lstm_size, minibatch_size_, device_number, d, false, this, i, false, 1, false, multi_attention_v2_mode);
    v_nodes_.push_back(attention_node);
  }

  // now construct p_device_total_hs_mat_
  T **p_host_total_hs_mat = (T **)malloc(longest_sentence * sizeof(T *));
  for (int i = 0; i < longest_sentence; ++i) {
    p_host_total_hs_mat[i] = v_top_source_states[i];
  }

  T **p_host_total_hs_mat_v2;
  if (multi_attention_v2_mode) {
    p_host_total_hs_mat_v2 = (T **)malloc(longest_sentence * sizeof(T *));
    for (int i = 0; i < longest_sentence; ++i) {
      p_host_total_hs_mat_v2[i] = v_top_source_states_v2[i];
    }
  }

  CudaErrorWrapper(cudaMalloc((void **)&p_device_ones_minibatch_int_, minibatch_size_ * sizeof(int)), "GPU memory allocation failed\n");
  thrust::device_ptr<int> ones_ptr = thrust::device_pointer_cast(p_device_ones_minibatch_int_);
  for (int i = 0; i < minibatch_size_; ++i) {
    ones_ptr[i] = 1;
  }

  v_nodes_[0].p_device_indices_mask_ = &p_device_ones_minibatch_int_;

  CudaErrorWrapper(cudaMalloc((void **)&p_device_total_hs_mat_, longest_sentence * sizeof(T *)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_batch_information_, 2 * minibatch_size_ * sizeof(int)), "GPU memory allocation failed\n");

  cudaMemcpy(p_device_total_hs_mat_, p_host_total_hs_mat, longest_sentence * sizeof(T *), cudaMemcpyHostToDevice);

  if (multi_attention_v2_mode) {
    CudaErrorWrapper(cudaMalloc((void **)&p_device_total_hs_mat_v2_, longest_sentence * sizeof(T *)), "GPU memory allocation failed\n");
    CudaErrorWrapper(cudaMalloc((void **)&p_device_batch_information_v2_, 2 * minibatch_size_ * sizeof(int)), "GPU memory allocation failed\n");
    cudaMemcpy(p_device_total_hs_mat_v2_, p_host_total_hs_mat_v2, longest_sentence * sizeof(T *), cudaMemcpyHostToDevice);
    free(p_host_total_hs_mat_v2);
  }

  if (unk_replacement_mode__) {
    CudaErrorWrapper(cudaMalloc((void **)&p_device_viterbi_alignments_, minibatch_size_ * sizeof(int)), "GPU memory allocation failed\n");
  }

  free(p_host_total_hs_mat);

}


template <typename T>
void AttentionLayer<T>::ClipGradientsFunc() {
  NormClipGpuV2(p_thrust_device_w_a_grad_, p_device_w_a_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2(p_thrust_device_w_p_grad_, p_device_w_p_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2(p_thrust_device_v_p_grad_, p_device_v_p_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2(p_thrust_device_output_bias_grad_, p_device_output_bias_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2(p_thrust_device_w_c_p1_grad_, p_device_w_c_p1_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2(p_thrust_device_w_c_p2_grad_, p_device_w_c_p2_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  // multi_attention_v2 is not written
}


template <typename T>
void AttentionLayer<T>::LoadWeights(std::ifstream &input_stream) {
  ReadMatrixGpu(p_device_w_a_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_w_p_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_v_p_, lstm_size_, 1, input_stream);
  ReadMatrixGpu(p_device_output_bias_, lstm_size_, 1, input_stream);
  ReadMatrixGpu(p_device_w_c_p1_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_w_c_p2_, lstm_size_, lstm_size_, input_stream);

  if (multi_attention_v2_mode_) {
    ReadMatrixGpu(p_device_w_a_v2_, lstm_size_, lstm_size_, input_stream);
    ReadMatrixGpu(p_device_w_p_v2_, lstm_size_, lstm_size_, input_stream);
    ReadMatrixGpu(p_device_v_p_v2_, lstm_size_, 1, input_stream);
    ReadMatrixGpu(p_device_w_c_p3_v2_, lstm_size_, lstm_size_, input_stream);
  }
}


template <typename T>
void AttentionLayer<T>::DumpWeights(std::ofstream &output) {
  WriteMatrixGpu(p_device_w_a_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_w_p_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_v_p_, lstm_size_, 1, output);
  WriteMatrixGpu(p_device_output_bias_, lstm_size_, 1, output);
  WriteMatrixGpu(p_device_w_c_p1_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_w_c_p2_, lstm_size_, lstm_size_, output);

  // multi_attention_v2 is not written
}


}

#endif




