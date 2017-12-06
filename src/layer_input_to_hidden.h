/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef INPUT_TO_HIDDEN_LAYER_H_
#define INPUT_TO_HIDDEN_LAYER_H_

#include <curand_kernel.h>
#include <Eigen/Core>

#include "layer_gpu.h"
#include "deep_rnn.h"
#include "deep_rnn_kernel.h"
#include "global_configuration.h"
#include "node_input_to_hidden.h"
#include "layer_attention.h"
#include "utility_cu.h"
#include "layer_hidden_to_hidden.h"
#include "layer_transfer.h"
#include "another_encoder.h"


namespace neural_machine_translation {


template <typename T>
class NeuralMachineTranslation;


template <typename T>
class InputToHiddenLayer {

public:
  std::vector<LstmInputHiddenNode<T>> v_nodes_;  // Stores all the LSTM nodes for forward and backward propagation

  // GPU parameters
public:
  LayerGpuInformation input_hidden_layer_information_;

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

  T *p_host_m_o_grad_;
  T *p_host_m_f_grad_;
  T *p_host_m_i_grad_;
  T *p_host_m_c_grad_;

  T *p_host_w_;

  T *p_host_b_o_grad_;
  T *p_host_b_f_grad_;
  T *p_host_b_i_grad_;
  T *p_host_b_c_grad_;

  T *p_host_ones_minibatch_;

  T *p_host_m_o_;
  T *p_host_m_f_;
  T *p_host_m_i_;
  T *p_host_m_c_;

  T *p_host_w_grad_;

  T *p_host_b_o_;
  T *p_host_b_f_;
  T *p_host_b_i_;
  T *p_host_b_c_;

  T *p_host_tmp5_;
  T *p_host_tmp6_;
  T *p_host_tmp7_;
  T *p_host_tmp8_;

public:
  int *p_host_input_vocab_indices_;       // Convert this into 0/1's and to one with no -1's as indices
  int *p_device_input_vocab_indices_;     // minibatch_size_ x longest_sent_, (-1 -1 -1 2 179 0 10 22 0)
  int current_length_;                    // the current length of this target or source sequence
  int w_grad_length_;                     // special length for the w_grad special preprocessing for vocab indices
  
public:
  // contains the entire input sequence, use pointer arithmetic to pass correct segments to LSTM cells
  int *p_host_input_vocab_indices_full_;      // only for debugging
  int *p_host_input_vocab_indices_01_full_;   // only for debugging
  int *p_host_input_vocab_indices_wgrad_;     // only for debugging
  int *p_device_input_vocab_indices_full_;    // minibatch_size_ x longest_sent_
                                              // (-1 -1 -1 2 179 0 10 22 0) => (0 0 0 2 179 0 10 22 0)
  int *p_device_input_vocab_indices_01_full_; // minibatch_szie_ x longest_sent_
                                              // (-1 -1 -1 2 179 0 10 22 0) => (0 0 0 1 1 1 1 1 1)
  int *p_device_input_vocab_indices_wgrad_;   // minibatch_size_ x longest_sent_
                                              // (-1 -1 -1 2 179 0 10 22 0) => (-1 -1 -1 2 179 0 10 22 -1) => (22 10 0 2 179 -1 -1 -1 -1)

public:
  // for setting inital cell and hidden state values
  T *p_host_init_hidden_vec_;
  T *p_host_init_cell_vec_;
  T *p_device_init_hidden_vec_;                                  // lstm_size_ x minibatch_size_, init with 0
  T *p_device_init_cell_vec_;                                    // lstm_size_ x minibatch_size_, init with 0

  T *p_host_init_d_errn_to_tp1_ht_;
  T *p_host_init_d_errn_to_tp1_ct_;
  T *p_device_init_d_errn_to_tp1_ht_;                            // lstm_size_ x minibatch_size_
  T *p_device_init_d_errn_to_tp1_ct_;                            // lstm_size_ x minibatch_size_

public:
  T *p_device_zeros_;                  // pass this in for backprop gpu prep from source size (all zero error matrix)

public:
  // stuff for norm clipping
  T *p_device_result_;
  T *p_device_result_tmp_;

public:
  // device pointers
  T *p_device_tmp1_;                                            // lstm_size_ x minibatch_size_, for calculating LstmInputHiddenNode::p_device_i_t_
  T *p_device_tmp2_;                                            // ...                         , for calculating LstmInputHiddenNode::p_device_i_t_
  T *p_device_tmp3_;                                            // ...                         , for calculating LstmInputHiddenNode::p_device_f_t_
  T *p_device_tmp4_;                                            // ...                         , for calculating LstmInputHiddenNode::p_device_f_t_

  T *p_device_w_ho_;                                            // lstm_size_ x lstm_size_
  T *p_device_w_hf_;                                            // ...
  T *p_device_w_hi_;                                            // ...
  T *p_device_w_hc_;                                            // ...

  T *p_device_w_ho_grad_;
  T *p_device_w_hf_grad_;
  T *p_device_w_hi_grad_;
  T *p_device_w_hc_grad_;

  T *p_device_m_o_grad_;
  T *p_device_m_f_grad_;
  T *p_device_m_i_grad_;
  T *p_device_m_c_grad_;

  T *p_device_w_;                                              // lstm_size_ x vocab_size_

  T *p_device_b_o_grad_;
  T *p_device_b_f_grad_;
  T *p_device_b_i_grad_;
  T *p_device_b_c_grad_;

  T *p_device_ones_minibatch_;

  T *p_device_m_o_;                                             // lstm_size_ x lstm_size_
  T *p_device_m_f_;                                             // ...
  T *p_device_m_i_;                                             // ...
  T *p_device_m_c_;                                             // ...

  ////////////
  T *p_device_w_grad_;

  T *p_device_small_w_grad_;       // lstm_size_ x (minibatch_size_ * longest_sent_) default this
                                   // or lstm_size_ x w_grad_length_
                                   // this is aligned to p_device_input_vocab_indices_wgrad_ (minibatch_size_ * longest_sent_)
  thrust::device_ptr<T> p_thrust_device_small_w_grad_;
  int *p_device_reverse_unique_indices_;                        // vocab_size_
  ////////////

  // bias
  T *p_device_b_o_;                                             // lstm_size_ x 1
  T *p_device_b_f_;                                             // ...
  T *p_device_b_i_;                                             // ...
  T *p_device_b_c_;                                             // ...

  T *p_device_tmp5_;                                            // lstm_size_ x minibatch_size_, for calculating LstmInputHiddenNode::p_device_c_prime_t_tanh_
  T *p_device_tmp6_;                                            // ...                         , for calculating LstmInputHiddenNode::p_device_c_prime_t_tanh_
  T *p_device_tmp7_;                                            // ...                         , for calculating LstmInputHiddenNode::p_device_o_t_
  T *p_device_tmp8_;                                            // ...                         , for calculating LstmInputHiddenNode::p_device_o_t_

  // for feed input, attention_model_mode & target & feed_input_mode
  T *p_device_tmp9_;                                            // lstm_size_ x minibatch_size_, for i_t
  T *p_device_tmp10_;                                           // ...                         , for f_t
  T *p_device_tmp11_;                                           // ...                         , for c_prime_t_tanh
  T *p_device_tmp12_;                                           // ...                         , for o_t

  // these are for the feed input connections 
  T *p_device_q_o_;                                             // lstm_size_ x lstm_size_
  T *p_device_q_f_;                                             // ...
  T *p_device_q_i_;                                             // ...
  T *p_device_q_c_;                                             // ...

  T *p_device_q_o_grad_;                                        // lstm_size_ x lstm_size_
  T *p_device_q_f_grad_;                                        // ...
  T *p_device_q_i_grad_;                                        // ...
  T *p_device_q_c_grad_;                                        // ...


public:
  // saving space in the LSTM
  // host pointers
  T *p_host_d_errn_to_t_ht_;
  T *p_host_d_errt_ct_;
  T *p_host_d_errn_to_t_ct_;
  T *p_host_d_errn_to_t_ot_;
  T *p_host_d_errn_to_t_ft_;
  T *p_host_d_errn_to_t_tanhcpt_;
  T *p_host_d_errn_to_t_it_;
  T *p_host_d_errn_to_t_ht_m1_;
  T *p_host_d_errn_to_t_ct_m1_;

  // device pointers
  T *p_device_d_errn_to_t_ht_;
  T *p_device_d_errt_ct_;
  T *p_device_d_errn_to_t_ct_;
  T *p_device_d_errn_to_t_ot_;
  T *p_device_d_errn_to_t_ft_;
  T *p_device_d_errn_to_t_tanhcpt_;
  T *p_device_d_errn_to_t_it_;
  T *p_device_d_errn_to_t_ht_m1_;
  T *p_device_d_errn_to_t_ct_m1_;

public:
  // thrust device pointers to doing parameter updates nicely (not input word embeddings though)
  thrust::device_ptr<T> p_thrust_device_w_ho_grad_;
  thrust::device_ptr<T> p_thrust_device_w_hf_grad_;
  thrust::device_ptr<T> p_thrust_device_w_hi_grad_;
  thrust::device_ptr<T> p_thrust_device_w_hc_grad_;

  thrust::device_ptr<T> p_thrust_device_m_o_grad_;
  thrust::device_ptr<T> p_thrust_device_m_f_grad_;
  thrust::device_ptr<T> p_thrust_device_m_i_grad_;
  thrust::device_ptr<T> p_thrust_device_m_c_grad_;

  thrust::device_ptr<T> p_thrust_device_q_o_grad_;
  thrust::device_ptr<T> p_thrust_device_q_f_grad_;
  thrust::device_ptr<T> p_thrust_device_q_i_grad_;
  thrust::device_ptr<T> p_thrust_device_q_c_grad_;

  thrust::device_ptr<T> p_thrust_device_w_grad_;

  thrust::device_ptr<T> p_thrust_device_b_o_grad_;
  thrust::device_ptr<T> p_thrust_device_b_f_grad_;
  thrust::device_ptr<T> p_thrust_device_b_i_grad_;
  thrust::device_ptr<T> p_thrust_device_b_c_grad_;


public:
  boost::random::mt19937 generator_;       // random number generator for initializing weights

public:
  NeuralMachineTranslation<T> *p_neural_mt_;

public:
  bool debug_mode_;                        // True if want debugging printout, false otherwise
  int minibatch_size_;
  T learning_rate_;
  bool clip_gradient_mode_;
  T norm_clip_;                            // for gradient clipping
  int lstm_size_;
  int longest_sentence_;
  int input_vocab_size_;
  AttentionLayer<T> *p_attention_layer_ = NULL;
  bool feed_input_mode_ = false;

public:
  bool multi_source_attention_mode_ = false;
  AttentionLayer<T> *p_attention_layer_bi_ = NULL;

public:
  bool char_cnn_mode_ = false;

public:
  bool bi_dir_mode_ = false;
  bool share_embeddings_mode_ = false;
  bool combine_embeddings_mode_ = false;

public:
  bool dropout_mode_;                      // for dropout
  T dropout_rate_;
  curandGenerator_t rand_generator_;

public:
  UpperTransferLayer<T> upper_layer_;
  
public:
  InputToHiddenLayer() {}                  // constructor

public:
  void InitInputToHiddenLayer(int lstm_size, int minibatch_size, int vocab_size, \
                              int longest_sentence, bool debug_mode, T learning_rate, \
                              bool clip_gradient_mode, T norm_clip, \
                              NeuralMachineTranslation<precision> *p_neural_mt, \
                              int seed, bool dropout_mode, T dropout_rate, \
                              bool bi_dir_mode, bool share_embeddings_mode, T *p_device_embedding_ptr, \
                              bool combine_embeddings_mode, GlobalConfiguration &config, bool source_mode);


  void InitInputToHiddenLayerGpu(int lstm_size, int minibatch_size, int vocab_size, \
                                 int longest_sentence, bool debug_mode, T learning_rate, \
                                 bool clip_gradient_mode, T norm_clip, \
                                 NeuralMachineTranslation<precision> *p_neural_mt, \
                                 int seed, bool share_embeddings_mode, T *p_device_embedding_ptr, \
                                 bool combine_embeddings_mode, GlobalConfiguration &config, bool source_mode);

public:
  void ClearGradients(bool init);      // Clear the previous gradients
  void ClearGradientsGpu(bool init);

public:
  void UpdateWeights();                // Update the weights of the model
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
  void LoadWeightsDecoderFeedInput(std::ifstream &input_stream);

public:
  void CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols);
  void CheckGradientGpuSparse(T epsilon, T *p_device_mat, T *p_device_grad, int lstm_size, int *p_host_unique_indices, int curr_num_unique);

public:
  // Convert to 0/1's and to indices where there are no -1's
  void PreprocessGpuVocabIndices(int *p_host_input_vocab_indices, int *p_host_input_vocab_indices_wgrad, int current_length, int len_w);


public:
  // swap the states during the decoding process
  // index specifies which node to swap at
  template <typename Derived>
  void SwapStatesDecoding(const Eigen::MatrixBase<Derived> &eigen_indices, int index, T *p_device_tmp_swap_vals);

public:
  void TransferDecodingStatesGpu(T *p_device_h_t, T *p_device_c_t);

public:
  void InitAttention(int device_number, int d, bool feed_input_mode, NeuralMachineTranslation<T> *p_neural_mt, GlobalConfiguration &config);

public:
  void ZeroAttentionError();

public:
  void InitFeedInput(HiddenToHiddenLayer<T> *p_hidden_to_hidden_layer, bool multi_attention_mode);

public:
  void ScaleGradients();

public:
  void UpdateParams();

public:
  void DecoderInitFeedInput();
};

/*
// CHECK: OK //
template <typename T>
void InputToHiddenLayer<T>::InitInputToHiddenLayer(int lstm_size, int minibatch_size, int vocab_size, \
                            int longest_sentence, bool debug_mode, T learning_rate, \
                            bool clip_gradient_mode, T norm_clip, NeuralMachineTranslation<precision> *p_neural_mt, \
                            int seed, bool dropout_mode, T dropout_rate) {

#ifdef DEBUG_CHECKPOINT_1
  std::cerr<<"\n************ In *InputToHiddenLayer* *InitInputToHiddenLayer*\n"<<std::flush;
  std::cerr<<"   lstm_size: "<<lstm_size<<"\n"
           <<"   minibatch_size: "<<minibatch_size<<"\n"
           <<"   vocab_size: "<<vocab_size<<"\n"
           <<"   longest_sentence: "<<longest_sentence<<"\n"
           <<"   debug_mode: "<<debug_mode<<"\n"
           <<"   learning_rate: "<<learning_rate<<"\n"
           <<"   gradient_clip_mode: "<<clip_gradient_mode<<"\n"
           <<"   norm_clip_threshold: "<<norm_clip<<"\n"
           <<"   seed: "<<seed<<"\n"
           <<"   dropout_mode: "<<dropout_mode<<"\n"
           <<"   dropout_rate: "<<dropout_rate<<"\n"<<std::flush;
#endif

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

  generator_.seed(seed);

  InitInputToHiddenLayerGpu(lstm_size, minibatch_size, vocab_size, longest_sentence, debug_mode, learning_rate, \
                            clip_gradient_mode, norm_clip, p_neural_mt, seed);

  // Initialize the vector of LSTM nodes to longest sentence
  v_nodes_.clear();
#ifdef DEBUG_CHECKPOINT_1
  std::cerr<< "\n************ In *InputToHiddenLayer* *v_nodes_.push_back*\n" << std::flush;
  std::cerr<< "   longest_sentence: " << longest_sentence << "\n"
           <<"   lstm_size: "<<lstm_size<<"\n"
           <<"   minibatch_size: "<<minibatch_size<<"\n"
           <<"   vocab_size: "<<vocab_size<<"\n"
           <<"   dropout_mode: "<<dropout_mode<<"\n"
           <<"   dropout_rate: "<<dropout_rate<<"\n"
           <<std::flush;
#endif
  for (int i = 0; i < longest_sentence; ++i) {
    v_nodes_.push_back(LstmInputHiddenNode<T>(lstm_size, minibatch_size, vocab_size, this, i, p_device_zeros_, dropout_mode, dropout_rate));
  }
}
*/


// CHECK DECODER: OK //
template <typename T>
void InputToHiddenLayer<T>::InitInputToHiddenLayer(int lstm_size, int minibatch_size, int vocab_size, \
                                                   int longest_sentence, bool debug_mode, T learning_rate, \
                                                   bool clip_gradient_mode, T norm_clip, \
                                                   NeuralMachineTranslation<precision> *p_neural_mt, \
                                                   int seed, bool dropout_mode, T dropout_rate, \
                                                   bool bi_dir_mode, bool share_embeddings_mode, T *p_device_embedding_ptr, \
                                                   bool combine_embeddings_mode, GlobalConfiguration &config, bool source_mode) {

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
  combine_embeddings_mode_ = combine_embeddings_mode;
  share_embeddings_mode_ = share_embeddings_mode;

  generator_.seed(seed);

  InitInputToHiddenLayerGpu(lstm_size, minibatch_size, vocab_size, longest_sentence, debug_mode, learning_rate, \
                            clip_gradient_mode, norm_clip, p_neural_mt, seed, share_embeddings_mode, p_device_embedding_ptr, \
                            combine_embeddings_mode, config, source_mode);

  // Initialize the vector of LSTM nodes to longest sentence
  v_nodes_.clear();
  for (int i = 0; i < longest_sentence; ++i) {
    LstmInputHiddenNode<T> lstm_input_hidden_node;
    lstm_input_hidden_node.Init(lstm_size, minibatch_size, vocab_size, this, i, p_device_zeros_, dropout_mode, dropout_rate);
    v_nodes_.push_back(lstm_input_hidden_node);
  }
}


/*
// CHECK: OK //
template <typename T>
void InputToHiddenLayer<T>::InitInputToHiddenLayerGpu(int lstm_size, int minibatch_size, int vocab_size, \
                                                      int longest_sentence, bool debug_mode, T learning_rate, \
                                                      bool clip_gradient_mode, T norm_clip, \
                                                      NeuralMachineTranslation<precision> *p_neural_mt, \
                                                      int seed) {

#ifdef DEBUG_CHECKPOINT_1
  std::cerr<<"\n************ In *InputToHiddenLayer* *InitInputToHiddenLayerGpu*\n"<<std::flush;
  std::cerr<<"   lstm_size: "<<lstm_size<<"\n"
           <<"   minibatch_size: "<<minibatch_size<<"\n"
           <<"   vocab_size: "<<vocab_size<<"\n"
           <<"   longest_sentence: "<<longest_sentence<<"\n"
           <<"   debug_mode: "<<debug_mode<<"\n"
           <<"   learning_rate: "<<learning_rate<<"\n"
           <<"   gradient_clip_mode: "<<clip_gradient_mode<<"\n"
           <<"   norm_clip_threshold: "<<norm_clip<<"\n"
           <<"   seed: "<<seed<<"\n"<<std::flush;
#endif
    
  cudaSetDevice(input_hidden_layer_information_.device_number_);
  
  FullMatrixSetup(&p_host_w_ho_, &p_device_w_ho_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hf_, &p_device_w_hf_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hi_, &p_device_w_hi_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hc_, &p_device_w_hc_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_w_ho_grad_, &p_device_w_ho_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hf_grad_, &p_device_w_hf_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hi_grad_, &p_device_w_hi_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hc_grad_, &p_device_w_hc_grad_, lstm_size, lstm_size);


  FullMatrixSetup(&p_host_m_o_, &p_device_m_o_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_f_, &p_device_m_f_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_i_, &p_device_m_i_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_c_, &p_device_m_c_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_m_o_grad_, &p_device_m_o_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_f_grad_, &p_device_m_f_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_i_grad_, &p_device_m_i_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_c_grad_, &p_device_m_c_grad_, lstm_size, lstm_size);


  FullMatrixSetup(&p_host_b_o_, &p_device_b_o_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_f_, &p_device_b_f_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_i_, &p_device_b_i_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_c_, &p_device_b_c_, lstm_size, 1);

  FullMatrixSetup(&p_host_b_o_grad_, &p_device_b_o_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_f_grad_, &p_device_b_f_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_i_grad_, &p_device_b_i_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_c_grad_, &p_device_b_c_grad_, lstm_size, 1);

  FullMatrixSetup(&p_host_w_, &p_device_w_, lstm_size, vocab_size);
  FullMatrixSetup(&p_host_w_grad_, &p_device_w_grad_, lstm_size, vocab_size);

  input_vocab_size_ = vocab_size;

  FullMatrixSetupZeros(&p_host_init_hidden_vec_, &p_device_init_hidden_vec_, lstm_size, minibatch_size);
  FullMatrixSetupZeros(&p_host_init_cell_vec_, &p_device_init_cell_vec_, lstm_size, minibatch_size);
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

  FullMatrixSetupZeros(&p_host_input_vocab_indices_, &p_device_input_vocab_indices_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_full_, &p_device_input_vocab_indices_full_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_01_full_, &p_device_input_vocab_indices_01_full_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_wgrad_, &p_device_input_vocab_indices_wgrad_, minibatch_size, longest_sentence);

  // Set p_device_zeros_ to 0
  CudaErrorWrapper(cudaMalloc((void **)&p_device_zeros_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed zeros\n");
  cudaMemset(p_device_zeros_, 0, lstm_size * minibatch_size * sizeof(T));

  // Set to all 1
  FullVectorSetupOnes(&p_host_ones_minibatch_, &p_device_ones_minibatch_, minibatch_size);

  // Get device pointers
  p_thrust_device_w_ho_grad_ = thrust::device_pointer_cast(p_device_w_ho_grad_);
  p_thrust_device_w_hf_grad_ = thrust::device_pointer_cast(p_device_w_hf_grad_);
  p_thrust_device_w_hi_grad_ = thrust::device_pointer_cast(p_device_w_hi_grad_);
  p_thrust_device_w_hc_grad_ = thrust::device_pointer_cast(p_device_w_hc_grad_);

  p_thrust_device_m_o_grad_ = thrust::device_pointer_cast(p_device_m_o_grad_);
  p_thrust_device_m_f_grad_ = thrust::device_pointer_cast(p_device_m_f_grad_);
  p_thrust_device_m_i_grad_ = thrust::device_pointer_cast(p_device_m_i_grad_);
  p_thrust_device_m_c_grad_ = thrust::device_pointer_cast(p_device_m_c_grad_);

  // Eventually this should be removed, since a custom reduction kernel does this
  p_thrust_device_w_grad_ = thrust::device_pointer_cast(p_device_w_grad_);

  p_thrust_device_b_o_grad_ = thrust::device_pointer_cast(p_device_b_o_grad_);
  p_thrust_device_b_f_grad_ = thrust::device_pointer_cast(p_device_b_f_grad_);
  p_thrust_device_b_i_grad_ = thrust::device_pointer_cast(p_device_b_i_grad_);
  p_thrust_device_b_c_grad_ = thrust::device_pointer_cast(p_device_b_c_grad_);


  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_, 1 * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_tmp_, NORM_THREADS * sizeof(T)), "GPU memory allocation failed\n");

  // Saving space in the LSTM
  FullMatrixSetup(&p_host_d_errn_to_t_ht_, &p_device_d_errn_to_t_ht_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errt_ct_, &p_device_d_errt_ct_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ct_, &p_device_d_errn_to_t_ct_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ot_, &p_device_d_errn_to_t_ot_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ft_, &p_device_d_errn_to_t_ft_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_tanhcpt_, &p_device_d_errn_to_t_tanhcpt_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_it_, &p_device_d_errn_to_t_it_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ht_m1_, &p_device_d_errn_to_t_ht_m1_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ct_m1_, &p_device_d_errn_to_t_ct_m1_, lstm_size, minibatch_size);


  curandCreateGenerator(&rand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
  boost::uniform_int<> unif_boost(1, 1000000);
  curandSetPseudoRandomGeneratorSeed(rand_generator_, unif_boost(generator__));

  ClearGradients(true);

  cudaSetDevice(input_hidden_layer_information_.device_number_);
  cudaDeviceSynchronize();
  cudaSetDevice(0);
}
*/


// CHECK DECODER: OK //
template <typename T>
void InputToHiddenLayer<T>::InitInputToHiddenLayerGpu(int lstm_size, int minibatch_size, int vocab_size, \
                                                      int longest_sentence, bool debug_mode, T learning_rate, \
                                                      bool clip_gradient_mode, T norm_clip, \
                                                      NeuralMachineTranslation<precision> *p_neural_mt, \
                                                      int seed, bool share_embeddings_mode, T *p_device_embedding_ptr, \
                                                      bool combine_embeddings_mode, GlobalConfiguration &config, bool source_mode) {
    
  cudaSetDevice(input_hidden_layer_information_.device_number_);
  
  FullMatrixSetup(&p_host_w_ho_, &p_device_w_ho_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hf_, &p_device_w_hf_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hi_, &p_device_w_hi_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hc_, &p_device_w_hc_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_w_ho_grad_, &p_device_w_ho_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hf_grad_, &p_device_w_hf_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hi_grad_, &p_device_w_hi_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_w_hc_grad_, &p_device_w_hc_grad_, lstm_size, lstm_size);


  FullMatrixSetup(&p_host_m_o_, &p_device_m_o_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_f_, &p_device_m_f_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_i_, &p_device_m_i_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_c_, &p_device_m_c_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_m_o_grad_, &p_device_m_o_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_f_grad_, &p_device_m_f_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_i_grad_, &p_device_m_i_grad_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_m_c_grad_, &p_device_m_c_grad_, lstm_size, lstm_size);


  FullMatrixSetup(&p_host_b_i_, &p_device_b_i_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_f_, &p_device_b_f_, lstm_size, 1);

  thrust::device_ptr<T> bias_ptr = thrust::device_pointer_cast(p_device_b_f_);
  for (int i = 0; i < lstm_size; ++i) {
    bias_ptr[i] = 1;
  }

  FullMatrixSetup(&p_host_b_o_, &p_device_b_o_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_c_, &p_device_b_c_, lstm_size, 1);

  FullMatrixSetup(&p_host_b_o_grad_, &p_device_b_o_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_f_grad_, &p_device_b_f_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_i_grad_, &p_device_b_i_grad_, lstm_size, 1);
  FullMatrixSetup(&p_host_b_c_grad_, &p_device_b_c_grad_, lstm_size, 1);

  if (share_embeddings_mode) {
    p_device_w_ = p_device_embedding_ptr;
  } else {
    FullMatrixSetup(&p_host_w_, &p_device_w_, lstm_size, vocab_size);
  }

  //FullMatrixSetup(&p_host_w_, &p_device_w_, lstm_size, vocab_size);
  //FullMatrixSetup(&p_host_w_grad_, &p_device_w_grad_, lstm_size, vocab_size);

  input_vocab_size_ = vocab_size;

  FullMatrixSetupZeros(&p_host_init_hidden_vec_, &p_device_init_hidden_vec_, lstm_size, minibatch_size);
  FullMatrixSetupZeros(&p_host_init_cell_vec_, &p_device_init_cell_vec_, lstm_size, minibatch_size);
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

  FullMatrixSetupZeros(&p_host_input_vocab_indices_, &p_device_input_vocab_indices_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_full_, &p_device_input_vocab_indices_full_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_01_full_, &p_device_input_vocab_indices_01_full_, minibatch_size, longest_sentence);
  FullMatrixSetupZeros(&p_host_input_vocab_indices_wgrad_, &p_device_input_vocab_indices_wgrad_, minibatch_size, longest_sentence);

  // Set p_device_zeros_ to 0
  CudaErrorWrapper(cudaMalloc((void **)&p_device_zeros_, lstm_size * minibatch_size * sizeof(T)), "GPU memory allocation failed zeros\n");
  cudaMemset(p_device_zeros_, 0, lstm_size * minibatch_size * sizeof(T));

  // Set to all 1
  FullVectorSetupOnes(&p_host_ones_minibatch_, &p_device_ones_minibatch_, minibatch_size);

  // Get device pointers
  p_thrust_device_w_ho_grad_ = thrust::device_pointer_cast(p_device_w_ho_grad_);
  p_thrust_device_w_hf_grad_ = thrust::device_pointer_cast(p_device_w_hf_grad_);
  p_thrust_device_w_hi_grad_ = thrust::device_pointer_cast(p_device_w_hi_grad_);
  p_thrust_device_w_hc_grad_ = thrust::device_pointer_cast(p_device_w_hc_grad_);

  p_thrust_device_m_o_grad_ = thrust::device_pointer_cast(p_device_m_o_grad_);
  p_thrust_device_m_f_grad_ = thrust::device_pointer_cast(p_device_m_f_grad_);
  p_thrust_device_m_i_grad_ = thrust::device_pointer_cast(p_device_m_i_grad_);
  p_thrust_device_m_c_grad_ = thrust::device_pointer_cast(p_device_m_c_grad_);

  // Eventually this should be removed, since a custom reduction kernel does this
  //p_thrust_device_w_grad_ = thrust::device_pointer_cast(p_device_w_grad_);

  FullMatrixSetup(&p_host_tmp1_, &p_device_small_w_grad_, lstm_size * minibatch_size, longest_sentence);
  p_thrust_device_small_w_grad_ = thrust::device_pointer_cast(p_device_small_w_grad_);
  CudaErrorWrapper(cudaMalloc((void **)&p_device_reverse_unique_indices_, vocab_size * sizeof(int)), "GPU memory allocation failed\n");
  cudaMemset(p_device_small_w_grad_, 0, lstm_size * longest_sentence * minibatch_size * sizeof(T));
  cudaMemset(p_device_reverse_unique_indices_, 0, vocab_size * sizeof(int));


  p_thrust_device_b_o_grad_ = thrust::device_pointer_cast(p_device_b_o_grad_);
  p_thrust_device_b_f_grad_ = thrust::device_pointer_cast(p_device_b_f_grad_);
  p_thrust_device_b_i_grad_ = thrust::device_pointer_cast(p_device_b_i_grad_);
  p_thrust_device_b_c_grad_ = thrust::device_pointer_cast(p_device_b_c_grad_);


  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_, 1 * sizeof(T)), "GPU memory allocation failed\n");
  CudaErrorWrapper(cudaMalloc((void **)&p_device_result_tmp_, NORM_THREADS * sizeof(T)), "GPU memory allocation failed\n");

  // Saving space in the LSTM
  FullMatrixSetup(&p_host_d_errn_to_t_ht_, &p_device_d_errn_to_t_ht_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errt_ct_, &p_device_d_errt_ct_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ct_, &p_device_d_errn_to_t_ct_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ot_, &p_device_d_errn_to_t_ot_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ft_, &p_device_d_errn_to_t_ft_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_tanhcpt_, &p_device_d_errn_to_t_tanhcpt_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_it_, &p_device_d_errn_to_t_it_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ht_m1_, &p_device_d_errn_to_t_ht_m1_, lstm_size, minibatch_size);
  FullMatrixSetup(&p_host_d_errn_to_t_ct_m1_, &p_device_d_errn_to_t_ct_m1_, lstm_size, minibatch_size);


  curandCreateGenerator(&rand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
  //boost::uniform_int<> unif_boost(1, 1000000);
  //curandSetPseudoRandomGeneratorSeed(rand_generator_, unif_boost(generator__));
  curandSetPseudoRandomGeneratorSeed(rand_generator_, curr_seed__);
  curr_seed__ += 7;

  // char_cnn_mode is not written //

  ClearGradients(true);

  cudaSetDevice(input_hidden_layer_information_.device_number_);
  cudaDeviceSynchronize();
  //cudaSetDevice(0);
}



// CHECK: OK //
template <typename T>
void InputToHiddenLayer<T>::InitAttention(int device_number, int d, bool feed_input_mode, NeuralMachineTranslation<T> *p_neural_mt, GlobalConfiguration &config) {

#ifdef DEBUG_DROPOUT
  std::cerr<<"\n************CP3 In *InputToHiddenLayer* *InitAttention*\n"
           <<"   device_number: "<<device_number<<"\n"
           <<"               d: "<<d<<"\n"
           <<"   feed_input_mode: "<<feed_input_mode<<"\n"<<std::flush;
#endif

  cudaSetDevice(input_hidden_layer_information_.device_number_);
  p_attention_layer_ = new AttentionLayer<T>(lstm_size_, minibatch_size_, input_hidden_layer_information_.device_number_, d, longest_sentence_, input_hidden_layer_information_.handle_, p_neural_mt, feed_input_mode, clip_gradient_mode_, norm_clip_, dropout_mode_, dropout_rate_, config, false);

  // multi_attention is not written


  // now switch on the attention flag in the attention nodes
  for (int i = 0; i < v_nodes_.size(); ++i) {
    v_nodes_[i].attention_model_mode_ = true;
  }
}


template <typename T>
void InputToHiddenLayer<T>::ZeroAttentionError() {
  cudaSetDevice(input_hidden_layer_information_.device_number_);
  for (int i = 0; i < v_nodes_.size(); ++i) {
    cudaMemset(v_nodes_[i].p_device_d_errt_ht_, 0, lstm_size_ * minibatch_size_ * sizeof(T));
  }
}


// CHECK: OK //
// pass in the pointer pointing to h_tild in the lowest layer
template <typename T>
void InputToHiddenLayer<T>::InitFeedInput(HiddenToHiddenLayer<T> *p_hidden_to_hidden_layer, bool multi_attention_mode) {

  for (int i = 0; i < v_nodes_.size(); ++i) {
    v_nodes_[i].AttentionExtra();
  }

  feed_input_mode_ = true;

  if (NULL != p_attention_layer_) {
    for (int i = 0; i < v_nodes_.size() - 1; ++i) {
      p_attention_layer_->v_nodes_[i].InitFeedInput(v_nodes_[i + 1].p_device_h_tild_);
    }

    for (int i = 0; i < v_nodes_.size() - 1; ++i) {
      v_nodes_[i + 1].p_device_errn_to_t_h_tild_cpy_ = p_attention_layer_->v_nodes_[i].p_device_errt_to_n_htild_below_;
    }

  } else {
    for (int i = 0; i < p_hidden_to_hidden_layer->v_nodes_.size() - 1; ++i) {
      p_hidden_to_hidden_layer->p_attention_layer_->v_nodes_[i].InitFeedInput(v_nodes_[i + 1].p_device_h_tild_);  
    }

    for (int i = 0; i < p_hidden_to_hidden_layer->v_nodes_.size() - 1; ++i) {
      v_nodes_[i + 1].p_device_errn_to_t_h_tild_cpy_ = p_hidden_to_hidden_layer->p_attention_layer_->v_nodes_[i].p_device_errt_to_n_htild_below_;
    }
  }

  cudaSetDevice(input_hidden_layer_information_.device_number_);

  T *p_host_tmp;
  
  FullMatrixSetup(&p_host_tmp, &p_device_q_i_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_f_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_o_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_c_, lstm_size_, lstm_size_);

  FullMatrixSetup(&p_host_tmp, &p_device_q_i_grad_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_f_grad_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_o_grad_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_c_grad_, lstm_size_, lstm_size_);

  FullMatrixSetup(&p_host_tmp, &p_device_tmp9_, lstm_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp10_, lstm_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp11_, lstm_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp12_, lstm_size_, minibatch_size_);

  p_thrust_device_q_i_grad_ = thrust::device_pointer_cast(p_device_q_i_grad_);
  p_thrust_device_q_f_grad_ = thrust::device_pointer_cast(p_device_q_f_grad_);
  p_thrust_device_q_o_grad_ = thrust::device_pointer_cast(p_device_q_o_grad_);
  p_thrust_device_q_c_grad_ = thrust::device_pointer_cast(p_device_q_c_grad_);

  cudaMemset(p_device_q_i_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T));
  cudaMemset(p_device_q_f_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T));
  cudaMemset(p_device_q_o_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T));
  cudaMemset(p_device_q_c_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T));
}


// CHECK DECODER: OK //
// CHECK: OK //
template <typename T>
void InputToHiddenLayer<T>::ClearGradients(bool init) {
  ClearGradientsGpu(init);
}


// CHECK DECODER: OK //
// CHECK: OK //
template <typename T>
void InputToHiddenLayer<T>::ClearGradientsGpu(bool init) {

#ifdef DEBUG_DROPOUT
    std::cerr << "\n************ In *InputToHiddenLayer* *ClearGradientsGpu*\n" << std::flush;
    std::cerr<<"   w_grad_length_: "<<w_grad_length_<<"\n"<<std::flush;
    std::cerr<<"   feed_input: "<<feed_input_mode_<<"\n"<<std::flush;
#endif

  cudaSetDevice(input_hidden_layer_information_.device_number_);

  cudaDeviceSynchronize();

  cudaMemsetAsync(p_device_w_hi_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s00_);
  cudaMemsetAsync(p_device_b_i_grad_, 0, lstm_size_ * 1 * sizeof(T), input_hidden_layer_information_.s01_);

  cudaMemsetAsync(p_device_w_hf_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s02_);
  cudaMemsetAsync(p_device_b_f_grad_, 0, lstm_size_ * 1 * sizeof(T), input_hidden_layer_information_.s03_);

  cudaMemsetAsync(p_device_w_hc_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s04_);
  cudaMemsetAsync(p_device_b_c_grad_, 0, lstm_size_ * 1 * sizeof(T), input_hidden_layer_information_.s05_);

  cudaMemsetAsync(p_device_w_ho_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s06_);
  cudaMemsetAsync(p_device_b_o_grad_, 0, lstm_size_ * 1 * sizeof(T), input_hidden_layer_information_.s07_);

  if (init) {
    //cudaMemset(p_device_w_grad_, 0, lstm_size_ * input_vocab_size_ * sizeof(T));
    cudaMemset(p_device_small_w_grad_, 0, lstm_size_ * minibatch_size_ * longest_sentence_ * sizeof(T));
    /////////// for decoder
    //cudaMemset(p_device_small_w_grad_, 0, lstm_size_ * minibatch_size_ * longest_sentence_ * sizeof(T));
  } else {
    //int threads_per_block = 256;
    //int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
    //dim3 kernel(num_block, 256, 1);
    //ZeroWGradientKernel<<<kernel, threads_per_block, 0, input_hidden_layer_information_.s08_>>>(p_device_w_grad_, p_device_input_vocab_indices_wgrad_, lstm_size_, w_grad_length_);

    ///////////
    cudaMemset(p_device_small_w_grad_, 0, lstm_size_ * w_grad_length_ * sizeof(T));
  }

  cudaMemsetAsync(p_device_m_i_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s09_);
  cudaMemsetAsync(p_device_m_f_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s10_);
  cudaMemsetAsync(p_device_m_o_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s11_);
  cudaMemsetAsync(p_device_m_c_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s12_);

#ifdef DEBUG_CHECKPOINT_1
  std::cerr<<"   feed_input_mode_: "<<feed_input_mode_<<"\n";
#endif

  if (feed_input_mode_) {
    cudaMemsetAsync(p_device_q_i_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s09_);
    cudaMemsetAsync(p_device_q_f_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s10_);
    cudaMemsetAsync(p_device_q_o_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s11_);
    cudaMemsetAsync(p_device_q_c_grad_, 0, lstm_size_ * lstm_size_ * sizeof(T), input_hidden_layer_information_.s12_);
  }

#ifdef DEBUG_CHECKPOINT_1
  std::cerr << "   p_attention_layer_: " << (p_attention_layer_ != NULL) << "\n";
#endif

  if (p_attention_layer_ != NULL) {
    p_attention_layer_->ClearGradients();
    
    // multi_source_attention is not written //
  }

  // char_cnn_mode is not written //

  DeviceSyncAll();
}


template <typename T>
void InputToHiddenLayer<T>::CheckAllGradients(T epsilon) {
  CheckAllGradientsGpu(epsilon);
}

template <typename T>
void InputToHiddenLayer<T>::CheckAllGradientsGpu(T epsilon) {

  cudaSetDevice(input_hidden_layer_information_.device_number_);

  logger<<">> Gradient checking for input layer gpu\n";
  logger<<"   Gradient checking for p_device_w_hi_\n";
  CheckGradientGpu(epsilon, p_device_w_hi_, p_device_w_hi_grad_, lstm_size_, lstm_size_);

  logger<<"   Gradient checking for p_device_w_hf_\n";
  CheckGradientGpu(epsilon, p_device_w_hf_, p_device_w_hf_grad_, lstm_size_, lstm_size_);

  logger<<"   Gradient checking for p_device_w_ho_\n";
  CheckGradientGpu(epsilon, p_device_w_ho_, p_device_w_ho_grad_, lstm_size_, lstm_size_);

  logger<<"   Gradient checking for p_device_w_hc_\n";
  CheckGradientGpu(epsilon, p_device_w_hc_, p_device_w_hc_grad_, lstm_size_, lstm_size_);

  logger<<"   Gradient checking for p_device_b_i_\n";
  CheckGradientGpu(epsilon, p_device_b_i_, p_device_b_i_grad_, lstm_size_, 1);

  logger<<"   Gradient checking for p_device_b_f_\n";
  CheckGradientGpu(epsilon, p_device_b_f_, p_device_b_f_grad_, lstm_size_, 1);

  logger<<"   Gradient checking for p_device_b_c_\n";
  CheckGradientGpu(epsilon, p_device_b_c_, p_device_b_c_grad_, lstm_size_, 1);

  logger<<"   Gradient checking for p_device_b_o_\n";
  CheckGradientGpu(epsilon, p_device_b_o_, p_device_b_o_grad_, lstm_size_, 1);

  logger<<"   Gradient checking for p_device_m_i_\n";
  CheckGradientGpu(epsilon, p_device_m_i_, p_device_m_i_grad_, lstm_size_, lstm_size_);

  logger<<"   Gradient checking for p_device_m_f_\n";
  CheckGradientGpu(epsilon, p_device_m_f_, p_device_m_f_grad_, lstm_size_, lstm_size_);

  logger<<"   Gradient checking for p_device_m_o_\n";
  CheckGradientGpu(epsilon, p_device_m_o_, p_device_m_o_grad_, lstm_size_, lstm_size_);

  logger<<"   Gradient checking for p_device_m_c_\n";
  CheckGradientGpu(epsilon, p_device_m_c_, p_device_m_c_grad_, lstm_size_, lstm_size_);

  if (feed_input_mode_) {
    logger<<"   Gradient checking for p_device_q_i_\n";
    CheckGradientGpu(epsilon, p_device_q_i_, p_device_q_i_grad_, lstm_size_, lstm_size_);

    logger<<"   Gradient checking for p_device_q_f_\n";
    CheckGradientGpu(epsilon, p_device_q_f_, p_device_q_f_grad_, lstm_size_, lstm_size_);

    logger<<"   Gradient checking for p_device_q_o_\n";
    CheckGradientGpu(epsilon, p_device_q_o_, p_device_q_o_grad_, lstm_size_, lstm_size_);

    logger<<"   Gradient checking for p_device_q_c_\n";
    CheckGradientGpu(epsilon, p_device_q_c_, p_device_q_c_grad_, lstm_size_, lstm_size_);
  }

  // share_embeddings is not written
  // combine_embeddings is not written

  logger<<"   Gradient checking for p_device_w_\n";
  //CheckGradientGpu(epsilon, p_device_w_, p_device_w_grad_, lstm_size_, input_vocab_size_);
  CheckGradientGpuSparse(epsilon, p_device_w_, p_device_small_w_grad_, lstm_size_, p_host_input_vocab_indices_wgrad_, w_grad_length_);

  if (NULL != p_attention_layer_) {
    p_attention_layer_->CheckGradients(epsilon);

    // multi_source_attention is not written
  }

  // char_cnn_layer is not written

}

template <typename T>
void InputToHiddenLayer<T>::CheckGradientGpu(T epsilon, T *p_device_mat, T *p_device_grad, int rows, int cols) {
  cudaSetDevice(input_hidden_layer_information_.device_number_);

  thrust::device_ptr<T> p_thrust_device_mat = thrust::device_pointer_cast(p_device_mat);
  thrust::device_ptr<T> p_thrust_device_grad = thrust::device_pointer_cast(p_device_grad);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      T loss = 0;
      p_thrust_device_mat[IDX2C(i, j, rows)] += epsilon;
      loss = p_neural_mt_->GetError(true);
      cudaSetDevice(input_hidden_layer_information_.device_number_);
      cudaDeviceSynchronize();

      p_thrust_device_mat[IDX2C(i, j, rows)] += -2 * epsilon;
      loss -= p_neural_mt_->GetError(true);
      cudaSetDevice(input_hidden_layer_information_.device_number_);
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

template <typename T>
void InputToHiddenLayer<T>::CheckGradientGpuSparse(T epsilon, T *p_device_mat, T *p_device_grad, int lstm_size, int *p_host_unique_indices, int curr_num_unique) {
  std::cerr<<"CheckGradientGpuSparse is not written!\n\n";
}




template <typename T>
void InputToHiddenLayer<T>::CalculateGlobalNorm() {
  cudaSetDevice(input_hidden_layer_information_.device_number_);
  DeviceSyncAll();
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

  if (feed_input_mode_) {
    NormClipGpuV2P1(p_thrust_device_q_i_grad_, p_device_q_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2P1(p_thrust_device_q_f_grad_, p_device_q_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2P1(p_thrust_device_q_o_grad_, p_device_q_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2P1(p_thrust_device_q_c_grad_, p_device_q_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  }

  if (NULL != p_attention_layer_) {
      p_attention_layer_->NormP1();

      // multi_source_attention is not written
  }

  // char_cnn_layer is not written

  DeviceSyncAll();
}


template <typename T>
void InputToHiddenLayer<T>::UpdateGlobalParams() {
  cudaSetDevice(input_hidden_layer_information_.device_number_);
  DeviceSyncAll();

  NormClipGpuV2P2(p_thrust_device_w_hi_grad_, p_device_w_hi_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_hf_grad_, p_device_w_hf_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_hc_grad_, p_device_w_hc_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_w_ho_grad_, p_device_w_ho_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  NormClipGpuV2P2(p_thrust_device_b_i_grad_, p_device_b_i_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_b_f_grad_, p_device_b_f_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_b_c_grad_, p_device_b_c_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_b_o_grad_, p_device_b_o_grad_, norm_clip_, lstm_size_ * 1, p_device_result_tmp_, p_device_result_);

  //NormClipWGpuV2P2(p_device_result_tmp_, p_device_w_grad_, p_device_input_vocab_indices_wgrad_, norm_clip_, w_grad_length_, lstm_size_);
  NormClipGpuV2P2(p_thrust_device_small_w_grad_, p_device_small_w_grad_, norm_clip_, lstm_size_ * w_grad_length_, p_device_result_tmp_, p_device_result_);

  NormClipGpuV2P2(p_thrust_device_m_i_grad_, p_device_m_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_m_f_grad_, p_device_m_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_m_o_grad_, p_device_m_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  NormClipGpuV2P2(p_thrust_device_m_c_grad_, p_device_m_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

  if (feed_input_mode_) {
    NormClipGpuV2P2(p_thrust_device_q_i_grad_, p_device_q_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2P2(p_thrust_device_q_f_grad_, p_device_q_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2P2(p_thrust_device_q_o_grad_, p_device_q_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2P2(p_thrust_device_q_c_grad_, p_device_q_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
  }

  if (NULL != p_attention_layer_) {
    p_attention_layer_->NormP2();

    // multi_source_attention is not written
  }

  // char_cnn_layer is not written

  UpdateParams();
  DeviceSyncAll();
}


template <typename T>
void InputToHiddenLayer<T>::UpdateParams() {

  cudaSetDevice(input_hidden_layer_information_.device_number_);

  T alpha = learning_rate_;
  T beta = 1;

  cudaDeviceSynchronize();

  if ((source_side_mode__ && train_source_rnn_mode__) || (!source_side_mode__ && train_target_rnn_mode__)) {
    // normal matrices
    // stream 00
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s00_);
    // p_device_w_hi_ += learning_rate_ * p_device_w_hi_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_hi_grad_, lstm_size_, &beta, p_device_w_hi_, lstm_size_, p_device_w_hi_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_w_hi_ failed\n");

    // stream 02
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s02_);
    // p_device_w_hf_ += learning_rate_ * p_device_w_hf_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_hf_grad_, lstm_size_, &beta, p_device_w_hf_, lstm_size_, p_device_w_hf_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_w_hf_ failed\n");

    // stream 04
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s04_);
    // p_device_w_hc_ += learning_rate_ * p_device_w_hc_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_hc_grad_, lstm_size_, &beta, p_device_w_hc_, lstm_size_, p_device_w_hc_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_w_hc_ failed\n");

    // stream 06
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s06_);
    // p_device_w_ho_ += learning_rate_ * p_device_w_ho_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_w_ho_grad_, lstm_size_, &beta, p_device_w_ho_, lstm_size_, p_device_w_ho_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_w_ho_ failed\n");

    // stream 09
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s09_);
    // p_device_m_i_ += learning_rate_ * p_device_m_i_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_i_grad_, lstm_size_, &beta, p_device_m_i_, lstm_size_, p_device_m_i_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_m_i_ failed\n");

    // stream 10
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s10_);
    // p_device_m_f_ += learning_rate_ * p_device_m_f_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_f_grad_, lstm_size_, &beta, p_device_m_f_, lstm_size_, p_device_m_f_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_m_f_ failed\n");

    // stream 11
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s11_);
    // p_device_m_o_ += learning_rate_ * p_device_m_o_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_o_grad_, lstm_size_, &beta, p_device_m_o_, lstm_size_, p_device_m_o_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_m_o_ failed\n");
  
    // stream 12
    cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s12_);
    // p_device_m_c_ += learning_rate_ * p_device_m_c_grad_
    // lstm_size_ x lstm_size_
    CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_m_c_grad_, lstm_size_, &beta, p_device_m_c_, lstm_size_, p_device_m_c_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_m_c_ failed\n");

    if (feed_input_mode_) {
      // stream 09
      cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s09_);
      // p_device_q_i_ += learning_rate_ * p_device_q_i_grad_
      // lstm_size_ x lstm_size_
      CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_q_i_grad_, lstm_size_, &beta, p_device_q_i_, lstm_size_, p_device_q_i_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_q_i_ failed\n");

      // stream 10
      cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s10_);
      // p_device_q_f_ += learning_rate_ * p_device_q_f_grad_
      // lstm_size_ x lstm_size_
      CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_q_f_grad_, lstm_size_, &beta, p_device_q_f_, lstm_size_, p_device_q_f_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_q_f_ failed\n");

      // stream 11
      cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s11_);
      // p_device_q_o_ += learning_rate_ * p_device_q_o_grad_
      // lstm_size_ x lstm_size_
      CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_q_o_grad_, lstm_size_, &beta, p_device_q_o_, lstm_size_, p_device_q_o_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_q_o_ failed\n");

      // stream 12
      cublasSetStream(input_hidden_layer_information_.handle_, input_hidden_layer_information_.s12_);
      // p_device_q_c_ += learning_rate_ * p_device_q_c_grad_
      // lstm_size_ x lstm_size_
      CublasErrorWrapper(CublasGeamWrapper(input_hidden_layer_information_.handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, lstm_size_, &alpha, p_device_q_c_grad_, lstm_size_, &beta, p_device_q_c_, lstm_size_, p_device_q_c_, lstm_size_), "InputToHiddenLayer::UpdateParams p_device_q_c_ failed\n");
    }

    // p_device_b_i_ += learning_rate_ * p_device_b_i_grad_
    // lstm_size x 1
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, input_hidden_layer_information_.s01_>>>(p_device_b_i_, p_device_b_i_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();

    // p_device_b_f_ += learning_rate_ * p_device_b_f_grad_
    // lstm_size x 1
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, input_hidden_layer_information_.s03_>>>(p_device_b_f_, p_device_b_f_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();

    // p_device_b_c_ += learning_rate_ * p_device_b_c_grad_
    // lstm_size x 1
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, input_hidden_layer_information_.s05_>>>(p_device_b_c_, p_device_b_c_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();

    // p_device_b_o_ += learning_rate_ * p_device_b_o_grad_
    // lstm_size x 1
    AddGradVecs<<<(lstm_size_ + 256 - 1) / 256, 256, 0, input_hidden_layer_information_.s07_>>>(p_device_b_o_, p_device_b_o_grad_, learning_rate_, lstm_size_ * 1);
    CudaGetLastError();
  }
  // special w
  //int threads_per_block = 256;
  //int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  //dim3 kernel(num_block, 256, 1);
  //UpdateWGradient<<<kernel, threads_per_block, 0, input_hidden_layer_information_.s08_>>>(p_device_w_, p_device_w_grad_, p_device_input_vocab_indices_wgrad_, learning_rate_, lstm_size_, w_grad_length_);
  //CudaGetLastError();

  if ((source_side_mode__ && train_source_input_embedding_mode__) || (!source_side_mode__ && train_target_input_embedding_mode__)) {
    // p_device_w_[i][vocab_index] (lstm_size_ x vocab_size_) += learning_rate_ * p_device_small_w_grad_[i][k] (lstm_size_ x w_grad_length_)
    // vocab_index = p_device_input_vocab_indices_wgrad_[k]
    UpdateSparseGradient<<<256, 256, 0, input_hidden_layer_information_.s08_>>>(p_device_w_, p_device_small_w_grad_, p_device_input_vocab_indices_wgrad_, w_grad_length_, learning_rate_, lstm_size_);
  }

  if (train_attention_target_rnn_mode__) {
    if (NULL != p_attention_layer_) {
      p_attention_layer_->UpdateParams();
      // multi_source_attention is not written
    }
  }

  // char_cnn_layer is not written

  DeviceSyncAll();
}


template <typename T>
void InputToHiddenLayer<T>::UpdateWeights() {
  UpdateWeightsGpu();
}


template <typename T>
void InputToHiddenLayer<T>::UpdateWeightsGpu() {

  cudaSetDevice(input_hidden_layer_information_.device_number_);
  
  ScaleGradients();

  if (individual_grad_clip_mode__) {
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_w_hi_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_w_hf_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_w_hc_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_w_ho_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);

    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_b_i_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_b_f_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_b_c_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);
    ClipMatKernel<<<std::min(256, (lstm_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_b_o_grad_, individual_norm_clip_threshold__, lstm_size_ * 1);

    // int threads_per_block = 256;
    // int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
    // dim3 kernel(num_block, 256, 1);
    // IndividualClipWGradient<<<kernel, threads_per_block, 0, input_hidden_layer_information_.s00_>>>(p_device_w_grad_, p_device_input_vocab_indices_wgrad_, lstm_size_, individual_norm_clip_threshold__, w_grad_length_);


    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_small_w_grad_, individual_norm_clip_threshold__, lstm_size_ * w_grad_length_);

    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_m_i_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_m_f_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_m_o_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_m_c_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);

    if (feed_input_mode_) {
      ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_q_i_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
      ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_q_f_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
      ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_q_o_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
      ClipMatKernel<<<std::min(256, (lstm_size_ * minibatch_size_ + 256 - 1) / 256), 256, 0, input_hidden_layer_information_.s00_>>>(p_device_q_c_grad_, individual_norm_clip_threshold__, lstm_size_ * lstm_size_);
    }

    if (NULL != p_attention_layer_) {
      p_attention_layer_->ClipIndividual();
      // multi_source_attention is not written
    }

    DeviceSyncAll();
  }

#ifdef DEBUG_DROPOUT_4
  std::cerr<<"   InputToHiddenLayer clip_gradient_mode_: "<<clip_gradient_mode_<<"\n"<<std::flush;
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

    //NormClipWGpuV2(p_device_result_tmp_, p_device_w_grad_, p_device_input_vocab_indices_wgrad_, norm_clip_, w_grad_length_, lstm_size_);
    NormClipGpuV2(p_thrust_device_small_w_grad_, p_device_small_w_grad_, norm_clip_, lstm_size_ * w_grad_length_, p_device_result_tmp_, p_device_result_);

    if (NULL != p_attention_layer_) {
      p_attention_layer_->ClipGradientsFunc();
      // multi_source_attention is not written
    }

    // char_cnn_layer is not written

    NormClipGpuV2(p_thrust_device_m_i_grad_, p_device_m_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_m_f_grad_, p_device_m_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_m_o_grad_, p_device_m_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    NormClipGpuV2(p_thrust_device_m_c_grad_, p_device_m_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);

    if (feed_input_mode_) {
      NormClipGpuV2(p_thrust_device_q_i_grad_, p_device_q_i_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
      NormClipGpuV2(p_thrust_device_q_f_grad_, p_device_q_f_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
      NormClipGpuV2(p_thrust_device_q_o_grad_, p_device_q_o_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
      NormClipGpuV2(p_thrust_device_q_c_grad_, p_device_q_c_grad_, norm_clip_, lstm_size_ * lstm_size_, p_device_result_tmp_, p_device_result_);
    }
  }

  UpdateParams();

  DeviceSyncAll();
}

template <typename T>
void InputToHiddenLayer<T>::LoadWeights(std::ifstream &input_stream) {
  LoadWeightsGpu(input_stream);
}


template <typename T>
void InputToHiddenLayer<T>::LoadWeightsGpu(std::ifstream &input_stream) {
  cudaSetDevice(input_hidden_layer_information_.device_number_);

  ReadMatrixGpu(p_device_w_hi_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_i_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_w_hf_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_f_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_w_hc_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_c_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_w_ho_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_b_o_, lstm_size_, 1, input_stream);

  ReadMatrixGpu(p_device_w_, lstm_size_, input_vocab_size_, input_stream);
  ReadMatrixGpu(p_device_m_i_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_f_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_o_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_c_, lstm_size_, lstm_size_, input_stream);

  if (feed_input_mode_) {
    ReadMatrixGpu(p_device_q_i_, lstm_size_, lstm_size_, input_stream);
    ReadMatrixGpu(p_device_q_f_, lstm_size_, lstm_size_, input_stream);
    ReadMatrixGpu(p_device_q_o_, lstm_size_, lstm_size_, input_stream);
    ReadMatrixGpu(p_device_q_c_, lstm_size_, lstm_size_, input_stream);
  }

  if (NULL != p_attention_layer_) {
    p_attention_layer_->LoadWeights(input_stream);
    if (multi_source_attention_mode_) {
      ; // not use
    }
  }

  // char_cnn_layer != NULL is not written
}


template <typename T>
void InputToHiddenLayer<T>::LoadWeightsDecoderFeedInput(std::ifstream &input_stream) {
  ReadMatrixGpu(p_device_q_i_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_q_f_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_q_o_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_q_c_, lstm_size_, lstm_size_, input_stream);
}



template <typename T>
void InputToHiddenLayer<T>::ScaleGradients() {

  cudaSetDevice(input_hidden_layer_information_.device_number_);

  ScaleFunctor unary_op(minibatch_size_);
  thrust::for_each(p_thrust_device_w_hi_grad_, p_thrust_device_w_hi_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_i_grad_, p_thrust_device_b_i_grad_ + lstm_size_ * 1, unary_op);

  thrust::for_each(p_thrust_device_w_hf_grad_, p_thrust_device_w_hf_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_f_grad_, p_thrust_device_b_f_grad_ + lstm_size_ * 1, unary_op);

  thrust::for_each(p_thrust_device_w_hc_grad_, p_thrust_device_w_hc_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_c_grad_, p_thrust_device_b_c_grad_ + lstm_size_ * 1, unary_op);

  thrust::for_each(p_thrust_device_w_ho_grad_, p_thrust_device_w_ho_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_b_o_grad_, p_thrust_device_b_o_grad_ + lstm_size_ * 1, unary_op);

  //int threads_per_block = 256;
  //int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  //dim3 kernel(num_block, 256, 1);
  //ScaleWGradient<<<kernel, threads_per_block>>>(p_device_w_grad_, p_device_input_vocab_indices_wgrad_, lstm_size_, ((T)1.0) / minibatch_size_, w_grad_length_);
  //CudaGetLastError();

  thrust::for_each(p_thrust_device_small_w_grad_, p_thrust_device_small_w_grad_ + lstm_size_ * w_grad_length_, unary_op);

  thrust::for_each(p_thrust_device_m_i_grad_, p_thrust_device_m_i_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_m_f_grad_, p_thrust_device_m_f_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_m_o_grad_, p_thrust_device_m_o_grad_ + lstm_size_ * lstm_size_, unary_op);
  thrust::for_each(p_thrust_device_m_c_grad_, p_thrust_device_m_c_grad_ + lstm_size_ * lstm_size_, unary_op);

  if (feed_input_mode_) {
    thrust::for_each(p_thrust_device_q_i_grad_, p_thrust_device_q_i_grad_ + lstm_size_ * lstm_size_, unary_op);
    thrust::for_each(p_thrust_device_q_f_grad_, p_thrust_device_q_f_grad_ + lstm_size_ * lstm_size_, unary_op);
    thrust::for_each(p_thrust_device_q_o_grad_, p_thrust_device_q_o_grad_ + lstm_size_ * lstm_size_, unary_op);
    thrust::for_each(p_thrust_device_q_c_grad_, p_thrust_device_q_c_grad_ + lstm_size_ * lstm_size_, unary_op);
  }

  if (NULL != p_attention_layer_) {
    p_attention_layer_->ScaleGradients();
    // multi_source_attention is not written
  }

  // char_cnn_layer is not written

  DeviceSyncAll();
}


// CHECK: OK //
template <typename T>
void InputToHiddenLayer<T>::PreprocessGpuVocabIndices(int *p_host_input_vocab_indices, int *p_host_input_vocab_indices_wgrad, int current_length, int len_w) {
  
  cudaSetDevice(input_hidden_layer_information_.device_number_);

  p_host_input_vocab_indices_ = p_host_input_vocab_indices;
  current_length_ = current_length;
  p_host_input_vocab_indices_wgrad_ = p_host_input_vocab_indices_wgrad;

  // transfer from CPU to the GPU
  cudaMemcpy(p_device_input_vocab_indices_, p_host_input_vocab_indices, minibatch_size_ * current_length * sizeof(int), cudaMemcpyHostToDevice);
  CudaGetLastError("InputToHiddenLayer::p_device_input_vocab_indices_ preprocess");
  cudaMemcpy(p_device_input_vocab_indices_wgrad_, p_host_input_vocab_indices_wgrad, len_w * sizeof(int), cudaMemcpyHostToDevice);
  CudaGetLastError("InputToHiddenLayer::p_device_input_vocab_indices_wgrad_ preprocess");

  w_grad_length_ = len_w;

  // launch kernel to turn into 0/1's and indices with no -1's
  int threads_per_block = 128;
  int blocks_per_grid = 128;
  VocabTo01Kernel<<<blocks_per_grid,threads_per_block>>>(p_device_input_vocab_indices_01_full_, p_device_input_vocab_indices_, current_length * minibatch_size_);
  CudaGetLastError("InputToHiddenLayer::p_device_input_vocab_indices_01_full_ preprocess");

  VocabToNonMinus1Kernel<<<blocks_per_grid,threads_per_block>>>(p_device_input_vocab_indices_full_, p_device_input_vocab_indices_, current_length * minibatch_size_);
  CudaGetLastError("InputToHiddenLayer::p_device_input_vocab_indices_full_ preprocess");

  DeviceSyncAll();
  SetupReverseIndices<<<256, 256>>>(p_device_reverse_unique_indices_, p_device_input_vocab_indices_wgrad_, w_grad_length_);
  CudaGetLastError("InputToHiddenLayer::p_device_reverse_unique_indices_ preprocess");
  DeviceSyncAll();

  if (NULL != p_attention_layer_) {
    p_attention_layer_->transfer_done_ = false;

    // multi_source_attention is not written
  }
}

template <typename T>
template <typename Derived>
void InputToHiddenLayer<T>::SwapStatesDecoding(const Eigen::MatrixBase<Derived> &eigen_indices, int index, T *p_device_tmp_swap_vals) {

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
void InputToHiddenLayer<T>::TransferDecodingStatesGpu(T *p_device_h_t, T *p_device_c_t) {
  for (int i = 0; i < minibatch_size_; ++i) {
    int step = i * lstm_size_;
    CudaErrorWrapper(cudaMemcpy(p_device_init_hidden_vec_ + step, p_device_h_t, lstm_size_ * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "transfer decoding states h_t memcpy failed\n");
    CudaErrorWrapper(cudaMemcpy(p_device_init_cell_vec_ + step, p_device_c_t, lstm_size_ * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "transfer decoding states c_t memcpy failed\n");
  }

  v_nodes_[0].p_device_h_t_prev_ = p_device_init_hidden_vec_;
  v_nodes_[0].p_device_c_t_prev_ = p_device_init_cell_vec_;
}



template <typename T>
void InputToHiddenLayer<T>::DumpWeights(std::ofstream &output) {
  DumpWeightsGpu(output);
}


template <typename T>
void InputToHiddenLayer<T>::DumpWeightsGpu(std::ofstream &output) {
  cudaSetDevice(input_hidden_layer_information_.device_number_);

  WriteMatrixGpu(p_device_w_hi_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_i_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_w_hf_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_f_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_w_hc_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_c_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_w_ho_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_b_o_, lstm_size_, 1, output);

  WriteMatrixGpu(p_device_w_, lstm_size_, input_vocab_size_, output);

  WriteMatrixGpu(p_device_m_i_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_m_f_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_m_o_, lstm_size_, lstm_size_, output);
  WriteMatrixGpu(p_device_m_c_, lstm_size_, lstm_size_, output);

  if (feed_input_mode_) {
    WriteMatrixGpu(p_device_q_i_, lstm_size_, lstm_size_, output);
    WriteMatrixGpu(p_device_q_f_, lstm_size_, lstm_size_, output);
    WriteMatrixGpu(p_device_q_o_, lstm_size_, lstm_size_, output);
    WriteMatrixGpu(p_device_q_c_, lstm_size_, lstm_size_, output);
  }

  if (NULL != p_attention_layer_) {
    p_attention_layer_->DumpWeights(output);
    // multi_source_attention is not written
  }

  // char_cnn_layer is not written
}



template <typename T>
void InputToHiddenLayer<T>::DecoderInitFeedInput() {
  T *p_host_tmp;
  FullMatrixSetup(&p_host_tmp, &p_device_q_i_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_f_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_o_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_q_c_, lstm_size_, lstm_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp9_, lstm_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp10_, lstm_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp11_, lstm_size_, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp12_, lstm_size_, minibatch_size_);
}



}

#endif






