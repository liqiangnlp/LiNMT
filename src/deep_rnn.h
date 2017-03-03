/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef RECURRENT_NEURAL_NETWORK_H_
#define RECURRENT_NEURAL_NETWORK_H_

#include <unordered_map>

#include "global_configuration.h"
#include "layer_gpu.h"
#include "layer_input_to_hidden.h"
#include "layer_hidden_to_hidden.h"
#include "layer_loss.h"
#include "layer_transfer.h"
#include "file_helper.h"


namespace neural_machine_translation {

// Declaration of multi classes
template <typename T>
class PreviousSourceState;

template <typename T>
class PreviousTargetState;

template <typename T>
class InputToHiddenLayer;

template <typename T>
class HiddenToHiddenLayer;

class FileHelper;


template <typename T>
class NeuralMachineTranslation {

public:
  BaseLossLayer<T> *p_softmax_layer_;         // loss layer for the model

public:
  InputToHiddenLayer<T> input_layer_source_;  // first layer of model, the source input to hidden layer
  InputToHiddenLayer<T> input_layer_target_;  // first layer of model, the target input to hidden layer

public:
  std::vector<HiddenToHiddenLayer<T>> v_hidden_layers_source_;    // source hidden layers of model
  std::vector<HiddenToHiddenLayer<T>> v_hidden_layers_target_;    // target hidden layers of model

public:
  FileHelper *p_file_information_;

public:
  SoftmaxLayerGpuInformation softmax_layer_gpu_information_;

public:
  std::ifstream input_stream_;
  std::ofstream output_stream_;

public:
  // passed in from imaginary softmax for source side
  //  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> zero_error_;

public:
  std::string input_weight_file_;
  std::string output_weight_file_;

public:
  bool debug_mode_;

public:
  bool train_perplexity_mode_;
  double train_perplexity_ = 0.0;

public:
  bool truncated_softmax_mode_;

public:
  bool sequence_to_sequence_mode_;    // true if sequence to sequence model, else language model

public:
  bool train_mode_ = false;           // this is for making sure dropout is not used at test time

public:
  bool grad_check_flag_ = false;

public:
  int source_length_ = -1;            // for the attention model

public:
  bool bi_dir_mode_ = false;
  bool multi_source_mode_ = false;


public:
  // for decoding multilayer models, on index for each layer
  bool decode_mode_ = false;
  std::vector<PreviousSourceState<T>> v_previous_source_states_;       // num_layers_ x (lstm_size_ * 1)
  std::vector<PreviousSourceState<T>> v_previous_source_states_bi_;
  std::vector<PreviousTargetState<T>> v_previous_target_states_;       // num_layers_ x (lstm_size_ * beam_size_)
  std::vector<T*> v_top_source_states_;          // longest_sentence_ x (lstm_size_ * beam_size_), for attention model in decoder
  std::vector<T*> v_top_source_states_v2_;       // longest_sentence_, for attention model in decoder
  AttentionLayer<T> decoder_attention_layer_;    // for decoding only 

public:
  bool multi_attention_ = false;
  bool multi_attention_v2_ = false;


public:
  bool char_cnn_mode_ = false;
  CharCnnConfiguration char_cnn_config_;

public:
  AttentionConfiguration attention_configuration_;
  std::ofstream output_alignments_;
    
public:
  NeuralMachineTranslation() {};

public:
  // called at begining of program once to initialize the weights
  void InitModel(int lstm_size, int minibatch_size, \
                 int source_vocab_size, int target_vocab_size, \
                 int longest_sentence, bool debug_mode, T learning_rate, \
                 bool clip_gradient_mode, T norm_clip, \
                 std::string input_weight_file, std::string output_weight_file, \
                 bool softmax_scaled_mode, bool training_perplexity_mode, \
                 bool truncated_softmax_mode, int shortlist_size, int sampled_size, \
                 bool sequence_to_sequence_mode, int layers_number, \
                 std::vector<int> gpu_indicies, bool dropout_mode, T dropout_rate, \
                 AttentionConfiguration attention_configuration, \
                 GlobalConfiguration &configuration);

public:
  // For the decoder
  void InitModelDecoding(int lstm_size, int beam_size, int source_vocab_size, int target_vocab_size, \
                         int num_layers, std::string input_weight_file, int gpu_num, GlobalConfiguration &config, \
                         bool attention_model_mode, bool feed_input_mode, bool multi_source_mode, bool combine_lstm_mode, bool char_cnn_mode);

public:
  /*
  template <typename Derived>
  void ComputeGradients(const Eigen::MatrixBase<Derived> &source_input_minibatch_const, const Eigen::MatrixBase<Derived> &source_output_minibatch_const, \
                        const Eigen::MatrixBase<Derived> &target_input_minibatch_const, const Eigen::MatrixBase<Derived> &target_output_minibatch_const, \
                        int *p_host_input_vocab_indices_source, int *p_host_output_vocab_indices_source, \
                        int *p_host_input_vocab_indices_target, int *p_host_output_vocab_indices_target, \
                        int current_source_length, int current_target_length,
                        int *p_host_input_vocab_indices_source_wgrad, int *p_host_input_vocab_indices_target_grad,
                        int len_source_wgrad, int len_target_wgrad, int *p_host_sampled_indices, \
                        int len_unique_words_trunc_softmax, int *p_host_batch_info);
  */

  void ComputeGradients(int *p_host_input_vocab_indices_source, int *p_host_output_vocab_indices_source, \
                        int *p_host_input_vocab_indices_target, int *p_host_output_vocab_indices_target, \
                        int current_source_length, int current_target_length,
                        int *p_host_input_vocab_indices_source_wgrad, int *p_host_input_vocab_indices_target_grad,
                        int len_source_wgrad, int len_target_wgrad, int *p_host_sampled_indices, \
                        int len_unique_words_trunc_softmax, int *p_host_batch_info, \
                        FileHelper *tmp_file_helper);


public:
  /* sets all gradient matrices to zero, called after a minibatch updates the gradients */
  void ClearGradients();

public:
  void InitFileInformation(FileHelper *file_information);

public:
  void PrintGpuInformation();

public:
  void InitPreviousStates(int num_layers, int lstm_size, int minibatch_size, int device_number, bool multi_source_mode);

public:
  void CheckAllGradients(T epsilon);

public:
  double GetError(bool gpu_flag);

public:
  void UpdateWeights();

public:
  void DumpAlignments(int target_length, int minibatch_size, int *p_host_input_vocab_indices_source, int *p_host_input_vocab_indices_target);

public:
  void UpdateLearningRate(T new_learning_rate);

public:
  double GetPerplexity(std::string test_file_name, int minibatch_size, int &test_num_lines_in_file, int longest_sent, int source_vocab_size, int target_vocab_size, bool load_weights_val, int &test_total_words, bool output_log_mode, bool force_decode_mode, std::string fd_filename);

public:
  void ForwardPropSource(int *p_device_input_vocab_indices_source, int *p_device_input_vocab_indices_source_bi, int *p_device_ones, \
                         int source_length, int source_length_bi, int lstm_size, int *p_device_char_cnn_indices);
  void ForwardPropTarget(int curr_index, int *p_device_current_indices, int *p_device_ones, int lstm_size, int beam_size, \
                         int *p_device_char_cnn_indices);

public:
  void DumpSentenceEmbedding(int lstm_size, std::ofstream &out_sentence_embedding);

public:
  template <typename Derived>
  void SwapDecodingStates(const Eigen::MatrixBase<Derived> &eigen_indices, int index, T *p_device_tmp_swap_vals);

public:
  void TargetCopyPrevStates(int lstm_size, int beam_size);

public:
  void LoadWeights();

public:
  void DumpBestModel(std::string best_model_name, std::string const_model_name);

public:
  void DumpWeights();
};


/////////////////// Implementations for Class Template NeuralMachineTranslation ///////////////////
/*
int lstm_size, int minibatch_size, \
int source_vocab_size, int target_vocab_size, \
int longest_sentence, bool debug_mode, T learning_rate, \
bool clip_gradient_mode, T norm_clip, \
std::string input_weight_file, std::string output_weight_file, \
bool softmax_scaled_mode, bool training_perplexity_mode, \
bool truncated_softmax_mode, int shortlist_size, int sampled_size, \
bool sequence_to_sequence_mode, int layers_number, \
std::vector<int> gpu_indicies, bool dropout_mode, T dropout_rate, \
AttentionConfiguration attention_configuration, \
GlobalConfiguration &configuration
*/
template <typename T>
void NeuralMachineTranslation<T>::InitModel(int lstm_size, int minibatch_size, \
    int source_vocab_size, int target_vocab_size, \
    int longest_sentence, bool debug_mode, T learning_rate, \
    bool clip_gradient_mode, T norm_clip, \
    std::string input_weight_file, std::string output_weight_file, \
    bool softmax_scaled_mode, bool training_perplexity_mode, \
    bool truncated_softmax_mode, int shortlist_size, int sampled_size, \
    bool sequence_to_sequence_mode, int layers_number, \
    std::vector<int> gpu_indices, bool dropout_mode, T dropout_rate, \
    AttentionConfiguration attention_configuration, \
    GlobalConfiguration &configuration) {

#ifdef DEBUG_DROPOUT
  std::cerr << "\n************ In *NeuralMachineTranslation* *InitModel*\n" << std::flush;
  std::cerr<<"   lstm_size: "<<lstm_size<<"\n"
           <<"   minibatch_size: "<<minibatch_size<<"\n"
           <<"   source_vocab_size: "<<source_vocab_size<<"\n"
           <<"   target_vocab_size: "<<target_vocab_size<<"\n"
           <<"   longest_sentence: "<<longest_sentence<<"\n"
           <<"   debug_mode: "<<debug_mode<<"\n"
           <<"   learning_rate: "<<learning_rate<<"\n"
           <<"   gradient_clip_mode: "<<clip_gradient_mode<<"\n"
           <<"   norm_clip_threshold: "<<norm_clip<<"\n"
           <<"   input_weight_file: "<<input_weight_file<<"\n"
           <<"   output_weight_file: "<<output_weight_file<<"\n"
           <<"   softmax_scaled_mode: "<<softmax_scaled_mode<<"\n"
           <<"   training_perplexity_mode: "<<training_perplexity_mode<<"\n"
           <<"   truncated_softmax_mode: "<<truncated_softmax_mode<<"\n"
           <<"   shortlist_size_: "<<shortlist_size<<"\n"
           <<"   sampled_size_: "<<sampled_size<<"\n"
           <<"   sequence_to_sequence_mode: "<<sequence_to_sequence_mode<<"\n"
           <<"   layers_number: "<<layers_number<<"\n"
           <<"   gpu_indices.size: "<<gpu_indices.size()<<"\n"
           <<"   dropout_mode: "<<dropout_mode<<"\n"
           <<"   dropout_rate: "<<dropout_rate<<"\n\n"<<std::flush;

#endif


  // Print GPU information
  logger<<"\n$$ GPU Memory Status\n";
  PrintGpuInformation();
  
  logger<<"\n$$ Initilized Model for NeuralMT\n";

  if (gpu_indices.size() != 0) {
    if (gpu_indices.size() != layers_number + 1) {
      logger<< "Error: multi gup indices you specified are invalid. There must be one index for each layer, plus one index for the softmax.\n";
      exit(EXIT_FAILURE);
    }
  }

  // char_cnn_mode is not written


  int tmp_max_gpu = 0;
  for (int i = 0; i < gpu_indices.size(); ++i) {
    if (gpu_indices[i] > tmp_max_gpu) {
      tmp_max_gpu = gpu_indices[i];
    }
  }

  // for outputting alignments
  if (attention_configuration.dump_alignments_mode_) {
    output_alignments_.open(attention_configuration.alignment_file_name_.c_str());
  }

  // layer 1, layer 2, ..., layer softmax 
  std::vector<int> final_gpu_indices;
  if (gpu_indices.size() != 0) {
    final_gpu_indices = gpu_indices;
  } else {
    for (int i = 0; i < layers_number + 1; ++i) {
      final_gpu_indices.push_back(0);
    }
  }

#ifdef DEBUG_CHECKPOINT_1
  std::cerr << "\n************ In *NeuralMachineTranslation* *InitModel*\n" << std::flush;
  std::cerr<<"   final_gpu_indices.size: "<<final_gpu_indices.size()<<"\n\n";
#endif

  std::unordered_map<int, LayerGpuInformation> layer_lookups;   // get the layer lookups for each GPU
  for (int i = 0; i < final_gpu_indices.size() - 1; ++i) {
    // one gpu just initializes one time
    if (layer_lookups.count(final_gpu_indices[i]) == 0) {
      LayerGpuInformation tmp_layer_gpu_information;
      tmp_layer_gpu_information.Init(final_gpu_indices[i]);
      layer_lookups[final_gpu_indices[i]] = tmp_layer_gpu_information;
    }
  }


#ifdef DEBUG_CHECKPOINT_1
  std::cerr << "\n************ In *NeuralMachineTranslation* *InitModel*\n" << std::flush;
  std::cerr << "   layer_lookups.size: " << layer_lookups.size() << "\n\n";
#endif

  // birdirectional part is not written 
  // bi_dir_mode_

  multi_attention_ = configuration.multi_source_params_.multi_attention_mode_;
  multi_attention_v2_ = configuration.multi_source_params_.multi_attention_v2_mode_;

  // multilanguage LM is not written
  // multi source is not written
  

  input_layer_source_.input_hidden_layer_information_ = layer_lookups[final_gpu_indices[0]];
  // bi_dir || multi_source is not written

  input_layer_target_.input_hidden_layer_information_ = layer_lookups[final_gpu_indices[0]];

  // Initialize the softmax layer
  if (configuration.softmax_mode_) {

#ifdef DEBUG_CHECKPOINT_1
      std::cerr << "\n************ In *NeuralMachineTranslation* *InitModel* *new SoftmaxLayer*\n" << std::flush;
#endif

    p_softmax_layer_ = new SoftmaxLayer<T>();

  } else if (configuration.nce_mode_) {
    p_softmax_layer_ = new NceLayer<T>();
  }

  // Initialize the softmax layer
  softmax_layer_gpu_information_ = p_softmax_layer_->InitGpu(final_gpu_indices.back());
  p_softmax_layer_->InitLossLayer(this, configuration);

  /* Print GPU information */
  logger<<"\n>> (Layer Softmax) Softmax layer was initialized\n";
  PrintGpuInformation();

  /* Initialize the input layer */
  if (sequence_to_sequence_mode) {

#ifdef DEBUG_CHECKPOINT_1
    std::cerr << "\n************ In *InitModel* *InitInputToHiddenLayer* *source*\n" << std::flush;
#endif

    bool top_layer_flag = false;
    //if (layers_number == 1 && )

    bool combine_embeddings = false;

    // Initialize the input layer
    input_layer_source_.InitInputToHiddenLayer(lstm_size, minibatch_size, source_vocab_size, \
                                               longest_sentence, debug_mode, learning_rate, \
                                               clip_gradient_mode, norm_clip, this, \
                                               101, dropout_mode, dropout_rate, top_layer_flag, \
                                               false, NULL, combine_embeddings, configuration, true);
    logger<<"\n>> (Layer 1) Source Input layer was initialized\n";
    PrintGpuInformation();
  }

  // bi_dir_mode is not written
  // multi_source is not written

#ifdef DEBUG_CHECKPOINT_1
  std::cerr << "\n************ In *InitModel* *InitInputToHiddenLayer* *target*\n" << std::flush;
#endif

  input_layer_target_.InitInputToHiddenLayer(lstm_size, minibatch_size, target_vocab_size, \
                                             longest_sentence, debug_mode, learning_rate, \
                                             clip_gradient_mode, norm_clip, this, \
                                             102, dropout_mode, dropout_rate, false, false, NULL, false, configuration, false);
  logger<<"\n>> (Layer 1) Target Input layer was initialized\n";
  PrintGpuInformation();


  input_weight_file_ = input_weight_file;
  output_weight_file_ = output_weight_file;
  debug_mode_ = debug_mode;

  // zero_error is not written

  train_perplexity_mode_ = training_perplexity_mode;
  truncated_softmax_mode_ = truncated_softmax_mode;
  sequence_to_sequence_mode_ = sequence_to_sequence_mode;
  attention_configuration_ = attention_configuration;
  

#ifdef DEBUG_CHECKPOINT_1
  std::cerr << "\n************ In *InitModel* *param1*\n" << std::flush;
  std::cerr<<"   input_weight_file_: "<<input_weight_file_<<"\n"
           <<"   output_weight_file_: "<<output_weight_file_<<"\n"
           <<"   debug_mode_: "<<debug_mode_<<"\n"
           <<"   train_perplexity_mode_: "<<train_perplexity_mode_<<"\n"
           <<"   truncated_softmax_mode_: "<<truncated_softmax_mode_<<"\n"
           <<"   sequence_to_sequence_mode_: "<<sequence_to_sequence_mode_<<"\n"
           <<std::flush;
#endif

  // Initialize hidden layers
  // do this to be sure addresses stay the same
  for (int i = 1; i < layers_number; ++i) {
    if (sequence_to_sequence_mode) {
      v_hidden_layers_source_.push_back(HiddenToHiddenLayer<T>()); 
    }
    v_hidden_layers_target_.push_back(HiddenToHiddenLayer<T>());
  }

  // Now initialize hidden layers
  for (int i = 1; i < layers_number; ++i) {
    if (sequence_to_sequence_mode) {
      bool top_layer_flag = false;

      v_hidden_layers_source_[i-1].hidden_hidden_layer_information_ = layer_lookups[final_gpu_indices[i]];
      v_hidden_layers_source_[i-1].InitHiddenToHiddenLayer(lstm_size, minibatch_size, longest_sentence, debug_mode, \
                                                           learning_rate, clip_gradient_mode, norm_clip, \
                                                           this, 103, dropout_mode, dropout_rate, top_layer_flag, i);
      logger<<"\n>> (Layer "<<i+1<<") Source Hidden layer was initialized\n";
      PrintGpuInformation();
    }

    // bi_dir is not written
    // multi_source is not written

    v_hidden_layers_target_[i-1].hidden_hidden_layer_information_ = layer_lookups[final_gpu_indices[i]];
    v_hidden_layers_target_[i-1].InitHiddenToHiddenLayer(lstm_size, minibatch_size, longest_sentence, debug_mode, \
                                                         learning_rate, clip_gradient_mode, norm_clip, \
                                                         this, 103, dropout_mode, dropout_rate, false, i);

    logger<<"\n>> (Layer "<<i+1<<") Target Hidden layer was initialized\n";
    PrintGpuInformation();
  }

  // bi_dir is not written, initialize the bidirectional layer here ...
  // multi_source is not written

  // Initialize the attention layer on top layer, by this time all the other layers have been initialized
  if (attention_configuration.attention_model_mode_) {
    if (1 == layers_number) {
      // CHECK: OK //
#ifdef DEBUG_CHECKPOINT_3
      std::cerr<<"\n************CP3 In *InitModel* *layer_number = 1*\n"
               <<"   attention_model_mode_: "<<attention_configuration_.attention_model_mode_<<"\n"<<std::flush;
#endif

      input_layer_target_.InitAttention(final_gpu_indices[0], attention_configuration.d_, attention_configuration.feed_input_mode_, this, configuration);

      for (int i = 0; i < longest_sentence; ++i) {
        input_layer_target_.p_attention_layer_->v_nodes_[i].p_device_h_t_ = input_layer_target_.v_nodes_[i].p_device_h_t_;
        input_layer_target_.p_attention_layer_->v_nodes_[i].p_device_d_errt_ht_tild_ = input_layer_target_.v_nodes_[i].p_device_d_errt_ht_;
        input_layer_target_.p_attention_layer_->v_nodes_[i].p_device_indices_mask_ = &input_layer_target_.v_nodes_[i].p_device_input_vocab_indices_01_;
      }

      if (attention_configuration.feed_input_mode_) {
        input_layer_target_.InitFeedInput(NULL, configuration.multi_source_params_.multi_attention_mode_);
        input_layer_target_.input_hidden_layer_information_.attention_forward_ = input_layer_target_.p_attention_layer_->attention_layer_gpu_information_.forward_prop_done_;
        input_layer_target_.p_attention_layer_->attention_layer_gpu_information_.error_htild_below_ = input_layer_target_.input_hidden_layer_information_.error_htild_below_;
      }

    } else {
     
      v_hidden_layers_target_[layers_number - 2].InitAttention(final_gpu_indices[layers_number - 1], attention_configuration.d_, attention_configuration.feed_input_mode_, this, configuration);
      for (int i = 0; i < longest_sentence; ++i) {

        v_hidden_layers_target_[layers_number - 2].p_attention_layer_->v_nodes_[i].p_device_h_t_ = v_hidden_layers_target_[layers_number - 2].v_nodes_[i].p_device_h_t_;
        v_hidden_layers_target_[layers_number - 2].p_attention_layer_->v_nodes_[i].p_device_d_errt_ht_tild_ = v_hidden_layers_target_[layers_number - 2].v_nodes_[i].p_device_d_errt_ht_;
        v_hidden_layers_target_[layers_number - 2].p_attention_layer_->v_nodes_[i].p_device_indices_mask_ = &v_hidden_layers_target_[layers_number - 2].v_nodes_[i].p_device_input_vocab_indices_01_;

        // multi_attention is not written

      }

      if (attention_configuration.feed_input_mode_) {
        input_layer_target_.InitFeedInput(&v_hidden_layers_target_[layers_number - 2], configuration.multi_source_params_.multi_attention_mode_);

        input_layer_target_.input_hidden_layer_information_.attention_forward_ = v_hidden_layers_target_[layers_number - 2].p_attention_layer_->attention_layer_gpu_information_.forward_prop_done_;
        v_hidden_layers_target_[layers_number - 2].p_attention_layer_->attention_layer_gpu_information_.error_htild_below_ = input_layer_target_.input_hidden_layer_information_.error_htild_below_;
      }
    }
    logger<<"\n>> (Layer "<<layers_number<<" Target) Attention layer was initialized\n";
    PrintGpuInformation();
  }

  // bi_dir is not written

  // Init upper transfer layer and lower transfer layer
  if (1 == layers_number) {

#ifdef DEBUG_CHECKPOINT_4
    std::cerr<<"\n************CP4 In *InitModel*\n"
             <<"   gpu_indices[0]: "<<final_gpu_indices[0]<<"\n"
             <<"   gpu_indices[1]: "<<final_gpu_indices[1]<<"\n"
             <<"   dropout_mode: "<<dropout_mode<<"\n"
             <<"   attention_config.attention_mode: "<<attention_configuration.attention_model_mode_<<"\n"
             <<std::flush;
#endif

    // 1 input layer, 1 softmax layer
    // Init upper layer for the input layer

    if (final_gpu_indices[0] == final_gpu_indices[1] && !dropout_mode && !attention_configuration.attention_model_mode_ && false) {
      // CHECK: OK //

      // upper layer lies on the same GPU
#ifdef DEBUG_CHECKPOINT_4
      std::cerr<<"\n Upper layer lies on the same GPU\n"<<std::flush;
#endif

      if (sequence_to_sequence_mode) {
        input_layer_source_.upper_layer_.InitUpperTransferLayer(true, false, true, p_softmax_layer_, NULL);
      }
      input_layer_target_.upper_layer_.InitUpperTransferLayer(true, false, false, p_softmax_layer_, NULL);
      p_softmax_layer_->InitLowerTransferLayer(true, false, &input_layer_target_, NULL);

    } else {
      // CHECK: OK //
      // upper layer lies on different GPUs

#ifdef DEBUG_CHECKPOINT_4
        std::cerr << "\n Upper layer lies on different GPUs\n" << std::flush;
#endif

      if (sequence_to_sequence_mode) {
        input_layer_source_.upper_layer_.InitUpperTransferLayer(true, true, true, p_softmax_layer_, NULL);
      }
      input_layer_target_.upper_layer_.InitUpperTransferLayer(true, true, false, p_softmax_layer_, NULL);
      p_softmax_layer_->InitLowerTransferLayer(true, true, &input_layer_target_, NULL);
    }

    // bi_dir || multi_source is not written

  } else {
    
#ifdef DEBUG_CHECKPOINT_4
    std::cerr<<"\n************CP4 In *InitModel* layer_number > 1\n"<< std::flush;
#endif

    
    // 1 input layer, (layer_num - 1) hidden layer
    // 1 softmax layer

    // Init upper layer for the input layer
    if (final_gpu_indices[0] == final_gpu_indices[1] && !dropout_mode && !attention_configuration.attention_model_mode_ && false) {
      // upper layer lies on the same GPU
      if (sequence_to_sequence_mode) {
        input_layer_source_.upper_layer_.InitUpperTransferLayer(false, false, true, NULL, &v_hidden_layers_source_[0]);
      } 
      input_layer_target_.upper_layer_.InitUpperTransferLayer(false, false, false, NULL, &v_hidden_layers_target_[0]);
    } else {
      // upper layer lies on different GPUs
      if (sequence_to_sequence_mode) {
        input_layer_source_.upper_layer_.InitUpperTransferLayer(false, true, true, NULL, &v_hidden_layers_source_[0]);
      }
      input_layer_target_.upper_layer_.InitUpperTransferLayer(false, true, false, NULL, &v_hidden_layers_target_[0]);

      // bi_dir || multi_source is not written

    }

    
    // Init lower layer for (layer_number - 1) hidden layers
    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {

      // Init lower layer for first hidden layer
      if (0 == i) {
        if (final_gpu_indices[0] == final_gpu_indices[1] && !dropout_mode && !attention_configuration.attention_model_mode_ && false) {
          // lower layer lies on the same GPU
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[0].lower_layer_.InitLowerTransferLayer(true, false, &input_layer_source_, NULL);
          }
          v_hidden_layers_target_[0].lower_layer_.InitLowerTransferLayer(true, false, &input_layer_target_, NULL);
        } else {
          // lower layer lies on different GPUs
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[0].lower_layer_.InitLowerTransferLayer(true, true, &input_layer_source_, NULL);
          }
          v_hidden_layers_target_[0].lower_layer_.InitLowerTransferLayer(true, true, &input_layer_target_, NULL);

          // bi_dir || multi_source is not written
        }
      } else {
        // Init lower layer for hidden layers except for the first hidden layer
        if (final_gpu_indices[i] == final_gpu_indices[i + 1] && !dropout_mode && !attention_configuration.attention_model_mode_ && false) {
          // lower layer lies on the same GPU
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[i].lower_layer_.InitLowerTransferLayer(false, false, NULL, &v_hidden_layers_source_[i-1]);
          }
          v_hidden_layers_target_[i].lower_layer_.InitLowerTransferLayer(false, false, NULL, &v_hidden_layers_target_[i-1]);
        } else {
          // lower layer lies on different GPUs
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[i].lower_layer_.InitLowerTransferLayer(false, true, NULL, &v_hidden_layers_source_[i-1]);
          }
          v_hidden_layers_target_[i].lower_layer_.InitLowerTransferLayer(false, true, NULL, &v_hidden_layers_target_[i-1]);

          // bi_dir || multi_source is not written
        }
      }

      // upper transfer stuff
      if (v_hidden_layers_target_.size() - 1 == i) {
        if (final_gpu_indices[i+1] == final_gpu_indices[i+2] && !dropout_mode && !attention_configuration.attention_model_mode_ && false) {
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[i].upper_layer_.InitUpperTransferLayer(true, false, true, p_softmax_layer_, NULL);
          }
          v_hidden_layers_target_[i].upper_layer_.InitUpperTransferLayer(true, false, false, p_softmax_layer_, NULL);
          // Init lower layer of softmax layer
          p_softmax_layer_->InitLowerTransferLayer(false, false, NULL, &v_hidden_layers_target_[i]);
        } else {
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[i].upper_layer_.InitUpperTransferLayer(true, true, true, p_softmax_layer_, NULL);
          }
          v_hidden_layers_target_[i].upper_layer_.InitUpperTransferLayer(true, true, false, p_softmax_layer_, NULL);
          // Init lower layer of softmax layer
          p_softmax_layer_->InitLowerTransferLayer(false, true, NULL, &v_hidden_layers_target_[i]);

          // bi_dir || multi_source is not written
        }
      } else {
        if (final_gpu_indices[i + 1] == final_gpu_indices[i + 2] && !dropout_mode && !attention_configuration.attention_model_mode_ && false) {
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[i].upper_layer_.InitUpperTransferLayer(false, false, true, NULL, &v_hidden_layers_source_[i + 1]);
          }
          v_hidden_layers_target_[i].upper_layer_.InitUpperTransferLayer(false, false, false, NULL, &v_hidden_layers_target_[i + 1]);
        } else {
          if (sequence_to_sequence_mode) {
            v_hidden_layers_source_[i].upper_layer_.InitUpperTransferLayer(false, true, true, NULL, &v_hidden_layers_source_[i + 1]);
          }
          v_hidden_layers_target_[i].upper_layer_.InitUpperTransferLayer(false, true, false, NULL, &v_hidden_layers_target_[i + 1]);

          // bi_dir || multi_source is not written
        }
      }
    }
  }

#ifdef DEBUG_CHECKPOINT_4
  exit(1);
#endif

  logger<<"\n>> Upper and Lower Transfer stuff was initialized\n";
  logger<<"   (Layer 1) Input layer initialized\n";
  for (int i = 0; i < layers_number - 1; ++i) {
    logger<<"   (Layer "<<(i+2)<<") Hidden layer initialized\n";
  }
  logger<<"   Softmax layer initialized\n";
  //std::cerr.flush();
  PrintGpuInformation();
}



template <typename T>
void NeuralMachineTranslation<T>::InitModelDecoding(int lstm_size, int beam_size, int source_vocab_size, int target_vocab_size, \
                                                    int num_layers, std::string input_weight_file, int gpu_num, GlobalConfiguration &config, \
                                                    bool attention_model_mode, bool feed_input_mode, bool multi_source_mode, \
                                                    bool combine_lstm_mode, bool char_cnn_mode) {
  
  p_softmax_layer_ = new SoftmaxLayer<T>();
  softmax_layer_gpu_information_ = p_softmax_layer_->InitGpu(gpu_num);

  input_layer_source_.input_hidden_layer_information_.Init(gpu_num);
  input_layer_target_.input_hidden_layer_information_ = input_layer_source_.input_hidden_layer_information_;

  decode_mode_ = true;

  // for initializing the model for decoding
  const int longest_sentence = 1;
  const int minibatch_size = 1;
  const bool debug_mode = false;
  const T learning_rate = 0;
  const bool clip_gradients_mode = false;
  const T norm_clip = 0;
  const bool sequence_to_sequence_mode = true;

  const std::string output_weight_file = "NULL";

  // change these for initialization each time
  config.minibatch_size_ = config.beam_size_;
  config.lstm_size_ = lstm_size;

  // need this for source forward prop and for loading in weights correctly
  multi_source_mode_ = multi_source_mode;

  char_cnn_mode_ = char_cnn_mode;

  if (multi_source_mode && attention_model_mode) {
    multi_attention_v2_ = true;
  }

  if (char_cnn_mode) {
    // char cnn is not written //
  }

  logger<<"\n$$ Initialized Model for Decoder of NeuralMT\n";

  PrintGpuInformation();

  // Initialize the softmax layer
  p_softmax_layer_->InitLossLayer(this, config);

  logger<<"\n>> (Layer Softmax) Softmax layer was initialized\n";
  PrintGpuInformation();

  if (sequence_to_sequence_mode) {
    // Initialize the input layer
    input_layer_source_.InitInputToHiddenLayer(lstm_size, minibatch_size, source_vocab_size, longest_sentence, \
                                               debug_mode, learning_rate, clip_gradients_mode, norm_clip, \
                                               this, 101, false, 0, false, false, NULL, false, config, true);

    logger<<"\n>> (Layer 1) Source Input layer was initialized\n";
    PrintGpuInformation();

    if (multi_source_mode) {
      // multi_source_mode is not written //
      ;
    }
  }

  input_layer_target_.InitInputToHiddenLayer(lstm_size, beam_size, target_vocab_size, longest_sentence, \
                                             debug_mode, learning_rate, clip_gradients_mode, norm_clip, \
                                             this, 102, false, 0, false, false, NULL, false, config, false);
  
  logger<<"\n>> (Layer 1) Target Input layer was initialized\n";
  PrintGpuInformation();

  if (attention_model_mode) {
    attention_configuration_.attention_model_mode_ = attention_model_mode;
    attention_configuration_.feed_input_mode_ = feed_input_mode;

    T *p_host_tmp;
    for (int i = 0; i < config.longest_sentence_; ++i) {
      v_top_source_states_.push_back(NULL);
      FullMatrixSetup(&p_host_tmp, &v_top_source_states_.back(), lstm_size, beam_size);

      v_top_source_states_v2_.push_back(NULL);
      FullMatrixSetup(&p_host_tmp, &v_top_source_states_v2_.back(), lstm_size, beam_size);
    }

    if (feed_input_mode) {
      input_layer_target_.DecoderInitFeedInput();
      input_layer_target_.v_nodes_[0].AttentionExtra();
      input_layer_target_.v_nodes_[0].index_ = 1;
    }

    // feed input is always set as false as not transfers should be automatically sent, this is done manually in decoding
    decoder_attention_layer_.InitAttentionDecoder(config.lstm_size_, config.beam_size_, gpu_num, attention_configuration_.d_, \
                                                  config.longest_sentence_, input_layer_source_.input_hidden_layer_information_.handle_, \
                                                  this, false, v_top_source_states_, multi_attention_v2_, v_top_source_states_v2_);

    logger<<"\n$$ (Layer Attention) Attention layer was initialized\n";
    PrintGpuInformation();
  }

  if (multi_source_mode) {
    // multi_source_mode is not written
  }

  input_weight_file_ = input_weight_file;
  output_weight_file_ = output_weight_file;
  truncated_softmax_mode_ = false;
  sequence_to_sequence_mode_ = true;

  //logger<<"\n$$ Memory status after Layer 1 was initialized\n";
  //PrintGpuInformation();

  // do this to be sure addresses stay the same
  for (int i = 1; i < num_layers; ++i) {
    if (sequence_to_sequence_mode) {
      v_hidden_layers_source_.push_back(HiddenToHiddenLayer<T>());
    }

    if (multi_source_mode) {
      // multi_source_mode is not written
    }

    v_hidden_layers_target_.push_back(HiddenToHiddenLayer<T>());
  }

  // now initialize hidden layers
  for (int i = 1; i < num_layers; ++i) {
    if (sequence_to_sequence_mode) {
      v_hidden_layers_source_[i - 1].hidden_hidden_layer_information_ = input_layer_target_.input_hidden_layer_information_;
      v_hidden_layers_source_[i - 1].InitHiddenToHiddenLayer(lstm_size, minibatch_size, longest_sentence, debug_mode, learning_rate, \
                                                             clip_gradients_mode, norm_clip, this, 103, false, 0, false, i);

      logger<<"\n>> (Layer "<<(i+1)<<") Source Hidden layer was initialized\n";
      PrintGpuInformation();

    }

    if (multi_source_mode) {
      // multi_source_mode is not written  
    }

    v_hidden_layers_target_[i - 1].hidden_hidden_layer_information_ = input_layer_target_.input_hidden_layer_information_;
    v_hidden_layers_target_[i - 1].InitHiddenToHiddenLayer(lstm_size, beam_size, longest_sentence, debug_mode, learning_rate, \
                                                           clip_gradients_mode, norm_clip, this, 103, false, 0, false, i);

    logger<<"\n>> (Layer "<<(i+1)<<") Target Hidden layer was initialized\n";
    PrintGpuInformation();
  }

  // now the layer information
  if (num_layers == 1) {
    input_layer_source_.upper_layer_.InitUpperTransferLayer(true, true, true, p_softmax_layer_, NULL);
    input_layer_target_.upper_layer_.InitUpperTransferLayer(true, true, false, p_softmax_layer_, NULL);
    p_softmax_layer_->InitLowerTransferLayer(true, true, &input_layer_target_, NULL);

    if (multi_source_mode) {
      // multi_source_mode is not written
    }
  } else {
    input_layer_source_.upper_layer_.InitUpperTransferLayer(false, true, true, NULL, &v_hidden_layers_source_[0]);
    input_layer_target_.upper_layer_.InitUpperTransferLayer(false, true, false, NULL, &v_hidden_layers_target_[0]);

    if (multi_source_mode) {
      // multi_source_mode is not written
    }

    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      // lower transfer stuff
      if (i == 0) {
        v_hidden_layers_source_[0].lower_layer_.InitLowerTransferLayer(true, true, &input_layer_source_, NULL);
        v_hidden_layers_target_[0].lower_layer_.InitLowerTransferLayer(true, true, &input_layer_target_, NULL);

        if (multi_source_mode) {
          // multi_source_mode is not written
        }
      } else {
        v_hidden_layers_source_[i].lower_layer_.InitLowerTransferLayer(false, true, NULL, &v_hidden_layers_source_[i - 1]);
        v_hidden_layers_target_[i].lower_layer_.InitLowerTransferLayer(false, true, NULL, &v_hidden_layers_target_[i - 1]);

        if (multi_source_mode) {
          // multi_source_mode is not written
        }
      }

      // upper transfer stuff
      if (i == v_hidden_layers_target_.size() - 1) {
        v_hidden_layers_source_[i].upper_layer_.InitUpperTransferLayer(true, true, true, p_softmax_layer_, NULL);
        v_hidden_layers_target_[i].upper_layer_.InitUpperTransferLayer(true, true, false, p_softmax_layer_, NULL);
        p_softmax_layer_->InitLowerTransferLayer(false, true, NULL, &v_hidden_layers_target_[i]);

        if (multi_source_mode) {
          // multi_source_mode is not written
        }
      } else {
        v_hidden_layers_source_[i].upper_layer_.InitUpperTransferLayer(false, true, true, NULL, &v_hidden_layers_source_[i + 1]);
        v_hidden_layers_target_[i].upper_layer_.InitUpperTransferLayer(false, true, false, NULL, &v_hidden_layers_target_[i + 1]);

        if (multi_source_mode) {
          // multi_source_mode is not written
        }
      }
    }
  }

  logger<<"\n>> Upper and Lower Transfer stuff was initialized\n";
  logger<<"   (Layer 1) Input layer initialized\n";
  for (int i = 0; i < num_layers - 1; ++i) {
    logger<<"   (Layer "<<(i+2)<<") Hidden layer initialized\n";
  }
  logger<<"   Softmax layer initialized\n";
  PrintGpuInformation();
}


// CHECK: OK //
template <typename T>
void NeuralMachineTranslation<T>::InitFileInformation(FileHelper *file_information) {
  p_file_information_ = file_information;
}



template <typename T>
void NeuralMachineTranslation<T>::PrintGpuInformation() {
  int origin_device;
  cudaGetDevice(&origin_device);

  int number_devices = -1;
  cudaGetDeviceCount(&number_devices);
  size_t free_bytes, total_bytes = 0;
//  int selected = 0;
  for (int i = 0; i < number_devices; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (0 != i) {
      logger<<"\n";
    }
    cudaSetDevice(i);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    logger<<"   GPU: "<<i<<", "<<prop.name<<", "
          <<(float)(total_bytes - free_bytes)/total_bytes<<" memory used\n";
    logger<<"        "<<total_bytes/(1.0e6)<<" MB total, "
          <<(total_bytes - free_bytes)/(1.0e6)<<" MB used, "
          <<free_bytes/(1.0e6)<<" MB free"<<"\n";
  }
  cudaSetDevice(origin_device);
}


template <typename T>
void NeuralMachineTranslation<T>::InitPreviousStates(int num_layers, int lstm_size, int minibatch_size, int device_number, bool multi_source_mode) {

  cudaSetDevice(device_number);
  for (int i = 0; i < num_layers; ++i) {
    PreviousSourceState<T> previous_source_state;
    previous_source_state.Init(lstm_size);
    v_previous_source_states_.push_back(previous_source_state);

    PreviousTargetState<T> previous_target_state;
    previous_target_state.Init(lstm_size, minibatch_size);
    v_previous_target_states_.push_back(previous_target_state);
    if (multi_source_mode) {
      // multi_source_mode is not written
    }
  }
}



// CHECK: OK //
template <typename T>
void NeuralMachineTranslation<T>::ComputeGradients(int *p_host_input_vocab_indices_source, int *p_host_output_vocab_indices_source, \
                                                   int *p_host_input_vocab_indices_target, int *p_host_output_vocab_indices_target, \
                                                   int current_source_length, int current_target_length,
                                                   int *p_host_input_vocab_indices_source_wgrad, int *p_host_input_vocab_indices_target_wgrad,
                                                   int len_source_wgrad, int len_target_wgrad, int *p_host_sampled_indices, \
                                                   int len_unique_words_trunc_softmax, int *p_host_batch_info, \
                                                   FileHelper *tmp_file_helper) {

#ifdef DEBUG_DROPOUT
  std::cerr<<"\n************DP In *NeuralMachineTranslation* *ComputeGradients*\n"<<std::flush;
  std::cerr<<"   current_source_length: "<<current_source_length<<"\n"
           <<"   current_target_length: "<<current_target_length<<"\n"
           <<"   len_source_wgrad: "<<len_source_wgrad<<"\n"
           <<"   len_target_wgrad: "<<len_target_wgrad<<"\n"
           <<"   len_unique_words_trunc_softmax: "<<len_unique_words_trunc_softmax<<"\n"
           <<std::flush;
#endif

  train_mode_ = true;

  source_length_ = current_source_length;

  // Send the CPU vocab input data to the GPU layers
  // For the input layer, 2 host vectors must be transfered since need special preprocessing for w_gradient
  if (sequence_to_sequence_mode_) {

    // neural MT
    // preprocess the source side of input layer
    input_layer_source_.PreprocessGpuVocabIndices(p_host_input_vocab_indices_source, p_host_input_vocab_indices_source_wgrad, current_source_length, len_source_wgrad);
    for (int i = 0; i<v_hidden_layers_source_.size(); ++i) {
      v_hidden_layers_source_[i].PreprocessGpuVocabIndices(p_host_input_vocab_indices_source, current_source_length);
    }
  }

  // bi_dir is not written

  // multi_source is not written

  // preprocess the target side of input layer
  input_layer_target_.PreprocessGpuVocabIndices(p_host_input_vocab_indices_target, p_host_input_vocab_indices_target_wgrad, current_target_length, len_target_wgrad);
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].PreprocessGpuVocabIndices(p_host_input_vocab_indices_target, current_target_length);
  }

  // preprocess the target side of loss layer
  p_softmax_layer_->PreprocessGpuVocabIndices(p_host_output_vocab_indices_target, current_target_length);

  // char_cnn is not written

#ifdef DEBUG_CHECKPOINT_7
  std::cerr<<"   attention_model_mode_: "<<attention_configuration_.attention_model_mode_<<"\n"<<std::flush;
#endif

  if (attention_configuration_.attention_model_mode_) {
    if (0 == v_hidden_layers_target_.size()) {
      // no hidden layers
      input_layer_target_.p_attention_layer_->PreprocessMinibatchInformation(p_host_batch_info);
    } else {
      // have hidden layers
      v_hidden_layers_target_[v_hidden_layers_target_.size() - 1].p_attention_layer_->PreprocessMinibatchInformation(p_host_batch_info);
    }
  }
  DeviceSyncAll();
  CudaGetLastError("POST INDICES SETUP GPU");

  // Starting source forward
  // std::cerr<<"   (Layer 1) Source input layer starting forward\n"<<std::flush;
  if (sequence_to_sequence_mode_) {
    // Do the source side forward pass for input layer
    // source input layer, v_nodes_[0]
    input_layer_source_.v_nodes_[0].UpdateVectorsForwardGpu(input_layer_source_.p_device_input_vocab_indices_full_, \
                                                            input_layer_source_.p_device_input_vocab_indices_01_full_, \
                                                            input_layer_source_.p_device_init_hidden_vec_, \
                                                            input_layer_source_.p_device_init_cell_vec_);
    input_layer_source_.v_nodes_[0].ForwardProp();

    // multi-GPUs stuff
    // source hidden layers, v_nodes_[0]
    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      // Do the source side forward pass for hidden layers
      v_hidden_layers_source_[i].v_nodes_[0].UpdateVectorsForwardGpu(v_hidden_layers_source_[i].p_device_input_vocab_indices_01_full_, \
                                                                     v_hidden_layers_source_[i].p_device_init_hidden_vector_, \
                                                                     v_hidden_layers_source_[i].p_device_init_cell_vector_);
      v_hidden_layers_source_[i].v_nodes_[0].ForwardProp();
    }

    // bi_dir is not written

#ifdef DEBUG_DROPOUT_2
    std::cerr<<"  current_source_length: "<<current_source_length<<"\n"<<std::flush;
#endif

    for (int i = 1; i < current_source_length; ++i) {

      int step = i * input_layer_source_.minibatch_size_;
      input_layer_source_.v_nodes_[i].UpdateVectorsForwardGpu(input_layer_source_.p_device_input_vocab_indices_full_ + step, input_layer_source_.p_device_input_vocab_indices_01_full_ + step, input_layer_source_.v_nodes_[i - 1].p_device_h_t_, input_layer_source_.v_nodes_[i - 1].p_device_c_t_);
      input_layer_source_.v_nodes_[i].ForwardProp();

      // multi-GPUs stuff
      for (int j = 0; j < v_hidden_layers_source_.size(); ++j) {
        v_hidden_layers_source_[j].v_nodes_[i].UpdateVectorsForwardGpu(v_hidden_layers_source_[j].p_device_input_vocab_indices_01_full_ + step, v_hidden_layers_source_[j].v_nodes_[i - 1].p_device_h_t_, v_hidden_layers_source_[j].v_nodes_[i - 1].p_device_c_t_);
        v_hidden_layers_source_[j].v_nodes_[i].ForwardProp();
      }

      // bi_dir is not written

    }
  }

  // multi_source is not written
  // bi_dir is not written
  // multi_source is not written


  // Do the target side forward pass
  if (!sequence_to_sequence_mode_) {
    // lstm language model
    // input layer target, v_nodes_[0]
    input_layer_target_.v_nodes_[0].UpdateVectorsForwardGpu(input_layer_target_.p_device_input_vocab_indices_full_, input_layer_target_.p_device_input_vocab_indices_01_full_, input_layer_target_.p_device_init_hidden_vec_, input_layer_target_.p_device_init_cell_vec_);

    // multi-GPUs stuff
    // hidden layers target, v_nodes_[0]
    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].v_nodes_[0].UpdateVectorsForwardGpu(v_hidden_layers_target_[i].p_device_input_vocab_indices_01_full_, v_hidden_layers_target_[i].p_device_init_hidden_vector_, v_hidden_layers_target_[i].p_device_init_cell_vector_);
    }
  } else {
    // neural mt
    // previous_source_index is equal to the last nodes in the multi-layers encoder
    int previous_source_index = current_source_length - 1;

    // bi_dir is not written
    // multi_source is not written

    // input layer target, v_nodes_[0]
    input_layer_target_.v_nodes_[0].UpdateVectorsForwardGpu(input_layer_target_.p_device_input_vocab_indices_full_, input_layer_target_.p_device_input_vocab_indices_01_full_, input_layer_source_.v_nodes_[previous_source_index].p_device_h_t_, input_layer_source_.v_nodes_[previous_source_index].p_device_c_t_);

    // multi-GPUs stuff
    // hidden layers target, v_nodes_[0]
    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].v_nodes_[0].UpdateVectorsForwardGpu(v_hidden_layers_target_[i].p_device_input_vocab_indices_01_full_, v_hidden_layers_source_[i].v_nodes_[previous_source_index].p_device_h_t_, v_hidden_layers_source_[i].v_nodes_[previous_source_index].p_device_c_t_);
    }
  }

  // input layer target, v_nodes_[0], forward propagation
  input_layer_target_.v_nodes_[0].ForwardProp();

  // multi-GPUs stuff
  // hidden layers target, v_nodes_[0], forward propagation
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].v_nodes_[0].ForwardProp();
  }

  // multi-GPUs stuff
  p_softmax_layer_->BackPropPreprocessGpuMGpu(0);
  p_softmax_layer_->ForwardProp(0);

  for (int i = 1; i < current_target_length; ++i) {
    int step = i * input_layer_target_.minibatch_size_;
    input_layer_target_.v_nodes_[i].UpdateVectorsForwardGpu(input_layer_target_.p_device_input_vocab_indices_full_ + step, input_layer_target_.p_device_input_vocab_indices_01_full_ + step, input_layer_target_.v_nodes_[i - 1].p_device_h_t_, input_layer_target_.v_nodes_[i - 1].p_device_c_t_);
    input_layer_target_.v_nodes_[i].ForwardProp();

    // multi-GPUs stuff
    for (int j = 0; j < v_hidden_layers_target_.size(); ++j) {
      v_hidden_layers_target_[j].v_nodes_[i].UpdateVectorsForwardGpu(v_hidden_layers_target_[j].p_device_input_vocab_indices_01_full_ + step, \
                                                                     v_hidden_layers_target_[j].v_nodes_[i - 1].p_device_h_t_, \
                                                                     v_hidden_layers_target_[j].v_nodes_[i - 1].p_device_c_t_);
      v_hidden_layers_target_[j].v_nodes_[i].ForwardProp();
    }
    
    // multi-gpus stuff
    p_softmax_layer_->BackPropPreprocessGpuMGpu(step);
    p_softmax_layer_->ForwardProp(i);
  }

  DeviceSyncAll();

  ///////////////////////////////////////////////////////////
  // backward pass, do the backward pass for the target first
  int last_index = current_target_length - 1;

  int step = (current_target_length - 1) * input_layer_target_.minibatch_size_;
  // useless????????
  p_softmax_layer_->BackPropPreprocessGpu(input_layer_target_.v_nodes_[last_index].p_device_h_t_, step);

  // multi-GPUs stuff
  p_softmax_layer_->BackPropPreprocessGpuMGpu(step);
  p_softmax_layer_->BackProp1(current_target_length - 1);

  // multi-GPUs stuff
  for (int i = v_hidden_layers_target_.size() - 1; i >= 0; --i) {
    v_hidden_layers_target_[i].v_nodes_[last_index].BackPropPreprocessGpu(v_hidden_layers_target_[i].p_device_init_d_errn_to_tp1_ht_, v_hidden_layers_target_[i].p_device_init_d_errn_to_tp1_ct_);
    v_hidden_layers_target_[i].v_nodes_[last_index].BackPropGpu(last_index);
  }

  input_layer_target_.v_nodes_[last_index].BackPropPreprocessGpu(input_layer_target_.p_device_init_d_errn_to_tp1_ht_, input_layer_target_.p_device_init_d_errn_to_tp1_ct_);
  input_layer_target_.v_nodes_[last_index].BackPropGpu(last_index);

  for (int i = current_target_length - 2; i >= 0; --i) {

    step = i * input_layer_target_.minibatch_size_;

    // useless????????
    p_softmax_layer_->BackPropPreprocessGpu(input_layer_target_.v_nodes_[i].p_device_h_t_, step);

    // multi-GPUs stuff
    p_softmax_layer_->BackPropPreprocessGpuMGpu(step);
    p_softmax_layer_->BackProp1(i);

    for (int j = v_hidden_layers_target_.size() - 1; j >= 0; --j) {
      v_hidden_layers_target_[j].v_nodes_[i].BackPropPreprocessGpu(v_hidden_layers_target_[j].p_device_d_errn_to_t_htm1_, v_hidden_layers_target_[j].p_device_d_errn_to_t_ctm1_);
      v_hidden_layers_target_[j].v_nodes_[i].BackPropGpu(i);
    }

    input_layer_target_.v_nodes_[i].BackPropPreprocessGpu(input_layer_target_.p_device_d_errn_to_t_ht_m1_, input_layer_target_.p_device_d_errn_to_t_ct_m1_);
    input_layer_target_.v_nodes_[i].BackPropGpu(i);
  }

  // bi_dir is not written

  // multi_source is not written

  // Now do the backward pass for the source
  if (sequence_to_sequence_mode_) {

    // sequence to sequence mode
    int prev_source_index = current_source_length - 1;
    
    // multi-GPUs stuff
    int backprop2_index = 0;
    p_softmax_layer_->BackPropPreprocessGpuMGpu(0);
    p_softmax_layer_->BackProp2(backprop2_index);
    ++backprop2_index;

    // multi-GPUs stuff
    for (int i = v_hidden_layers_source_.size() - 1; i >= 0; --i) {
      // bi_dir is not written

      // multi_source is not written

      v_hidden_layers_source_[i].v_nodes_[prev_source_index].BackPropPreprocessGpu(v_hidden_layers_target_[i].p_device_d_errn_to_t_htm1_, v_hidden_layers_target_[i].p_device_d_errn_to_t_ctm1_);
      v_hidden_layers_source_[i].v_nodes_[prev_source_index].BackPropGpu(prev_source_index);
    }

    // bi_dir is not written

    // multi_source is not written

    input_layer_source_.v_nodes_[prev_source_index].BackPropPreprocessGpu(input_layer_target_.p_device_d_errn_to_t_ht_m1_, input_layer_target_.p_device_d_errn_to_t_ct_m1_);
    input_layer_source_.v_nodes_[prev_source_index].BackPropGpu(prev_source_index);

    // bi_dir is not written

    for (int i = current_source_length - 2; i >= 0; --i) {

      for (int j = v_hidden_layers_source_.size() - 1; j >= 0; --j) {
        v_hidden_layers_source_[j].v_nodes_[i].BackPropPreprocessGpu(v_hidden_layers_source_[j].p_device_d_errn_to_t_htm1_, v_hidden_layers_source_[j].p_device_d_errn_to_t_ctm1_);
        v_hidden_layers_source_[j].v_nodes_[i].BackPropGpu(i);
      }

      input_layer_source_.v_nodes_[i].BackPropPreprocessGpu(input_layer_source_.p_device_d_errn_to_t_ht_m1_, input_layer_source_.p_device_d_errn_to_t_ct_m1_);
      input_layer_source_.v_nodes_[i].BackPropGpu(i);

      // bi_dir is not written



      // multi-GPUs stuff
      if (backprop2_index < current_target_length) {
        int step = backprop2_index * input_layer_target_.minibatch_size_;
        p_softmax_layer_->BackPropPreprocessGpuMGpu(step);
        p_softmax_layer_->BackProp2(backprop2_index);
        ++backprop2_index;
      }
    }

    // multi-GPUs stuff
    for (int i = backprop2_index; i < current_target_length; ++i) {
      int step = backprop2_index * input_layer_target_.minibatch_size_;
      p_softmax_layer_->BackPropPreprocessGpuMGpu(step);
      p_softmax_layer_->BackProp2(backprop2_index);
      ++backprop2_index;
    }

  } else {
    // sequence mode
    // mgpu stuff
    for (int i = 0; i < current_target_length; ++i) {
      int step = i * input_layer_target_.minibatch_size_;
      p_softmax_layer_->BackPropPreprocessGpuMGpu(step);
      p_softmax_layer_->BackProp2(i);
    }
  }

  // multi_source is not written

  if (debug_mode_) {
    grad_check_flag_ = true;
    T epsilon = (T)1e-4;
    DeviceSyncAll();
    CheckAllGradients(epsilon);
    grad_check_flag_ = false;
  }

  // update the model parameter weights
  UpdateWeights();

  ClearGradients();

  DeviceSyncAll();

  if (train_perplexity_mode_) {
    train_perplexity_ += p_softmax_layer_->GetTrainPerplexity();
  }

  train_mode_ = false;
}


template <typename T>
void NeuralMachineTranslation<T>::ClearGradients() {
  DeviceSyncAll();
  if (sequence_to_sequence_mode_) {
    input_layer_source_.ClearGradients(false);
    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      v_hidden_layers_source_[i].ClearGradients(false);
    }
  }

  input_layer_target_.ClearGradients(false);
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].ClearGradients(false);
  }

  p_softmax_layer_->ClearGradients();

  // bi_dir || multi_source is not written

  DeviceSyncAll();
}


template <typename T>
void NeuralMachineTranslation<T>::UpdateWeights() {

  DeviceSyncAll();

#ifdef DEBUG_DROPOUT_4
  std::cerr<<"   global_grad_clip_mode__: "<<global_grad_clip_mode__<<"\n"<<std::flush;
#endif

  if (global_grad_clip_mode__) {
    // for global gradient clipping
    global_norm_clip__ = 0; 
    p_softmax_layer_->CalculateGlobalNorm();

    if (sequence_to_sequence_mode_) {
      input_layer_source_.CalculateGlobalNorm();
      for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
        v_hidden_layers_source_[i].CalculateGlobalNorm();
      }
    }

    input_layer_target_.CalculateGlobalNorm();
    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].CalculateGlobalNorm();
    }

    // bi_dir || multi_source is not written

    DeviceSyncAll();

    global_norm_clip__ = std::sqrt(global_norm_clip__);

    p_softmax_layer_->UpdateGlobalParams();
    source_side_mode__ = true;
    if (sequence_to_sequence_mode_) {
      input_layer_source_.UpdateGlobalParams();
      for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
        v_hidden_layers_source_[i].UpdateGlobalParams();
      }
    }

    source_side_mode__ = false;
    input_layer_target_.UpdateGlobalParams();
    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].UpdateGlobalParams();
    }

    // bi_dir || multi_source is not written

    DeviceSyncAll();
  } else {
    // !global_grad_clip_mode__

    p_softmax_layer_->UpdateWeights();

    source_side_mode__ = true;
    if (sequence_to_sequence_mode_) {
      input_layer_source_.UpdateWeights();

      for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
        v_hidden_layers_source_[i].UpdateWeights();
      }
    }

    source_side_mode__ = false;
    input_layer_target_.UpdateWeights();

    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].UpdateWeights();
    }

    // bi_dir || multi_source is not written
  }

  DeviceSyncAll();

  if (attention_configuration_.attention_model_mode_) {
    if (0 == v_hidden_layers_source_.size()) {
      input_layer_source_.ZeroAttentionError();
      // bi_dir || multi_attention || multi_attention_v2 is not written
    } else {
      v_hidden_layers_source_[v_hidden_layers_source_.size() - 1].ZeroAttentionError();
      // bi_dir || multi_attention || multi_attention_v2 is not written
    }
  }

  DeviceSyncAll();
}


// this function will only be entered in force-decode
// not check
template <typename T>
void NeuralMachineTranslation<T>::DumpAlignments(int target_length, int minibatch_size, int *p_host_input_vocab_indices_source, int *p_host_input_vocab_indices_target) {
  DeviceSyncAll();

  T *p_host_p_t;
  int *p_host_batch_information;
  p_host_p_t = (T *)malloc(minibatch_size * sizeof(T));
  p_host_batch_information = (int *)malloc(minibatch_size * 2 * sizeof(int));

  std::vector<std::vector<int>> v_v_output_indices;
  for (int i = 0; i < minibatch_size * 2; ++i) {
    std::vector<int> v_tmp;
    v_v_output_indices.push_back(v_tmp);
  }

  /* stores in string format 1-3 2-4 4-5, etc.. */
  std::vector<std::string> v_alignment_nums;
  for (int i = 0; i < minibatch_size; ++i) {
    v_alignment_nums.push_back(" ");
  }

  if (0 == v_hidden_layers_target_.size()) {
    cudaMemcpy(p_host_batch_information, input_layer_target_.p_attention_layer_->p_device_batch_information_, minibatch_size * 2 * sizeof(int), cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(p_host_batch_information, v_hidden_layers_target_[v_hidden_layers_target_.size() - 1].p_attention_layer_->p_device_batch_information_, minibatch_size * 2 * sizeof(int), cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < target_length; ++i) {
    if (0 == v_hidden_layers_target_.size()) {
      cudaMemcpy(p_host_p_t, input_layer_target_.p_attention_layer_->v_nodes_[i].p_device_p_t_, minibatch_size * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpy(p_host_p_t, v_hidden_layers_target_[v_hidden_layers_target_.size() - 1].p_attention_layer_->v_nodes_[i].p_device_p_t_, minibatch_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    for (int j = 0; j < minibatch_size; ++j) {
      if (-1 != p_host_input_vocab_indices_target[IDX2C(j, i, minibatch_size)]) {
        v_v_output_indices[0 + 2 * j].push_back(p_host_input_vocab_indices_source[IDX2C(j, (int)p_host_p_t[j] + p_host_batch_information[j + minibatch_size], minibatch_size)]);
        v_v_output_indices[1 + 2 * j].push_back(p_host_input_vocab_indices_target[IDX2C(j, i, minibatch_size)]);
        v_alignment_nums[j] += std::to_string((int)p_host_p_t[j]) + "-" + std::to_string(i) + " ";
      }
    }
  }

  std::cerr<<"Printing alignments\n";
  for (int i = 0; i < minibatch_size; ++i) {
    std::cerr<<v_alignment_nums[i]<<"\n";
  }

  free(p_host_p_t);
  free(p_host_batch_information);

  return;
}


template <typename T>
void NeuralMachineTranslation<T>::UpdateLearningRate(T new_learning_rate) {
  input_layer_source_.learning_rate_ = new_learning_rate;
  input_layer_target_.learning_rate_ = new_learning_rate;
  for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
    v_hidden_layers_source_[i].learning_rate_ = new_learning_rate;
  }
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].learning_rate_ = new_learning_rate;
  }

  // bi_dir || multi_source is not written
  
  p_softmax_layer_->UpdateLearningRate(new_learning_rate);
}


template <typename T>
double NeuralMachineTranslation<T>::GetPerplexity(std::string test_file_name, int minibatch_size, int &test_num_lines_in_file, int longest_sent, int source_vocab_size, int target_vocab_size, bool load_weights_val, int &test_total_words, bool output_log_mode, bool force_decode_mode, std::string fd_filename) {

  std::chrono::time_point<std::chrono::system_clock> begin_force_decoding, end_force_decoding;

  if (load_weights_val) {
    LoadWeights();
  }

  // set trunc softmax to zero always for perplexity
  // Initialize the file information
  FileHelper file_information(test_file_name, minibatch_size, test_num_lines_in_file, longest_sent, source_vocab_size, target_vocab_size, test_total_words, false, 0, 0, char_cnn_config_, char_cnn_config_.char_dev_file_);
  InitFileInformation(&file_information);

  // multi_source is not written

  std::ofstream fd_stream;
  if (force_decode_mode) {
    fd_stream.open(fd_filename);
  }

  int current_epoch = 1;
  
  if (force_decode_mode__) {
    logger<<"\n$$ Force decoding\n";
  }
  logger<<"   Adaptive data     : "<<test_file_name<<"\n";

  double p_data_gpu = 0;
  int num_sents = 0;       // for force decoding
  begin_force_decoding = std::chrono::system_clock::now();
  while (current_epoch <= 1) {

    bool success = file_information.ReadMinibatch();

    // multi_source is not written

    num_sents += file_information.minibatch_size_;
    double tmp = GetError(true);
    fd_stream<<tmp<<"\n";
    p_data_gpu += tmp;

    if (!success) {
      current_epoch += 1;
    }

    if (force_decode_mode__ && (num_sents % 100 == 0)) {
      end_force_decoding = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end_force_decoding - begin_force_decoding;
      logger<<"\r   "<<num_sents<<" sents, "
            <<elapsed_seconds.count() / 60.0<<" minutes, "
            <<num_sents/elapsed_seconds.count()<<" sents/s";
    }
  }

  if (force_decode_mode__) {
    end_force_decoding = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_force_decoding - begin_force_decoding;
    logger<<"\r   "<<num_sents<<" sents, "
          <<elapsed_seconds.count() / 60.0<<" minutes, "
          <<num_sents/elapsed_seconds.count()<<" sents/s"<<"\n";
  }


  // change to base 2 log
  p_data_gpu = p_data_gpu / std::log(2.0);

  double perplexity_gpu = std::pow(2, -1 * p_data_gpu / file_information.total_target_words_);
  logger<<"   sum(log2(p(x_i))) : "<<p_data_gpu<<"\n";
  logger<<"   Total target words: "<<file_information.total_target_words_<< "\n";
  if (force_decode_mode__) {
    logger<<"   Perplexity        : " <<perplexity_gpu<< "\n";
  }

  if (neural_machine_translation::print_partition_function_mode__) {
    neural_machine_translation::PrintPartitionStats();
  }

  return perplexity_gpu;
}


template <typename T>
void NeuralMachineTranslation<T>::ForwardPropSource(int *p_device_input_vocab_indices_source, int *p_device_input_vocab_indices_source_bi, \
                                                    int *p_device_ones, int source_length, int source_length_bi, int lstm_size, \
                                                    int *p_device_char_cnn_indices) {

  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
  }

  DeviceSyncAll();

  cudaSetDevice(input_layer_target_.input_hidden_layer_information_.device_number_);
  input_layer_source_.v_nodes_[0].UpdateVectorsForwardGpu(p_device_input_vocab_indices_source, p_device_ones, \
                                                         input_layer_source_.p_device_init_hidden_vec_, input_layer_source_.p_device_init_cell_vec_);
  input_layer_source_.v_nodes_[0].ForwardProp();

  DeviceSyncAll();

  for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
    v_hidden_layers_source_[i].v_nodes_[0].UpdateVectorsForwardGpu(p_device_ones, v_hidden_layers_source_[i].p_device_init_hidden_vector_, \
                                                                   v_hidden_layers_source_[i].p_device_init_cell_vector_);
    v_hidden_layers_source_[i].v_nodes_[0].ForwardProp();
  }

  DeviceSyncAll();

  if (attention_configuration_.attention_model_mode_) {
    if (v_hidden_layers_source_.size() == 0) {
      for (int i = 0; i < input_layer_target_.minibatch_size_; ++i) {
        CudaErrorWrapper(cudaMemcpy(v_top_source_states_[0] + lstm_size * i, input_layer_source_.v_nodes_[0].p_device_h_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU fprop attention copy decoder source\n");
      }
    } else {
      for (int i = 0; i < input_layer_target_.minibatch_size_; ++i) {
        CudaErrorWrapper(cudaMemcpy(v_top_source_states_[0] + lstm_size * i, v_hidden_layers_source_[v_hidden_layers_source_.size() - 1].v_nodes_[0].p_device_h_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU fprop attention copy decoder source\n");
      }
    }
  }

  DeviceSyncAll();

  for (int i = 1; i < source_length; ++i) {
    CudaErrorWrapper(cudaMemcpy(v_previous_source_states_[0].p_device_h_t_previous_, input_layer_source_.v_nodes_[0].p_device_h_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memory allocation failed s1\n");
    CudaErrorWrapper(cudaMemcpy(v_previous_source_states_[0].p_device_c_t_previous_, input_layer_source_.v_nodes_[0].p_device_c_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memcpy allocation failed s2\n");

    for (int j = 0; j < v_hidden_layers_source_.size(); ++j) {
      CudaErrorWrapper(cudaMemcpy(v_previous_source_states_[j + 1].p_device_h_t_previous_, v_hidden_layers_source_[j].v_nodes_[0].p_device_h_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memory allocation failed s3\n");
      CudaErrorWrapper(cudaMemcpy(v_previous_source_states_[j + 1].p_device_c_t_previous_, v_hidden_layers_source_[j].v_nodes_[0].p_device_c_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memory allocation failed s3\n");
    }

    input_layer_source_.v_nodes_[0].UpdateVectorsForwardGpu(p_device_input_vocab_indices_source + i, p_device_ones, \
                                                            v_previous_source_states_[0].p_device_h_t_previous_, \
                                                            v_previous_source_states_[0].p_device_c_t_previous_);
    input_layer_source_.v_nodes_[0].ForwardProp();

    for (int j = 0; j < v_hidden_layers_source_.size(); ++j) {
      v_hidden_layers_source_[j].v_nodes_[0].UpdateVectorsForwardGpu(p_device_ones, v_previous_source_states_[j + 1].p_device_h_t_previous_, \
                                                                     v_previous_source_states_[j + 1].p_device_c_t_previous_);
      v_hidden_layers_source_[j].v_nodes_[0].ForwardProp();
    }

    DeviceSyncAll();

    if (attention_configuration_.attention_model_mode_) {
      if (v_hidden_layers_source_.size() == 0) {
        for (int j = 0; j < input_layer_target_.minibatch_size_; ++j) {
          CudaErrorWrapper(cudaMemcpy(v_top_source_states_[i] + j * lstm_size, input_layer_source_.v_nodes_[0].p_device_h_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU fprop attention copy decoder source\n");  
        }
      } else {
        for (int j = 0; j < input_layer_target_.minibatch_size_; ++j) {
          CudaErrorWrapper(cudaMemcpy(v_top_source_states_[i] + j * lstm_size, v_hidden_layers_source_[v_hidden_layers_source_.size() - 1].v_nodes_[0].p_device_h_t_, lstm_size * 1 * sizeof(T), cudaMemcpyDeviceToDevice), "GPU fprop attention copy decoder source\n");
        }
      }
    }

    DeviceSyncAll();
  }

  DeviceSyncAll();

  if (multi_source_mode_) {
    // multi_source_mode_ is not written
  }

  if (tsne_dump_mode__) {
    // tsne_dump_mode__ is not written
  }
}


template <typename T>
void NeuralMachineTranslation<T>::DumpSentenceEmbedding(int lstm_size, std::ofstream &out_sentence_embedding) {
  T *p_host_h_t_tmp = (T *)malloc(lstm_size * sizeof(T));
  for (int i = 0; i < v_hidden_layers_source_.size() + 1; ++i) {
    if (0 == i) {
      cudaMemcpy(p_host_h_t_tmp, input_layer_source_.v_nodes_[0].p_device_h_t_, lstm_size * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpy(p_host_h_t_tmp, v_hidden_layers_source_[i - 1].v_nodes_[0].p_device_h_t_, lstm_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    bool first_flag = true;
    for (int j = 0; j < lstm_size; ++j) {
      if (first_flag) {
        out_sentence_embedding<<p_host_h_t_tmp[j];
        first_flag = false;
      } else {
        out_sentence_embedding<<" "<<p_host_h_t_tmp[j];
      }
    }
    out_sentence_embedding<<"\n";
  }
  out_sentence_embedding<<"\n";
  free(p_host_h_t_tmp);
  return;
}




template <typename T>
void NeuralMachineTranslation<T>::ForwardPropTarget(int curr_index, int *p_device_current_indices, int *p_device_ones, int lstm_size, \
                                                    int beam_size, int *p_device_char_cnn_indices) {
  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
  }

  input_layer_target_.v_nodes_[0].index_ = curr_index;

  int num_layers = 1 + v_hidden_layers_target_.size();

  cudaSetDevice(input_layer_target_.input_hidden_layer_information_.device_number_);

  if (curr_index == 0) {
    if (multi_source_mode_) {
      // multi_source_mode_ is not written
    } else {
      input_layer_target_.TransferDecodingStatesGpu(input_layer_source_.v_nodes_[0].p_device_h_t_, \
                                                    input_layer_source_.v_nodes_[0].p_device_c_t_);

      for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
        v_hidden_layers_target_[i].TransferDecodingStatesGpu(v_hidden_layers_source_[i].v_nodes_[0].p_device_h_t_, v_hidden_layers_source_[i].v_nodes_[0].p_device_c_t_);
      }

      input_layer_target_.v_nodes_[0].UpdateVectorsForwardDecoder(p_device_current_indices, p_device_ones);

      for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
        v_hidden_layers_target_[i].v_nodes_[0].UpdateVectorsForwardDecoder(p_device_ones);
      }
    }
  } else {
    input_layer_target_.v_nodes_[0].UpdateVectorsForwardGpu(p_device_current_indices, p_device_ones, \
                                                            v_previous_target_states_[0].p_device_h_t_previous_, \
                                                            v_previous_target_states_[0].p_device_c_t_previous_);

    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].v_nodes_[0].UpdateVectorsForwardGpu(p_device_ones, v_previous_target_states_[i + 1].p_device_h_t_previous_, \
                                                                     v_previous_target_states_[i + 1].p_device_c_t_previous_);
    }
  }

  input_layer_target_.v_nodes_[0].ForwardProp();
  DeviceSyncAll();
  
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].v_nodes_[0].ForwardProp();
  }
  DeviceSyncAll();

  
  // now attention stuff
  if (attention_configuration_.attention_model_mode_) {
    if (num_layers == 1) {
      decoder_attention_layer_.v_nodes_[0].p_device_h_t_ = input_layer_target_.v_nodes_[0].p_device_h_t_;
    } else {
      decoder_attention_layer_.v_nodes_[0].p_device_h_t_ = v_hidden_layers_target_[v_hidden_layers_target_.size() - 1].v_nodes_[0].p_device_h_t_;
    }

    decoder_attention_layer_.v_nodes_[0].ForwardProp();

    DeviceSyncAll();
  }

  if (attention_configuration_.attention_model_mode_) {
    p_softmax_layer_->BackPropPreprocessGpu(decoder_attention_layer_.v_nodes_[0].p_device_final_tmp_2_, 0);
  } else if (num_layers == 1) {
    p_softmax_layer_->BackPropPreprocessGpu(input_layer_target_.v_nodes_[0].p_device_h_t_, 0);
  } else {
    p_softmax_layer_->BackPropPreprocessGpu(v_hidden_layers_target_[v_hidden_layers_target_.size() - 1].v_nodes_[0].p_device_h_t_, 0);
  }

  p_softmax_layer_->GetDistributionGpuDecoderWrapper();
  DeviceSyncAll();

  // copy the h_t and c_t to the previous hidden state of node 0
  cudaSetDevice(input_layer_target_.input_hidden_layer_information_.device_number_);
}



template <typename T>
template <typename Derived>
void NeuralMachineTranslation<T>::SwapDecodingStates(const Eigen::MatrixBase<Derived> &eigen_indices, \
                                                     int index, T *p_device_tmp_swap_vals) {
  input_layer_target_.SwapStatesDecoding(eigen_indices, index, p_device_tmp_swap_vals);

  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].SwapStatesDecoding(eigen_indices, index, p_device_tmp_swap_vals);
  }

  if (attention_configuration_.feed_input_mode_) {
    for (int i = 0; i < input_layer_target_.minibatch_size_; ++i) {
      cudaMemcpy(input_layer_target_.v_nodes_[0].p_device_h_tild_ + i * input_layer_target_.lstm_size_, decoder_attention_layer_.v_nodes_[0].p_device_final_tmp_2_ + eigen_indices(i) * input_layer_target_.lstm_size_, input_layer_target_.lstm_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    }
  }

  DeviceSyncAll();
}


template <typename T>
void NeuralMachineTranslation<T>::TargetCopyPrevStates(int lstm_size, int beam_size) {
  
  cudaSetDevice(input_layer_target_.input_hidden_layer_information_.device_number_);

  CudaErrorWrapper(cudaMemcpy(v_previous_target_states_[0].p_device_h_t_previous_, input_layer_target_.v_nodes_[0].p_device_h_t_, lstm_size * beam_size * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memcpy failed 1\n");
  CudaErrorWrapper(cudaMemcpy(v_previous_target_states_[0].p_device_c_t_previous_, input_layer_target_.v_nodes_[0].p_device_c_t_, lstm_size * beam_size * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memcpy failed 2\n");

  for (int j = 0; j < v_hidden_layers_target_.size(); ++j) {
    CudaErrorWrapper(cudaMemcpy(v_previous_target_states_[j + 1].p_device_h_t_previous_, v_hidden_layers_target_[j].v_nodes_[0].p_device_h_t_, lstm_size * beam_size * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memcpy failed 3\n");
    CudaErrorWrapper(cudaMemcpy(v_previous_target_states_[j + 1].p_device_c_t_previous_, v_hidden_layers_target_[j].v_nodes_[0].p_device_c_t_, lstm_size * beam_size * sizeof(T), cudaMemcpyDeviceToDevice), "GPU memcpy failed 4\n");
  }
}



// Load in the weights from a file, so the model can be used
template <typename T>
void NeuralMachineTranslation<T>::LoadWeights() {

  logger<<"\n$$ Start loading weights\n";
  input_stream_.open(input_weight_file_.c_str());

  // now load the weights by bypassing the intro stuff
  std::string str;
  std::string word;
  std::getline(input_stream_, str);
  std::getline(input_stream_, str);
  while (std::getline(input_stream_, str)) {
    if (str.size() > 3 && '=' == str[0] && '=' == str[1] && '=' == str[2]) {
      break;             // done with source mapping
    }
  }

  if (sequence_to_sequence_mode_) {
    while (std::getline(input_stream_, str)) {
      if (str.size() > 3 && '=' == str[0] && '=' == str[1] && '=' == str[2]) {
        break;           // done with target mapping
      }
    }
  }

  if (sequence_to_sequence_mode_) {
    logger<<"   (Layer 1) Source input layer loading ... ";
    input_layer_source_.LoadWeights(input_stream_);
    logger<<"Done\n";

    if (char_cnn_mode_ && decode_mode_) {
      // char_cnn_mode and decode_mode is not written
    }

    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      logger<<"   (Layer "<<i+2<<") Source hidden layer loading ... ";
      v_hidden_layers_source_[i].LoadWeights(input_stream_);
      logger<<"Done\n";
    }

    logger<<"\n";
  }

  logger<<"   (Layer 1) Target input layer loading ... ";
  input_layer_target_.LoadWeights(input_stream_);
  logger<<"Done\n";

  if (attention_configuration_.feed_input_mode_ && decode_mode_) {
    // feed_input_mode and decode_mode is not written
    logger<<"             Feed input loading ... ";
    input_layer_target_.LoadWeightsDecoderFeedInput(input_stream_);
    logger<<"Done\n";
  }

  if (char_cnn_mode_ && decode_mode_) {
    // char_cnn_mode_ and decode_mode_ is not written
  }

  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    logger<<"   (Layer "<<i+2<<") Target hidden layer loading ... ";
    v_hidden_layers_target_[i].LoadWeights(input_stream_);
    logger<<"Done\n";
  }
  logger<<"\n";

  if (decode_mode_) {
    if (attention_configuration_.attention_model_mode_) {
      logger<<"   (Layer Attention) Attention layer loading ... ";
      decoder_attention_layer_.LoadWeights(input_stream_);
      logger<<"Done\n";
    }
  }

  logger<<"   (Layer Softmax) Softmax layer loading ... ";
  p_softmax_layer_->LoadWeights(input_stream_);
  logger<<"Done\n";

  if (bi_dir_mode_ || multi_source_mode_) {
    // bi_dir_mode_ and multi_source_mode_ is not written
  }

  input_stream_.close();
}

template <typename T>
void NeuralMachineTranslation<T>::DumpBestModel(std::string best_model_name, std::string const_model_name) {

  if (dump_every_best__) {
    best_model_name += "-" + std::to_string(curr_dump_num__);
    curr_dump_num__ += 1;
  }

  logger<<"   Best model file   : "<<best_model_name<<"\n";

  if (boost::filesystem::exists(best_model_name)) {
    boost::filesystem::remove(best_model_name);
  }

  std::ifstream const_model_stream;
  const_model_stream.open(const_model_name.c_str());

  std::ofstream best_model_stream;
  best_model_stream.open(best_model_name.c_str());

  best_model_stream.precision(std::numeric_limits<T>::digits10 + 2);

  // now create the new model file
  std::string str;
  std::string word;
  // first line, parameters
  std::getline(const_model_stream, str);
  best_model_stream<<str<<"\n";
  // second line, =======...
  std::getline(const_model_stream, str);
  best_model_stream<<str<<"\n";

  // first part of vocab, if sequence model, just have this
  while (std::getline(const_model_stream, str)) {
    best_model_stream<<str<<"\n";
    if (str.size() > 3 && '=' == str[0] && '=' == str[1] && '=' == str[2]) {
      break;
    }
  }

  // second part of vocab, if sequence_to_sequence_mode_, have this
  if (sequence_to_sequence_mode_) {
    while (std::getline(const_model_stream, str)) {
      best_model_stream<<str<<"\n";
      if (str.size() > 3 && '=' == str[0] && '=' == str[1] && '=' == str[2]) {
        break;
      }
    }
  }

  // if sequence_to_sequence_mode_, have source input & hidden layers, output parameters
  if (sequence_to_sequence_mode_) {
    input_layer_source_.DumpWeights(best_model_stream);
    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      v_hidden_layers_source_[i].DumpWeights(best_model_stream);
    }
  }

  // output parameters of target input & hidden layers
  input_layer_target_.DumpWeights(best_model_stream);
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].DumpWeights(best_model_stream);
  }

  // output parameters of softmax layer
  p_softmax_layer_->DumpWeights(best_model_stream);

  // bi_dir || multi_source is not written

  best_model_stream.flush();
  best_model_stream.close();
  const_model_stream.close();
}


template <typename T>
void NeuralMachineTranslation<T>::DumpWeights() {
  logger<<"   Model file        : "<<output_weight_file_<<"\n";
  output_stream_.open(output_weight_file_.c_str(), std::ios_base::app);
  output_stream_.precision(std::numeric_limits<T>::digits10 + 2);

  if (sequence_to_sequence_mode_) {
    input_layer_source_.DumpWeights(output_stream_);
    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      v_hidden_layers_source_[i].DumpWeights(output_stream_);
    }
  }

  input_layer_target_.DumpWeights(output_stream_);
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].DumpWeights(output_stream_);
  }

  p_softmax_layer_->DumpWeights(output_stream_);

  // bi_dir || multi_source is not written

  output_stream_.close();
}


template <typename T>
void NeuralMachineTranslation<T>::CheckAllGradients(T epsilon) {
  DeviceSyncAll();
  if (sequence_to_sequence_mode_) {
    logger<<"$$ Checking gradients on source side\n";
    input_layer_source_.CheckAllGradients(epsilon);
    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      v_hidden_layers_source_[i].CheckAllGradients(epsilon);
    }
  }

  logger<<"$$ Checking gradients on target side\n";
  input_layer_target_.CheckAllGradients(epsilon);
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].CheckAllGradients(epsilon);
  }

  p_softmax_layer_->CheckAllGradients(epsilon);

  // bi_dir and multi_source is not written
}


template <typename T>
double NeuralMachineTranslation<T>::GetError(bool gpu_flag) {
  double loss = 0; 
  source_length_ = p_file_information_->current_source_length_;

  if (sequence_to_sequence_mode_) {
    input_layer_source_.PreprocessGpuVocabIndices(p_file_information_->p_host_input_vocab_indices_source_, p_file_information_->p_host_input_vocab_indices_source_wgrad_, p_file_information_->current_source_length_, p_file_information_->length_source_wgrad_);
    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      v_hidden_layers_source_[i].PreprocessGpuVocabIndices(p_file_information_->p_host_input_vocab_indices_source_, p_file_information_->current_source_length_);
    }
  }

  input_layer_target_.PreprocessGpuVocabIndices(p_file_information_->p_host_input_vocab_indices_target_, p_file_information_->p_host_input_vocab_indices_target_wgrad_, p_file_information_->current_target_length_, p_file_information_->length_target_wgrad_);
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].PreprocessGpuVocabIndices(p_file_information_->p_host_input_vocab_indices_target_, p_file_information_->current_target_length_);
  }

  p_softmax_layer_->PreprocessGpuVocabIndices(p_file_information_->p_host_output_vocab_indices_target_, p_file_information_->current_target_length_);

  // char_cnn is not written

  if (attention_configuration_.attention_model_mode_) {
    if (0 == v_hidden_layers_target_.size()) {
      input_layer_target_.p_attention_layer_->PreprocessMinibatchInformation(p_file_information_->p_host_batch_information_);
      // multi_attention_v2 is not written
    } else {
      v_hidden_layers_target_[v_hidden_layers_target_.size() - 1].p_attention_layer_->PreprocessMinibatchInformation(p_file_information_->p_host_batch_information_);
      // multi_attention_v2 is not written
    }
  }

  // bi_dir is not written

  // multi_source is not written

  DeviceSyncAll();
  CudaGetLastError("NeuralMachineTranslation::GetError Post indices setup GetError");

  if (sequence_to_sequence_mode_) {
    input_layer_source_.v_nodes_[0].UpdateVectorsForwardGpu(input_layer_source_.p_device_input_vocab_indices_full_, input_layer_source_.p_device_input_vocab_indices_01_full_, input_layer_source_.p_device_init_hidden_vec_, input_layer_source_.p_device_init_cell_vec_);
    input_layer_source_.v_nodes_[0].ForwardProp();

    // multi-GPUs stuff
    for (int i = 0; i < v_hidden_layers_source_.size(); ++i) {
      v_hidden_layers_source_[i].v_nodes_[0].UpdateVectorsForwardGpu(v_hidden_layers_source_[i].p_device_input_vocab_indices_01_full_, v_hidden_layers_source_[i].p_device_init_hidden_vector_, v_hidden_layers_source_[i].p_device_init_cell_vector_);
      v_hidden_layers_source_[i].v_nodes_[0].ForwardProp();
    }

    // bi_dir is not written

    for (int i = 1; i < p_file_information_->current_source_length_; ++i) {
      int step = i * input_layer_source_.minibatch_size_;
      input_layer_source_.v_nodes_[i].UpdateVectorsForwardGpu(input_layer_source_.p_device_input_vocab_indices_full_ + step, input_layer_source_.p_device_input_vocab_indices_01_full_ + step, input_layer_source_.v_nodes_[i - 1].p_device_h_t_, input_layer_source_.v_nodes_[i - 1].p_device_c_t_);
      input_layer_source_.v_nodes_[i].ForwardProp();

      // multi-GPUs stuff
      for (int j = 0; j < v_hidden_layers_source_.size(); ++j) {
        v_hidden_layers_source_[j].v_nodes_[i].UpdateVectorsForwardGpu(v_hidden_layers_source_[j].p_device_input_vocab_indices_01_full_ + step, v_hidden_layers_source_[j].v_nodes_[i - 1].p_device_h_t_, v_hidden_layers_source_[j].v_nodes_[i - 1].p_device_c_t_);
        v_hidden_layers_source_[j].v_nodes_[i].ForwardProp();
      }

      // bi_dir is not written
    }
  }

  // multi_source is not written

  // bi_dir is not written

  // multi_source is not written

  // Do the target side forward pass
  if (!sequence_to_sequence_mode_) {
    // sequence mode
    input_layer_target_.v_nodes_[0].UpdateVectorsForwardGpu(input_layer_target_.p_device_input_vocab_indices_full_, input_layer_target_.p_device_input_vocab_indices_01_full_, input_layer_target_.p_device_init_hidden_vec_, input_layer_target_.p_device_init_cell_vec_);

    // multi-GPUs stuff
    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].v_nodes_[0].UpdateVectorsForwardGpu(v_hidden_layers_target_[i].p_device_input_vocab_indices_01_full_, v_hidden_layers_target_[i].p_device_init_hidden_vector_, v_hidden_layers_target_[i].p_device_init_cell_vector_);
    }
  } else {

    // bi_dir is not written

    // multi_source is not written

    // sequence to sequence mode
    int prev_source_index = p_file_information_->current_source_length_ - 1;
    input_layer_target_.v_nodes_[0].UpdateVectorsForwardGpu(input_layer_target_.p_device_input_vocab_indices_full_, input_layer_target_.p_device_input_vocab_indices_01_full_, input_layer_source_.v_nodes_[prev_source_index].p_device_h_t_, input_layer_source_.v_nodes_[prev_source_index].p_device_c_t_);

    // multi-GPUs stuff
    for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
      v_hidden_layers_target_[i].v_nodes_[0].UpdateVectorsForwardGpu(v_hidden_layers_target_[i].p_device_input_vocab_indices_01_full_, v_hidden_layers_source_[i].v_nodes_[prev_source_index].p_device_h_t_, v_hidden_layers_source_[i].v_nodes_[prev_source_index].p_device_c_t_);
    }
  }

  input_layer_target_.v_nodes_[0].ForwardProp();

  // multi-GPUs stuff
  for (int i = 0; i < v_hidden_layers_target_.size(); ++i) {
    v_hidden_layers_target_[i].v_nodes_[0].ForwardProp();
  }

  DeviceSyncAll();

  // note p_device_h_t can be null for these as all we need is the vocab pointers corrent for getting the error
  p_softmax_layer_->BackPropPreprocessGpu(input_layer_target_.v_nodes_[0].p_device_h_t_, 0);

  if (gpu_flag) {
    loss += p_softmax_layer_->ComputeLossGpu(0);
  } else {
    std::cerr<<"Error: only use gpu\n";
    exit(EXIT_FAILURE);
  }

  DeviceSyncAll();

  for (int i = 1; i < p_file_information_->current_target_length_; ++i) {
    int step = i * input_layer_target_.minibatch_size_;
    input_layer_target_.v_nodes_[i].UpdateVectorsForwardGpu(input_layer_target_.p_device_input_vocab_indices_full_ + step, input_layer_target_.p_device_input_vocab_indices_01_full_ + step, input_layer_target_.v_nodes_[i - 1].p_device_h_t_, input_layer_target_.v_nodes_[i - 1].p_device_c_t_);

    input_layer_target_.v_nodes_[i].ForwardProp();

    // multi-GPUs stuff
    for (int j = 0; j < v_hidden_layers_target_.size(); ++j) {
      v_hidden_layers_target_[j].v_nodes_[i].UpdateVectorsForwardGpu(v_hidden_layers_target_[j].p_device_input_vocab_indices_01_full_ + step, v_hidden_layers_target_[j].v_nodes_[i - 1].p_device_h_t_, v_hidden_layers_target_[j].v_nodes_[i - 1].p_device_c_t_);
      v_hidden_layers_target_[j].v_nodes_[i].ForwardProp();
    }

    DeviceSyncAll();
    p_softmax_layer_->BackPropPreprocessGpu(input_layer_target_.v_nodes_[i].p_device_h_t_, step);

    if (gpu_flag) {
      loss += p_softmax_layer_->ComputeLossGpu(i);
    } else {
      std::cerr<<"Error: only use gpu\n";
      exit(EXIT_FAILURE);
    }
    DeviceSyncAll();
  }

  if (attention_configuration_.dump_alignments_mode_) {
    DumpAlignments(p_file_information_->current_target_length_, input_layer_target_.minibatch_size_, p_file_information_->p_host_input_vocab_indices_source_, p_file_information_->p_host_input_vocab_indices_target_);
  }
  return loss;
}



////////////////////////// CLASS
////////////////////////// PreviousSourceState
// for decoding multilayer models
template <typename T>
class PreviousSourceState {

public:
  T *p_device_h_t_previous_;
  T *p_device_c_t_previous_;

public:
  PreviousSourceState() {}

public:
  void Init(int lstm_size) {
    cudaMalloc((void **)&p_device_h_t_previous_, lstm_size * 1 * sizeof(T));
    cudaMalloc((void **)&p_device_c_t_previous_, lstm_size * 1 * sizeof(T));
  }
};


////////////////////////// CLASS
////////////////////////// PreviousTargetState
template <typename T>
class PreviousTargetState {

public:
  T *p_device_h_t_previous_;
  T *p_device_c_t_previous_;

public:
  PreviousTargetState() {}

public:
  void Init(int lstm_size, int beam_size) {
    cudaMalloc((void **)&p_device_h_t_previous_, lstm_size * beam_size * sizeof(T));
    cudaMalloc((void **)&p_device_c_t_previous_, lstm_size * beam_size * sizeof(T));
  }
};




} // end of namespace neural_machine_translation



#endif

