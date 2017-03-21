/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#ifndef GLOBAL_CONFIGURATION_H_
#define GLOBAL_CONFIGURATION_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>


#include "boost/program_options.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"


#include "utility_cu.h"
#include "utility_cc.h"
#include "input_file_preprocess.h"


#define GPU_MODE
#define EIGEN



namespace neural_machine_translation {


class AttentionConfiguration {

public:  
  bool attention_model_mode_ = false;
  bool feed_input_mode_ = false;
  bool dump_alignments_mode_ = false;

public:
  int d_ = 10;

public:
  std::string tmp_alignment_file_name_ = "NULL";
  std::string alignment_file_name_ = "alignments.txt";
};


class MultiSourceParams {
public:
  bool multi_source_mode_ = false;
  bool multi_attention_mode_ = false;
  bool multi_attention_v2_mode_ = false;
  bool add_ht_mode_ = false;                 // add the hidden states instead of sending them through a neural network
  bool lstm_combine_mode_ = false;
  std::string file_name_ = "NULL";           // this is for the training data for the additional file
  std::string int_file_name_ = "/multi_source.txt";  // the integerized file name in the booth path
  std::string source_model_name_ = "NULL";   // specified by the user
  std::string int_file_name_test_ = "/validation_multi.txt";
  std::string test_file_name_ = "NULL";      // specified by the user
  std::string ensemble_train_file_name_ = "NULL";
};


class CharCnnConfiguration {
public:
  bool char_cnn_mode_ = false;

  std::string char_train_file_ = "train_char.txt.brz";
  std::string char_dev_file_ = "dev_char.txt.brz";
};


class GlobalConfiguration {

public:
  std::string unique_dir_ = "NULL";          // for file system cleanup

public:
  bool load_model_train_mode_ = false;       // for restarting model training
  std::string load_model_name_ = "NULL";     // for continue-train

public:
  // for training a model with the same indices as another models for ensembles
  std::string ensemble_train_file_name_ = "NULL";
  bool ensemble_train_mode_ = false;

public:
  bool dropout_mode_ = false;                // for dropout
  precision dropout_rate_ = 1.0;

public:
  bool random_seed_mode_ = false;
  int random_seed_int_ = -1;

public:
  std::string tmp_location_ = "";  // location where tmp directory will be created

public:
  // for char_cnn
  CharCnnConfiguration char_cnn_config_;

public:
  AttentionConfiguration attention_configuration_;      // attention model

public:
  // individual gradient clipping
  bool individual_gradient_clip_mode_ = false;
  precision individual_norm_clip_threshold_ = 0.1;

public:
  bool clip_gradient_mode_ = true;       // gradient clipping whole matrices
  precision norm_clip_ = 5.0;  // Renormalize the gradients so they fit in normball this size, this is also used for total threshold

public:
  // Loss Functions
  bool softmax_mode_ = true;
  bool nce_mode_ = false;
  int negative_samples_number_ = 500;
  bool share_samples_mode_ = true;

public:
  // unk replacement
  bool unk_replace_mode_ = false;
  int unk_aligned_width_ = 7;

public:
  MultiSourceParams multi_source_params_;

public:
  static const bool debug_mode_ = false;

public:
  bool training_mode_ = true;                // If you want to train the model
  bool test_mode_ = false;                   // If you want to test the model
  bool decode_mode_ = false;                 // If you want to decode
  bool postprocess_unk_mode_ = false;
  bool calculate_bleu_mode_ = false;
  bool average_models_mode_ = false;
  bool vocabulary_replacement_mode_ = false;
  bool dump_word_embedding_mode_ = false;
  bool training_perplexity_mode_ = true;     // Print out the train perplexity every epoch (or half epoch if you have a learning rate schedule)

public:
  bool train_bpe_mode_ = false;              // If you want to train bpe model
  bool segment_bpe_mode_ = false;            // use bpe codes to segment
  int bpe_vocabulary_size_ = 1000;           // Default 1000
  int bpe_min_frequency_ = 2;                // Default 2
  std::string bpe_input_codes_file_name_;    // Input codes file name
  std::string bpe_input_file_name_;          // Input file name
  std::string bpe_output_file_name_;         // Output file name
  
public:
  bool sequence_to_sequence_mode_ = true;    // If true it is only a sequence-to-sequence model, not sequence
  bool shuffle_mode_ = true;                 // shuffle the training data

public:
  bool stochastic_generation_mode_ = false;  // This is for language modeling only
  int stochastic_generation_length_ = 10;    // how many tokens to generate
  double stochastic_generation_temperature_ = 1.0;
  
public:
  bool output_log_mode_ = true;              // flush the output to a file, so it can be read as the program executes
  std::string output_log_file_name_ = "neural-log.txt";

public:
  int minibatch_size_ = 8;                 // size of the minibatch
  int epochs_number_ = 10;                   // number passes through the dataset
  precision learning_rate_ = 0.5;            // the learning rate for SGD

public:
  bool google_learning_rate_ = false;        // stuff for the google learning rate, this halves the learning rate every 0.5 epochs after some initial epoch
  int epoch_to_start_halving_ = 6;           // after what epoch do you halve the learning rate
  int half_way_count_ = -1;                  // what is the total number of words that mark half an epoch
  
public:
  bool stanford_learning_rate_ = false;      // stuff for the stanford learning rate
  precision stanford_decrease_factor_ = 0.5;
  int epoch_to_start_halving_full_ = 6;

public:
  // stuff for normal halving of the learning rate where every half epoch the validation set is looked at and if it didn't improve, or did worse, the learning rate is halved.
  bool learning_rate_schedule_ = false;
  precision decrease_factor_ = 0.5;
  double margin_ = 0.0;
  std::string dev_source_file_name_;
  std::string dev_target_file_name_;

public:
  bool softmax_scaled_mode_ = true;

public:
  // the truncated softmax
  // top_fixed + sampled = target vocabulary
  bool truncated_softmax_mode_ = false;
  int shortlist_size_ = 10000;
  int sampled_size_ = 5000;
  
public:
  // model size information
  // vocab size of -1 defaults to the size of the train file specified
  int source_vocab_size_ = -1;   
  int target_vocab_size_ = -1;    // size in input vocabulary, ranging from 0-input_vocab_size, where 0 is start symbol
  
public:
  int lstm_size_ = 100;           // lstm cell size, by definition it is the same as the word embedding layer
  int layers_number_ = 1;         // this is the number of stacked lstm's in the model
  
public:
  std::vector<int> gpu_indices_;  // for training with multiple gpu's
  
public:
  // Decoder Settings
  int beam_size_ = 12;
  precision penalty_ = 0;
  int hypotheses_number_ = 1;           // This prints out the k best paths from the beam decoder for the input
  precision min_decoding_ratio_ = 0.5;  // target translation divided by source sentence length must be greater than min_decoding_ratio
  precision max_decoding_ratio_ = 1.5;
  precision decoding_lp_alpha_ = 0.65;
  precision decoding_cp_beta_ = 0.20;
  precision decoding_diversity_ = 0;
  bool decoding_dump_sentence_embedding_mode_ = false;
  std::string decoding_sentence_embedding_file_name_ = "sentence-embedding.txt";
  bool finetune_mode_ = false;
  bool print_decoding_information_mode_ = false;
  bool print_alignments_scores_mode_ = false;
  //bool print_score_mode_ = false;       // Whether to print the score of the hypotheses or not
  //bool print_alignment_mode_ = false;
  //bool print_unk_alignment_mode_ = false;

public:
  bool decode_sentence_mode_ = false;        // If you want to decode sentence by sentence
  std::string decode_sentence_config_file_;
  std::string decode_sentence_input_file_;
  std::string decode_sentence_output_file_;

public:
  //std::vector<std::string> bpe_parameters_;                // parameters for bpe

public:
  std::vector<std::string> decode_user_files_;             // source file being decoded 
  std::vector<std::string> decode_user_files_additional_;  // source file being decoded
  std::vector<std::string> decode_tmp_files_;              // one for each model being decoded
  std::vector<std::string> decode_tmp_files_additional_;   // one for each model being decoded
  std::string decoder_output_file_ = "NULL";               // decoder output in tmp before integerization
  std::vector<std::string> model_names_;                   // for kbest ensembles
  std::vector<std::string> model_names_multi_source_;      // NULL value represents not using one

  std::string decoder_final_file_;                         // what to output the final outputs to for decoding

  // this is the file for outputting the hidden, cell states, etc.
  // format
  // 1. xt=a, embedding
  // 2. forget gate
  // 3. input gate
  // 4. c_t
  // 5. output
  // 6. h_t
  // 7. probabilities
  bool dump_lstm_mode_ = false;
  std::string lstm_dump_file_;

public:
  int screen_print_rate_ = 20;     // for printing stuff to the screen

public:
  std::string best_model_file_name_;     // for saving the best model for training
  bool best_model_mode_ = false;
  double best_model_perp_ = DBL_MAX;
  
public:
  // I/O File Information
  std::string source_file_name_;         // for training, kbest, force decode and sg
  std::string target_file_name_;         // for training, kbest, force decode and sg
  std::string output_force_decode_;

  int longest_sentence_ = 100;           // Note this doubles when doing translation, it is really 4 less than it is

  std::string train_file_name_ = "NULL"; // Integerized training file, Input file where source is first line then target is second line
  int train_num_lines_in_file_ = -1;     // This is learned
  int train_total_words_ = -1;           // This is learned

  std::string test_file_name_ = "NULL";  // Input file where source is first line then target is second line
  int test_num_lines_in_file_ = -1;      // This is learned
  int test_total_words_ = -1;            // This is learned
  
public:
  std::string input_weight_file_ = "model.nn";  // for weights
  std::string output_weight_file_ = "model.nn";


public:
  std::string unk_config_file_name_;
  std::string unk_dict_file_name_;
  std::string unk_source_file_name_;
  std::string unk_1best_file_name_;
  std::string unk_output_file_name_;
  std::string unk_stopword_file_name_;
  bool unk_output_oov_mode_ = false;

public:
  std::string bleu_one_best_file_name_;
  std::string bleu_dev_file_name_;
  int bleu_number_references_;
  bool bleu_remove_oov_mode_ = false;
  std::string bleu_output_file_name_;

public:
  std::vector<std::string> v_average_models_input_file_;
  std::string average_models_output_file_;

public:
  std::string vocab_replace_config_file_name_;
  std::string vocab_replace_model_file_name_;
  std::string vocab_replace_output_file_name_;

public:
  std::string word_embedding_model_file_name_;
  std::string word_embedding_source_vocab_file_name_;
  std::string word_embedding_target_vocab_file_name_;

public:
  void ParseCommandLine(int argc, char **argv);


public:
  void NormalSettingsAndChecking(boost::program_options::options_description &description, boost::program_options::variables_map &v_map, \
                                 std::vector<precision> &lower_upper_range, std::vector<precision> &clip_cell_values);

public:
  void TuningAndContinueTuning(boost::program_options::variables_map &v_map, std::vector<std::string> &training_files, \
                               std::vector<std::string> &continue_train, std::vector<int> &gpu_indices, \
                               std::vector<precision> &lower_upper_range, std::vector<std::string> &adaptive_learning_rate, \
                               std::vector<std::string> &multi_source, std::vector<std::string> &truncated_information);
  void TuningErrorSettings(boost::program_options::variables_map &v_map);

public:
  void ContinueTuning(boost::program_options::variables_map &v_map, std::vector<std::string> &continue_train);
  void Tuning(boost::program_options::variables_map &v_map, std::vector<std::string> &training_files);

public:
  // Settings for tuning and continue-tuning
  void FinetuneSettings();
  void BasicErrorChecking(boost::program_options::variables_map &v_map);
  void LearningMethodSettings(boost::program_options::variables_map &v_map, std::vector<std::string> &adaptive_learning_rate);
  void ParameterRangeSettings(boost::program_options::variables_map &v_map, std::vector<precision> &lower_upper_range);
  void TruncatedSoftmaxSettings(boost::program_options::variables_map &v_map, std::vector<std::string> &truncated_information);
  void ClipGradientsSettings(boost::program_options::variables_map &v_map);

public:
  void Decoding(boost::program_options::variables_map &v_map, std::vector<std::string> &kbest_files, \
                std::vector<precision> &decoding_ratio, std::vector<int> &gpu_indices);
  void DecodingSentence(std::vector<std::string> &v_decoding_sentences);


public:
  void ForceDecoding(boost::program_options::variables_map &v_map, std::vector<std::string> &test_files, std::vector<int> &gpu_indices);

public:
  void PostProcessUnk(boost::program_options::variables_map &v_map, std::vector<std::string> &v_postprocess_unk);

public:
  void CalculateBleuScore(boost::program_options::variables_map &v_map, std::vector<std::string> &v_bleu_score);

public:
  void AverageModels(boost::program_options::variables_map &v_map, std::vector<std::string> &v_average_models);

public:
  void ReplaceVocab(boost::program_options::variables_map &v_map, std::vector<std::string> &v_vocabulary_replacement);

public:
  void DumpWordEmbedding(boost::program_options::variables_map &v_map, std::vector<std::string> &v_words_embeddings);

public:
  void TrainBytePairEncoding(boost::program_options::variables_map &v_map, std::vector<std::string> &v_parameters_of_bpe);
  void SegmentBytePairEncoding(boost::program_options::variables_map &v_map, std::vector<std::string> &v_parameters_of_bpe);

public:
  void OptionsSetByUser(boost::program_options::variables_map &v_map);
  void AddOptions(boost::program_options::options_description &description, std::vector<std::string> &training_files, \
                  std::vector<std::string> &continue_train, std::vector<std::string> &test_files, \
                  std::vector<std::string> &kbest_files, std::vector<std::string> &v_decoding_sentences, \
                  std::vector<std::string> &v_postprocess_unk, std::vector<precision> &decoding_ratio, \
                  std::vector<int> &gpu_indices, std::vector<precision> &lower_upper_range, \
                  std::vector<std::string> &adaptive_learning_rate, std::vector<precision> &clip_cell_values, \
                  std::vector<std::string> &v_bleu_score, std::vector<std::string> &v_average_models, \
                  std::vector<std::string> &v_vocabulary_replacement, std::vector<std::string> &v_words_embeddings, \
				  std::vector<std::string> &v_parameters_of_bpe);



public:
  void PrintParameters();
  void PrintDecodingParameters();
  void PrintDecodingSentParameters();

};

}


#endif

