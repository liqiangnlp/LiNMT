/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#ifndef FILE_HELPER_H_
#define FILE_HELPER_H_

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <iostream>

#include <boost/random/uniform_real.hpp>
#include <boost/random.hpp>

#include "utility_cu.h"
#include "global_configuration.h"
#include "debug.h"




namespace neural_machine_translation {

class FileHelper {

public:
  std::string file_name_;       // input file name
  int minibatch_size_;          // size of minibatches
  std::ifstream input_file_;    // input file stream
  int current_line_in_file_ = 1;
  int nums_lines_in_file_;

public:
  int words_in_minibatch_;      // used for computing the maximum sentence length of previous minibatch

  /*
public:
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_source_input_;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_source_output_;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_target_input_;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_target_output_;
  */

public:
  // GPU parameters
  // This is for storing the vocab indices on the GPU
  int max_sentence_length_;      // max sentence length
  int current_source_length_;
  int current_target_length_;
  int source_vocab_size_;
  int target_vocab_size_;

  // host pointers
  int *p_host_input_vocab_indices_source_;                 // lstm_size_ x longest_sent, column major
  int *p_host_output_vocab_indices_source_;                // lstm_size_ x longest_sent, column major

  int *p_host_input_vocab_indices_target_;                 // lstm_size_ x longest_sent, column major
  int *p_host_output_vocab_indices_target_;                // lstm_size_ x longest_sent, column major

  int *p_host_input_vocab_indices_source_tmp_;             // lstm_size_ x longest_sent, row major
  int *p_host_output_vocab_indices_source_tmp_;            // lstm_size_ x longest_sent, row major
  int *p_host_input_vocab_indices_target_tmp_;             // lstm_size_ x longest_sent, row major
  int *p_host_output_vocab_indices_target_tmp_;            // lstm_size_ x longest_sent, row major

public:
  // These are the special vocab indices for the w gradient updates
  int *p_host_input_vocab_indices_source_wgrad_;           // minibatch_size_ x longest_sent_
  int *p_host_input_vocab_indices_target_wgrad_;

public:
  bool *p_bitmap_source_;       // This is for preprocessing the input vocab for quick updates on the w gradient
  bool *p_bitmap_target_;       // This is for preprocessing the input vocab for quick updates on the w gradient

public:
  // length for the special w gradient stuff
  int length_source_wgrad_;
  int length_target_wgrad_;
  
public:
  // host devices
  // for the attention model
  int *p_host_batch_information_;

public:
  // for perplexity
  int total_target_words_;

  bool truncated_softmax_mode_;
  int shortlist_size_;
  int sampled_size_;
  int len_unique_words_trunc_softmax_;             // use the sample rate for words above this index
  int *p_host_sampled_indices_;                    // size of sampled size, for truncated softmax
  std::unordered_map<int, int> resevoir_mapping_;  // stores mapping for word in vocab to row number in output distribution/weight matrices
  


public:
  FileHelper(std::string fn, int minibatch_size, int &num_lines_in_file, int longest_sentence, int source_vocab_size, int target_vocab_size, int &total_words, bool truncated_softmax_mode, int shortlist_size, int sample_size, CharCnnConfiguration &char_cnn_config, std::string char_file);
  ~FileHelper();

public:
  bool ReadMinibatch();


private:
  void ZeroBitmaps();
  void PreprocessInputWgrad();
  void PreprocessOutputTruncatedSoftmax();

};


}

#endif




