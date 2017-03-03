/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef FILE_HELPER_DECODER_H_
#define FILE_HELPER_DECODER_H_

#include <string>
#include <fstream>
#include <iostream>

#include "utility_cc.h"

#include <Eigen/Core>



#include "debug.h"


namespace neural_machine_translation {

class FileHelperDecoder {
  
public:
  std::string file_name_;       // name of input file
  std::ifstream input_file_;    // input file stream
  int current_line_in_file_ = 1;
  int num_lines_in_file_;

public:
  // used for computing the maximum sentence length of previous minibatch
  int words_in_sentence_;
  int sentence_length_;         // the length of the current source sentence

  int max_sentence_length_;     // the max length for a source sentence

  // num rows is the length of minibatch, num columns is length of longest sentence
  // unused positions are padded with -1, since that is not a valid token
  Eigen::Matrix<int, 1, Eigen::Dynamic> eigen_minibatch_tokens_source_input_;     // 1 x word_in_sentences_

public:
  int *p_host_input_vocab_indices_source_;     // max_sentence_length
  int *p_host_batch_information_;              // 2

  bool char_cnn_mode_ = false;
  
public:
  // constructor
  FileHelperDecoder() {}
  ~FileHelperDecoder();

public:
  void Init(std::string file_name, int &num_lines_in_file, int max_sentence_length);


public:
  bool ReadSentence();

};

}

#endif


