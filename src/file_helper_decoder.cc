/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#include "file_helper_decoder.h"

namespace neural_machine_translation {

void FileHelperDecoder::Init(std::string file_name, int &num_lines_in_file, int max_sentence_length) {

  file_name_ = file_name;
  input_file_.open(file_name.c_str(), std::ifstream::in);  // Open the stream to the file
  max_sentence_length_ = max_sentence_length;

  // GPU allocation
  p_host_input_vocab_indices_source_ = (int *)malloc(max_sentence_length * sizeof(int));
  p_host_batch_information_ = (int *)malloc(2 * sizeof(int));

  //int total_words;
  //int target_words;

  //cc_util::GetFileStats(num_lines_in_file, total_words, input_file_, target_words);
  cc_util::GetFileStatsSource(num_lines_in_file, input_file_);
  num_lines_in_file_ = num_lines_in_file;

  // char cnn is not written //
}


FileHelperDecoder::~FileHelperDecoder() {
  free(p_host_input_vocab_indices_source_);
  free(p_host_batch_information_);
  input_file_.close();
}


bool FileHelperDecoder::ReadSentence() {

  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
  }

  bool more_lines_in_file_mode = true;            // returns false when the file is finished
  words_in_sentence_ = 0;                         // for throughput calculation
  std::vector<int> v_tmp_input_sentence_source;   // this stores the source sentence

  std::string tmp_input_source;                   // tmp string for getline to put into
  std::getline(input_file_, tmp_input_source);    // Get the line from the file
  std::istringstream iss_input_source(tmp_input_source, std::istringstream::in);
  std::string word;                               // the tmp word

  int current_tmp_source_input_index = 0;
  while (iss_input_source >> word) {
    v_tmp_input_sentence_source.push_back(std::stoi(word));
    p_host_input_vocab_indices_source_[current_tmp_source_input_index] = std::stoi(word);
    current_tmp_source_input_index += 1;
  }

  words_in_sentence_ = v_tmp_input_sentence_source.size();

  // Now increase current line in file because we have seen two more sentences
  current_line_in_file_ += 1;

  if (current_line_in_file_ > num_lines_in_file_) {
    current_line_in_file_ = 1;
    input_file_.clear();
    input_file_.seekg(0, std::ios::beg);
    more_lines_in_file_mode = false;
    if (char_cnn_mode_) {
      // char_cnn_mode_ is not written
    }
  }

  // Now fill in the minibatch_tokens_input and minibatch_tokens_output
  eigen_minibatch_tokens_source_input_.resize(1, words_in_sentence_);
  sentence_length_ = words_in_sentence_;
  for (int i = 0; i < v_tmp_input_sentence_source.size(); ++i) {
    eigen_minibatch_tokens_source_input_(i) = v_tmp_input_sentence_source[i];
  }

  p_host_batch_information_[0] = sentence_length_;
  p_host_batch_information_[1] = 0;

  return more_lines_in_file_mode;
}


}



