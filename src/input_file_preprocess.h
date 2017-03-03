/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef INPUT_FILE_PREPROCESS_H_
#define INPUT_FILE_PREPROCESS_H_

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <queue>

#include "utility_cu.h"
#include "utility_cc.h"

namespace neural_machine_translation {

////////////////////////////////////// CombineSentenceInformation //////////////////////////////////////
////////////////////////////////////// CombineSentenceInformation //////////////////////////////////////
////////////////////////////////////// CombineSentenceInformation //////////////////////////////////////
class CombineSentenceInformation {

public:
  std::vector<std::string> v_source_sentence_;
  std::vector<std::string> v_target_sentence_;

  std::vector<int> v_source_sentence_int_;
  std::vector<int> v_minus_two_source_;
  std::vector<int> v_target_sentence_int_i_;
  std::vector<int> v_target_sentence_int_o_;

  int total_length_;

public:
  CombineSentenceInformation(std::vector<std::string> &v_source_sentence, std::vector<std::string> &v_target_sentence){
    v_source_sentence_ = v_source_sentence;
    v_target_sentence_ = v_target_sentence;
    total_length_ = v_target_sentence_.size() + v_source_sentence_.size();
  }
};


////////////////////////////////////// CompareNeuralMT //////////////////////////////////////
////////////////////////////////////// CompareNeuralMT //////////////////////////////////////
////////////////////////////////////// CompareNeuralMT //////////////////////////////////////
class CompareNeuralMT {

public:
  bool operator() (const CombineSentenceInformation &first, const CombineSentenceInformation &second) {
    return first.total_length_ < second.total_length_;
  }
};



////////////////////////////////////// MappingPair //////////////////////////////////////
////////////////////////////////////// MappingPair //////////////////////////////////////
////////////////////////////////////// MappingPair //////////////////////////////////////
class MappingPair {

public:
  std::string word_;
  int count_;

public:
  MappingPair(std::string word, int count) {
    word_ = word;
    count_ = count;
  }
};


////////////////////////////////////// MappingPairCompareFunctor //////////////////////////////////////
////////////////////////////////////// MappingPairCompareFunctor //////////////////////////////////////
////////////////////////////////////// MappingPairCompareFunctor //////////////////////////////////////
class MappingPairCompareFunctor {

public:
  bool operator() (MappingPair &a, MappingPair &b) const {
    return (a.count_ < b.count_);
  }
};



////////////////////////////////////// InputFilePreprocess //////////////////////////////////////
////////////////////////////////////// InputFilePreprocess //////////////////////////////////////
////////////////////////////////////// InputFilePreprocess //////////////////////////////////////
/* this will unk based on the source and target vocabulary */
class InputFilePreprocess {

public:
  std::ifstream source_input_;
  std::ifstream target_input_;
  std::ofstream final_output_;

public:
  std::unordered_map<std::string, int> uno_source_mapping_;
  std::unordered_map<std::string, int> uno_target_mapping_;


public:
  std::unordered_map<int, std::string> uno_target_reverse_mapping_;
  std::unordered_map<int, std::string> uno_source_reverse_mapping_;

public:
  std::unordered_map<std::string, int> uno_source_counts_;
  std::unordered_map<std::string, int> uno_target_counts_;

public:
  int minibatch_mult_ = 10;      // montreal uses 20

public:
  std::vector<CombineSentenceInformation> data_;

public:
  BasicMethod basic_method_;

public:
  bool PreprocessFilesTrainNeuralLM(int minibatch_size, int max_sent_cutoff, std::string target_file_name, std::string output_file_name, \
                                    int &target_vocab_size, bool shuffle_flag, std::string model_output_file_name, \
                                    int hiddenstate_size, int num_layers);

public:
  bool PreprocessFilesTrainNeuralMT(int minibatch_size, int max_sent_cutoff, \
                                    std::string source_file_name, std::string target_file_name, std::string output_file_name, \
                                    int &source_vocab_size, int &target_vocab_size, \
                                    bool shuffle_flag, std::string model_output_file_name, \
                                    int hiddenstate_size, int num_layers, bool unk_replace_mode, int unk_align_range, bool attention_mode);

public:
  bool IntegerizeFileNeuralLM(std::string output_weights_name, std::string target_file_name, std::string tmp_output_name, \
                              int max_sent_cutoff, int minibatch_size, bool dev_flag, int &hiddenstate_size, \
                              int &target_vocab_size, int &num_layers);


public:
  bool IntegerizeFileNeuralMT(std::string output_weights_name, std::string source_file_name, std::string target_file_name, \
                              std::string tmp_output_name, int max_sent_cutoff, int minibatch_size, int &hiddenstate_size, \
                              int &source_vocab_size, int &target_vocab_size, int &num_layers, bool attention_mode, \
                              bool multi_source_mode, std::string multi_source_file, std::string tmp_output_name_ms, std::string ms_mapping_file);

public:
  bool IntegerizeFileDecoding(std::string output_weights_name, std::string source_file_name, std::string tmp_output_name, \
                              int max_sent_cutoff, int &target_vocab_size, bool multi_source_model, std::string multi_source_mapping_file);

public:
  void UnintFile(std::string output_weights_name, std::string unint_file, std::string output_final_name, bool sequence_to_sequence_mode, bool decoder_mode);

};


}

#endif


