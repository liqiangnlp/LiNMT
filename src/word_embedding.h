/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 01/04/2017
 * Time  : 09:20
 *
 */

#ifndef WORD_EMBEDDING_H_
#define WORD_EMBEDDING_H_

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include "utility_cu.h"
#include "utility_cc.h"


namespace neural_machine_translation {

class WordEmbedding {

public:
  BasicMethod basic_method_;

public:
  void Process(std::string &model_file_name, std::string &source_vocabulary_file_name, std::string &target_vocabulary_file_name);

private:
  void ReadMatrix(std::vector<float> &v_tmp_mat, int rows, int cols, std::ifstream &input_stream);

private:
  void GenerateSourceEmbedding(int &num_layers, int &lstm_size, int &source_vocab_size, std::vector<std::string> &v_source_vocabulary, \
                               std::ifstream &in_model, std::ofstream &out_source_vocabulary);
  void ProcessSourceInputLayer(int &lstm_size, int &source_vocab_size, std::vector<std::string> &v_source_vocabulary, \
                               std::ifstream &in_model, std::ofstream &out_source_vocabulary);
  void ProcessSourceHiddenLayer(int &lstm_size, std::ifstream &in_model);

private:
  void GenerateTargetEmbedding(int &lstm_size, int &target_vocab_size, std::vector<std::string> &v_target_vocabulary, \
                               std::ifstream &in_model, std::ofstream &out_target_vocabulary);
  void ProcessTargetInputLayer(int &lstm_size, int &target_vocab_size, std::vector<std::string> &v_target_vocabulary, \
                               std::ifstream &in_model, std::ofstream &out_target_vocabulary);



};

} // end of neural_machine_translation namespace



#endif


