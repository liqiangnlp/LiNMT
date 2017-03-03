/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/20/2016
 * Time  : 15:01
 *
 */


#include "word_embedding.h"

namespace neural_machine_translation {

void WordEmbedding::Process(std::string &model_file_name, std::string &source_vocabulary_file_name, std::string &target_vocabulary_file_name) {

  std::ifstream in_model;
  in_model.open(model_file_name.c_str());
  if (!in_model) {
    logger<<"   Error: can not open "<<model_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  std::ofstream out_source_vocabulary;
  std::ofstream out_target_vocabulary;
  out_source_vocabulary.open(source_vocabulary_file_name.c_str());
  out_target_vocabulary.open(target_vocabulary_file_name.c_str());
  out_source_vocabulary<<std::fixed<<std::setprecision(9);
  out_target_vocabulary<<std::fixed<<std::setprecision(9);

  if (!out_source_vocabulary || !out_target_vocabulary) {
    logger<<"   Error: can not write "<<source_vocabulary_file_name<<" or "<<target_vocabulary_file_name<<" file\n";
    exit(EXIT_FAILURE);
  }

  // put in the first line of the model file with the correct information
  // format:
  //    0: num_layers
  //    1: lstm_size
  //    2: target_vocab_size
  //    3: source_vocab_size
  //    4: attention_mode
  //    5: feed_input
  //    6: multi_source
  //    7: combine_lstm
  //    8: char_cnn
  int num_layers;
  int lstm_size;
  int target_vocab_size;
  int source_vocab_size;
  bool attention_mode; 
  bool feed_input_mode;
  bool multi_source_mode; 
  bool combine_lstm_mode;
  bool char_cnn_mode;

  std::vector<std::string> model_params;
  std::string tmp_line;
  std::string tmp_word;

  std::getline(in_model, tmp_line);
  std::istringstream my_ss(tmp_line, std::istringstream::in);
  while (my_ss >> tmp_word) {
    model_params.push_back(tmp_word);
  }

  if (model_params.size() != 9) {
    logger<<"Error: model format is not correct for decoding with file: "<<model_file_name<<"\n";
  }

  num_layers = std::stoi(model_params[0]);
  lstm_size = std::stoi(model_params[1]);
  target_vocab_size = std::stoi(model_params[2]);
  source_vocab_size = std::stoi(model_params[3]);
  attention_mode = std::stoi(model_params[4]);
  feed_input_mode = std::stoi(model_params[5]);
  multi_source_mode = std::stoi(model_params[6]);
  combine_lstm_mode = std::stoi(model_params[7]);
  char_cnn_mode = std::stoi(model_params[8]);

  logger<<"\n$$ Model Information\n"
        <<"   Layers number            : "<<num_layers<<"\n"
        <<"   Lstm size                : "<<lstm_size<<"\n"
        <<"   Target vocabulary size   : "<<target_vocab_size<<"\n"
        <<"   Source vocabulary size   : "<<source_vocab_size<<"\n";

  if (attention_mode) {
    logger<<"   Attention mode           : TRUE\n";
  } else {
    logger<<"   Attention mode           : FALSE\n";
  }

  if (feed_input_mode) {
    logger<<"   Feed input mode          : TRUE\n";
  } else {
    logger<<"   Feed input mode          : FALSE\n";
  }

  if (multi_source_mode) {
    logger<<"   Multi source mode        : TRUE\n";
  } else {
    logger<<"   Multi source mode        : FALSE\n";
  }

  if (combine_lstm_mode) {
    logger<<"   Tree combine lstm mode   : TRUE\n";
  } else {
    logger<<"   Tree combine lstm mode   : FALSE\n";
  }

  if (char_cnn_mode) {
    logger<<"   Char RNN mode            : TRUE\n";
  } else {
    logger<<"   Char RNN mode            : FALSE\n";
  }


  // skip `======'
  std::getline(in_model, tmp_line);


  // source mapping
  std::vector<std::string> v_source_vocabulary;
  while (std::getline(in_model, tmp_line)) {
    if (tmp_line.size() > 3 && '=' == tmp_line[0] && '=' == tmp_line[1] && '=' == tmp_line[2]) {
      break;             // done with source mapping
    } else {
      std::istringstream my_ss(tmp_line, std::istringstream::in);
      while (my_ss >> tmp_word) {
        ;
      }
      v_source_vocabulary.push_back(tmp_word);
    }
  }

  // target mapping
  std::vector<std::string> v_target_vocabulary;
  while (std::getline(in_model, tmp_line)) {
    if (tmp_line.size() > 3 && '=' == tmp_line[0] && '=' == tmp_line[1] && '=' == tmp_line[2]) {
      break;           // done with target mapping
    } else {
      std::istringstream my_ss(tmp_line, std::istringstream::in);
      while (my_ss >> tmp_word) {
        ;
      }
      v_target_vocabulary.push_back(tmp_word);
    }
  }

  GenerateSourceEmbedding(num_layers, lstm_size, source_vocab_size, v_source_vocabulary, in_model, out_source_vocabulary);
  GenerateTargetEmbedding(lstm_size, target_vocab_size, v_target_vocabulary, in_model, out_target_vocabulary);

  in_model.clear();
  in_model.close();
  out_source_vocabulary.clear();
  out_source_vocabulary.close();
  out_target_vocabulary.clear();
  out_target_vocabulary.close();
  return;
}


void WordEmbedding::GenerateSourceEmbedding(int &num_layers, int &lstm_size, int &source_vocab_size, std::vector<std::string> &v_source_vocabulary, \
                                            std::ifstream &in_model, std::ofstream &out_source_vocabulary) {

  ProcessSourceInputLayer(lstm_size, source_vocab_size, v_source_vocabulary, in_model, out_source_vocabulary);
  for (int i = 0; i < num_layers - 1; ++i) {
    ProcessSourceHiddenLayer(lstm_size, in_model);
  }
  return;
}



void WordEmbedding::ProcessSourceInputLayer(int &lstm_size, int &source_vocab_size, std::vector<std::string> &v_source_vocabulary, \
                                            std::ifstream &in_model, std::ofstream &out_source_vocabulary) {
  std::vector<float> v_tmp_mat;
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hi
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_i
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hf
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_f
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hc
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_c
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_ho
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_o

  ReadMatrix(v_tmp_mat, lstm_size, source_vocab_size, in_model);
  for (int i = 0; i < source_vocab_size; ++i) {
    out_source_vocabulary<<v_source_vocabulary.at(i)<<"\t";
    bool first_flag = true;
    for (int j = 0; j < lstm_size; ++j) {
      if (first_flag) {
        out_source_vocabulary<<v_tmp_mat.at(j + i * lstm_size);
        first_flag = false;
      } else {
        out_source_vocabulary<<" "<<v_tmp_mat.at(j + i * lstm_size);
      }
    }
    out_source_vocabulary<<"\n";
  }

  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_i
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_f
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_o
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_c
  return;
}


void WordEmbedding::ProcessSourceHiddenLayer(int &lstm_size, std::ifstream &in_model) {
  std::vector<float> v_tmp_mat;
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hi
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_i
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hf
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_f
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hc
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_c
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_ho
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_o
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_i
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_f
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_o
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_c
  return;
}

void WordEmbedding::GenerateTargetEmbedding(int &lstm_size, int &target_vocab_size, std::vector<std::string> &v_target_vocabulary, \
                                            std::ifstream &in_model, std::ofstream &out_target_vocabulary) {
  ProcessTargetInputLayer(lstm_size, target_vocab_size, v_target_vocabulary, in_model, out_target_vocabulary);
  return;
}


void WordEmbedding::ProcessTargetInputLayer(int &lstm_size, int &target_vocab_size, std::vector<std::string> &v_target_vocabulary, \
                                            std::ifstream &in_model, std::ofstream &out_target_vocabulary) {
  std::vector<float> v_tmp_mat;
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hi
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_i
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hf
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_f
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_hc
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_c
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // w_ho
  ReadMatrix(v_tmp_mat, lstm_size, 1, in_model);             // b_o

  ReadMatrix(v_tmp_mat, lstm_size, target_vocab_size, in_model);
  for (int i = 0; i < target_vocab_size; ++i) {
    out_target_vocabulary<<v_target_vocabulary.at(i)<<"\t";
    bool first_flag = true;
    for (int j = 0; j < lstm_size; ++j) {
      if (first_flag) {
        out_target_vocabulary<<v_tmp_mat.at(j + i * lstm_size);
        first_flag = false;
      } else {
        out_target_vocabulary<<" "<<v_tmp_mat.at(j + i * lstm_size);
      }
    }
    out_target_vocabulary<<"\n";
  }

  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_i
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_f
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_o
  ReadMatrix(v_tmp_mat, lstm_size, lstm_size, in_model);     // m_c
  return;
}



void WordEmbedding::ReadMatrix(std::vector<float> &v_tmp_mat, int rows, int cols, std::ifstream &input_stream) {
  //std::vector<float> v_tmp_mat(rows * cols);
  v_tmp_mat.clear();
  v_tmp_mat.resize(rows * cols);

  std::string tmp_string;
  std::string tmp_token;

  for (int i = 0; i < rows; ++i) {
    std::getline(input_stream, tmp_string);
    std::istringstream iss_input(tmp_string, std::istringstream::in);
    for (int j = 0; j < cols; ++j) {
      iss_input>>tmp_token;
      v_tmp_mat.at(IDX2C(i, j, rows)) = std::stod(tmp_token);
    }
  }
  std::getline(input_stream, tmp_string);
  //v_tmp_mat.clear();
}


} // end of neural_machine_translation namespace



