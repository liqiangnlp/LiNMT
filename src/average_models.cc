/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/20/2016
 * Time  : 15:01
 *
 */


#include "average_models.h"

namespace neural_machine_translation {

void AverageNeuralModels::Process(std::vector<std::string> &v_input_file_names, std::string &output_file_name) {

  std::vector<std::ifstream> v_input_streams(v_input_file_names.size());
  for (int i = 0; i < v_input_file_names.size(); ++i) {
    v_input_streams.at(i).open(v_input_file_names.at(i).c_str());
    if (!v_input_streams.at(i)) {
      logger<<"   Error: can not open "<<v_input_file_names.at(i)<<"\n";
      exit(EXIT_FAILURE);
    }
  }

  std::ofstream output_streams(output_file_name);
  if (!output_streams) {
    logger<<"   Error: can not write "<<output_file_name<<"\n";
    exit(EXIT_FAILURE);
  }
  output_streams<<std::fixed<<std::setprecision(8);

  // Extract information from first input model
  int line_num_of_first_model = 0;
  std::string tmp_line;
  while (std::getline(v_input_streams.at(0), tmp_line)) {
    ++line_num_of_first_model;
  }
  v_input_streams.at(0).clear();
  v_input_streams.at(0).seekg(0, std::ios::beg);

  for (int i = 1; i < v_input_streams.size(); ++i) {
    int line_num_of_other_model = 0;
    std::string tmp_line_other;
    while (std::getline(v_input_streams.at(i), tmp_line_other)) {
      ++line_num_of_other_model;
    }

    if (line_num_of_other_model != line_num_of_first_model) {
      logger<<"   Error: the input models don't have same parameters!\n";
      exit(EXIT_FAILURE);
    }
    v_input_streams.at(i).clear();
    v_input_streams.at(i).seekg(0, std::ios::beg);
  }

  std::vector<std::string> v_first_lines(v_input_streams.size());
  std::getline(v_input_streams.at(0), v_first_lines.at(0));
  for (int i = 1; i < v_input_streams.size(); ++i) {
    std::getline(v_input_streams.at(i), v_first_lines.at(i));
  }

  for (int i = 1; i < v_first_lines.size(); ++i) {
    if (v_first_lines.at(i) != v_first_lines.front()) {
      logger<<"   Error: the input models don't have the same parameters!\n";
      exit(EXIT_FAILURE);
    }
  }
  output_streams<<v_first_lines.at(0)<<"\n";


  std::vector<std::string> v_tmp_lines(v_input_streams.size());
  std::getline(v_input_streams.at(0), v_tmp_lines.at(0));
  for (int i = 1; i < v_input_streams.size(); ++i) {
      std::getline(v_input_streams.at(i), v_tmp_lines.at(i));
  }
  output_streams<<v_tmp_lines.at(0)<<"\n";

  // process source words
  logger<<"\n$$ Process source words\n";
  while (std::getline(v_input_streams.at(0), v_tmp_lines.at(0))) {

    for (int i = 1; i < v_input_streams.size(); ++i) {
      std::getline(v_input_streams.at(i), v_tmp_lines.at(i));
      if (v_tmp_lines.at(i) != v_tmp_lines.at(0)) {
        logger<<"   Error: the input models don't have the same source words!\n";
        exit(EXIT_FAILURE);
      }
    }

    output_streams<<v_tmp_lines.at(0)<<"\n";

    if ('=' == v_tmp_lines.at(0).at(0) && '=' == v_tmp_lines.at(0).at(1) && '=' == v_tmp_lines.at(0).at(2)) {
      break;  
    }
  }

  // process target words
  logger<<"\n$$ Process target words\n";
  while (std::getline(v_input_streams.at(0), v_tmp_lines.at(0))) {

    for (int i = 1; i < v_input_streams.size(); ++i) {
      std::getline(v_input_streams.at(i), v_tmp_lines.at(i));
      if (v_tmp_lines.at(i) != v_tmp_lines.at(0)) {
        logger<<"   Error: the input models don't have the same source words!\n";
        exit(EXIT_FAILURE);
      }
    }

    output_streams<<v_tmp_lines.at(0)<<"\n";

    if ('=' == v_tmp_lines.at(0).at(0) && '=' == v_tmp_lines.at(0).at(1) && '=' == v_tmp_lines.at(0).at(2)) {
      break;  
    }
  }

  // average parameters
  logger<<"\n$$ Average parameters\n";
  while (std::getline(v_input_streams.at(0), v_tmp_lines.at(0))) {
    std::vector<std::vector<std::string> > v_v_parameters(v_input_streams.size());
    basic_method_.Split(v_tmp_lines.at(0), ' ', v_v_parameters.at(0));

    for (int i = 1; i < v_input_streams.size(); ++i) {
      std::getline(v_input_streams.at(i), v_tmp_lines.at(i));
      basic_method_.Split(v_tmp_lines.at(i), ' ', v_v_parameters.at(i));
      if (v_v_parameters.at(i).size() != v_v_parameters.at(0).size()) {
         logger<<"   Error: the input models don't have the same number parameters!\n";
         exit(EXIT_FAILURE);
      }
    }

    if (v_v_parameters.at(0).size() == 0) {
      output_streams<<"\n";
      continue;
    }

    std::vector<float> v_tmp_parameters;
    for (int i = 0; i < v_v_parameters.at(0).size(); ++i) {
      float tmp_paramter = std::atof(v_v_parameters.at(0).at(i).c_str());
      for (int j = 1; j < v_input_streams.size(); ++j) {
        tmp_paramter += std::atof(v_v_parameters.at(j).at(i).c_str());
      }
      tmp_paramter /= v_input_streams.size();
      v_tmp_parameters.push_back(tmp_paramter);
    }

    for (int i = 0; i < v_tmp_parameters.size(); ++i) {
      if (i != 0) {
        output_streams<<" ";
      }
      output_streams<<v_tmp_parameters.at(i);
    }
    output_streams<<"\n";


  }



  for (int i = 0; i < v_input_streams.size(); ++i) {
    v_input_streams.at(i).close();
  }
  output_streams.close();


  /*
  num_layers_ = std::stoi(model_params[0]);
  lstm_size_ = std::stoi(model_params[1]);
  target_vocab_size_ = std::stoi(model_params[2]);
  source_vocab_size_ = std::stoi(model_params[3]);
  attention_model_mode_ = std::stoi(model_params[4]);
  feed_input_mode_ = std::stoi(model_params[5]);
  multi_source_mode_ = std::stoi(model_params[6]);
  combine_lstm_mode_ = std::stoi(model_params[7]);
  char_cnn_mode_ = std::stoi(model_params[8]);
  */


  return;
}

} // end of neural_machine_translation namespace



