/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/20/2016
 * Time  : 15:01
 *
 */


#include "replace_vocabulary.h"

namespace neural_machine_translation {

void VocabularyReplacement::Process(std::string &replaced_words_file_name, std::string &input_file_name, std::string &output_file_name) {

  std::ifstream in_replaced_words;
  in_replaced_words.open(replaced_words_file_name.c_str());
  if (!in_replaced_words) {
    logger<<"   Error: can not open "<<replaced_words_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  std::ifstream in_model;
  in_model.open(input_file_name.c_str());
  if (!in_model) {
    logger<<"   Error: can not open "<<input_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  std::ofstream out_model;
  out_model.open(output_file_name.c_str());
  if (!out_model) {
    logger<<"   Error: can not write "<<output_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  // loading replaced words
  logger<<"\n$$ Loading replaced words\n";
  std::unordered_map<int, std::string> uno_src_replaced_words;
  std::unordered_map<int, std::string> uno_tgt_replaced_words;
  int line_num = 0;
  std::string line;
  while(std::getline(in_replaced_words, line)) {
    ++line_num;
    basic_method_.ClearIllegalChar(line);
    std::vector<std::string> v_informations;
    basic_method_.Split(line, '\t', v_informations);
    if (3 != v_informations.size()) {
      logger<<"   Warning: format error in "<<line_num<<" of "<<replaced_words_file_name<<"\n";
    } else {
      if ("SRC" == v_informations.at(0)) {
        uno_src_replaced_words[std::atoi(v_informations.at(1).c_str())] = v_informations.at(2);
      } else if ("TGT" == v_informations.at(0)) {
        uno_tgt_replaced_words[std::atoi(v_informations.at(1).c_str())] = v_informations.at(2);
      } else {
        logger<<"   Warning: format error in "<<line_num<<" of "<<replaced_words_file_name<<"\n";
      }
    }
  }
  in_replaced_words.close();

  // processing the first two lines
  line_num = 0;
  std::getline(in_model, line);
  basic_method_.ClearIllegalChar(line);
  out_model<<line<<"\n";
  std::getline(in_model, line);
  basic_method_.ClearIllegalChar(line);
  out_model << line << "\n";
  line_num += 2;


  // processing the source vocabulary
  logger<<"\n$$ Source vocabulary replacement\n";
  line_num = 0;
  int src_replaced_num = 0;
  while (std::getline(in_model, line)) {
    basic_method_.ClearIllegalChar(line);

    if ('=' == line.at(0) && '=' == line.at(1) && '=' == line.at(2)) {
      out_model<<line<<"\n";
      break;  
    }

    if (0 != uno_src_replaced_words.count(line_num)) {
      ++src_replaced_num;
      out_model<<line_num<<" "<<uno_src_replaced_words[line_num]<<"\n";
    } else {
      out_model<<line<<"\n";
    }
    ++line_num;
  }
  logger<<"   replaced number          : "<<src_replaced_num<<"\n";



  // processing the target vocabulary
  logger<<"\n$$ Target vocabulary replacement\n";
  line_num = 0;
  int tgt_replaced_num = 0;
  while (std::getline(in_model, line)) {
    basic_method_.ClearIllegalChar(line);

    if ('=' == line.at(0) && '=' == line.at(1) && '=' == line.at(2)) {
      out_model<<line<<"\n";
      break;  
    }

    if (0 != uno_tgt_replaced_words.count(line_num)) {
      ++tgt_replaced_num;
      out_model<<line_num<<" "<<uno_tgt_replaced_words[line_num]<<"\n";
    } else {
      out_model<<line<<"\n";
    }
    ++line_num;
  }
  logger<<"   replaced number          : "<<tgt_replaced_num<<"\n";


  // the rest parts
  logger<<"\n$$ Output the rest parameters\n";
  line_num = 0;
  while (std::getline(in_model, line)) {
    ++line_num;
    basic_method_.ClearIllegalChar(line);
    out_model<<line<<"\n";
    if (line_num % 10000 == 0) {
      logger<<"\r   "<<line_num<<" lines";
    }
  }
  logger<<"\r   "<<line_num<<" lines\n";

  in_model.close();
  out_model.close();
  return;
}

} // end of neural_machine_translation namespace



