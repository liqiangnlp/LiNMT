/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#ifndef CC_UTIL_H_
#define CC_UTIL_H_

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

//#include <Eigen/Core>

#include "debug.h"

namespace cc_util {

void GetFileStats(int &num_lines_in_file, int &total_words, std::ifstream &input_file, int &total_target_words);

void GetFileStatsSource(int &num_lines_in_file, std::ifstream &input);

void AddModelInformation(int num_layers, int lstm_size, int target_vocab_size, int source_vocab_size, bool attention_model, bool feed_input, bool multi_source, bool combine_lstm, bool char_cnn, std::string filename);

}


namespace neural_machine_translation {

class SystemTime {

public:
  const std::string GetCurrentSystemTime();
};


class OutputLogger {

public:
  bool log_output_mode_;
  std::string file_name_;
  std::ofstream out_stream_;

public:
  OutputLogger();

public:
  void InitOutputLogger(std::string file_name, bool log_output_mode);

};


template <typename T>
OutputLogger& operator<< (OutputLogger &out, T t) {
  std::cerr<<t;
  if (out.log_output_mode_) {
    out.out_stream_<<t;
    out.out_stream_.flush();
  }
  return out;
}



//////////////////
class BasicMethod {

public:
  bool Split(const std::string &input_string, const char &split_char, std::vector<std::string> &v_output);

public:
  bool SplitWithString(const std::string &input_string, const std::string &separator, std::vector<std::string> &v_output);

public:
  bool ClearIllegalChar(std::string &str);

public:
  bool RmEndSpace(std::string &line);
  bool RmStartSpace(std::string &line);

public:
  bool ToLower(std::string &line);

public:
  bool RemoveExtraSpace(std::string &input_string, std::string &output_string);
  bool RemoveAllSpace(const std::string &input_string, std::string &output_string);

public:
  std::string size_tToString(size_t &source);

};

} // end of neural_machine_translation namespace


#endif


