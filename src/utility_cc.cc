/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#include "utility_cc.h"

namespace cc_util {

// CHECK: OK //
// counts the total number of words in file format, so you can halve learning rate at half epochs
// counts the total number of lines
void GetFileStats(int &num_lines_in_file, int &total_words, std::ifstream &input_file, int &total_target_words) {
  
  std::string str;
  std::string word;
  num_lines_in_file = 0;
  total_words = 0;
  total_target_words = 0;
  while (std::getline(input_file, str)) {
    ++num_lines_in_file;
  }

  input_file.clear();
  input_file.seekg(0, std::ios::beg);

  for (int i = 0; i < num_lines_in_file; i += 4) {
    // source input
    std::getline(input_file, str);
    std::istringstream iss_input_source(str, std::istringstream::in);
    while (iss_input_source >> word) {
      if (std::stoi(word) != -1) {
        total_words += 1;
      }
    }

    // source output, do not use
    std::getline(input_file, str);
    // target input
    std::getline(input_file, str);
    std::istringstream iss_input_target(str, std::istringstream::in);
    while (iss_input_target >> word) {
      if (std::stoi(word) != -1) {
        total_words += 1;
        ++total_target_words;
      }
    }

    // target output, do not use
    std::getline(input_file, str);
  }
  input_file.clear();
  input_file.seekg(0, std::ios::beg);

}


void GetFileStatsSource(int &num_lines_in_file, std::ifstream &input) {
  std::string str;
  std::string word;
  num_lines_in_file = 0;
  while (std::getline(input, str)) {
    ++ num_lines_in_file;
  }

  input.clear();
  input.seekg(0, std::ios::beg);
}



void AddModelInformation(int num_layers, int lstm_size, int target_vocab_size, int source_vocab_size, bool attention_model, bool feed_input, bool multi_source, bool combine_lstm, bool char_cnn, std::string filename) {
  
  std::ifstream input(filename.c_str());
  std::string output_string = std::to_string(num_layers) + " " + std::to_string(lstm_size) + " "\
                            + std::to_string(target_vocab_size) + " " + std::to_string(source_vocab_size) + " "\
                            + std::to_string(attention_model) + " " + std::to_string(feed_input) + " "\
                            + std::to_string(multi_source) + " " + std::to_string(combine_lstm) + " "\
                            + std::to_string(char_cnn);

  std::string str;
  std::vector<std::string> file_lines;
  std::getline(input, str);
  file_lines.push_back(output_string);
  while (std::getline(input, str)) {
    file_lines.push_back(str);
  }
  input.close();
  std::ofstream output(filename.c_str());
  for (int i = 0; i < file_lines.size(); ++i) {
    output<<file_lines[i]<<"\n";
  }
  output.close();
}


} // end_namespace cc_util


namespace neural_machine_translation {

const std::string SystemTime::GetCurrentSystemTime() {
  auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  struct tm* ptm = localtime(&tt);
  char date[60] = {0};
  sprintf(date, "%02d:%02d:%02d, %02d/%02d/%d", 
         (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec, (int)ptm->tm_mon + 1, (int)ptm->tm_mday, (int)ptm->tm_year + 1900);
  return std::string(date);
}


OutputLogger::OutputLogger() {
  log_output_mode_ = false;
}


void OutputLogger::InitOutputLogger(std::string file_name, bool log_output_mode) {
  log_output_mode_ = log_output_mode;
  file_name_ = file_name;
  if (log_output_mode) {
    out_stream_.open(file_name.c_str());
  }
}



//////////////////
bool BasicMethod::Split(const std::string &input_string, const char &split_char, std::vector<std::string> &v_output) {
  std::string::size_type splitPos = input_string.find(split_char);
  std::string::size_type lastSplitPos = 0;
  std::string tempString;
  while (splitPos != std::string::npos) {
    tempString = input_string.substr(lastSplitPos, splitPos - lastSplitPos);
    if (!tempString.empty()) {
      v_output.push_back(tempString);
    }
    lastSplitPos = splitPos + 1;
    splitPos = input_string.find(split_char, lastSplitPos);
  }
  if (lastSplitPos < input_string.size()) {
    tempString = input_string.substr(lastSplitPos);
    v_output.push_back(tempString);
  }
  if (!v_output.empty()) {
    return true;
  } else {
    return false;
  }
}



bool BasicMethod::SplitWithString(const std::string &src, const std::string &separator, std::vector<std::string> &dest) {
  std::string str = src;
  std::string substring;
  std::string::size_type start = 0, index = 0;
  std::string::size_type separator_len = separator.size();
  while (index != std::string::npos && start < src.size()) {
    index = src.find(separator, start);
    if (index == 0) {
      start = start + separator_len;
      continue;
    }
    if (index == std::string::npos) {
      dest.push_back(src.substr(start));
      break;
    }
    dest.push_back(src.substr(start,index-start));
    start = index + separator_len;
  }
  return true;
}



bool BasicMethod::ClearIllegalChar(std::string &str) {
  std::string::size_type pos = 0;
  while((pos = str.find("\r", pos)) != std::string::npos) {
    str.replace(pos, 1, "");
  }

  pos = 0;
  while((pos = str.find("\n", pos)) != std::string::npos) {
    str.replace(pos, 1, "");
  }
  return true;
}


bool BasicMethod::RmEndSpace(std::string &str) {
  if (str != "") {
    std::string tmpStr;
    int pos = (int)str.length() - 1;
    while (pos >= 0 && str[ pos ] == ' ') {
      --pos;
    }
    tmpStr = str.substr(0, pos + 1);
    str = tmpStr;
  }
  return true;
}


bool BasicMethod::RmStartSpace(std::string &str) {
  std::string tmpStr;
  size_t pos = 0;
  for (std::string::iterator iter = str.begin(); iter != str.end(); ++iter) {
    if (*iter != ' ') {
      tmpStr = str.substr(pos, str.length() - pos);
      break;
    } else {
      ++pos;
    }
  }
  str = tmpStr;
  return true;
}


bool BasicMethod::ToLower(std::string &str) {
  for (std::string::size_type i = 0; i < str.size(); ++i) {
    if (isupper((unsigned char)str.at(i))) {
      str.at(i) = tolower((unsigned char)str.at(i));
    }
  }
  return true;
}


bool BasicMethod::RemoveExtraSpace(std::string &input_string, std::string &output_string) {
  char preceded_char = ' ';
  for (std::string::iterator iter = input_string.begin(); iter != input_string.end(); ++ iter ) {
    if (*iter == ' ' && preceded_char == ' ') {
      continue;
    } else {
      output_string.push_back(*iter);
      preceded_char = *iter;
    }
  }
  return true;
}


bool BasicMethod::RemoveAllSpace(const std::string &input_string, std::string &output_string) {
  output_string = input_string;
  std::string::size_type pos;
  while ((pos = output_string.find(" ")) != std::string::npos) {
    output_string.replace(pos, 1, "");
  }
  return true;
}


std::string BasicMethod::size_tToString(size_t &source) {
  std::stringstream oss;
  oss << source;
  return oss.str();
}




} // end namespace neural_machine_translation



