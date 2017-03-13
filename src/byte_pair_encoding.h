/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 08/08/2017
 * Time  : 15:18
 *
 */

#ifndef BYTE_PAIR_ENCODING_H_
#define BYTE_PAIR_ENCODING_H_

#include <string>
#include <fstream>
//#include <hash_map>
#include <unordered_map>
#include <map>

#include "utility_cu.h"
#include "utility_cc.h"

#include "debug.h"


#ifdef _WIN32
#include <Windows.h>
#else
#include <locale>
#include <cstdlib>
#endif



namespace neural_machine_translation {


class EncodingConversion {
public:
  std::wstring UTF8ToUnicode (const char* line);
  std::string UnicodeToUTF8(std::wstring& line);
};



class BigramUpdateParameters {
public:
  int index_ = 0;
  int frequency_ = 0;
  std::string old_word_;
  std::string word_;

public:
  BigramUpdateParameters() {}

  BigramUpdateParameters(const int &index, int &frequency, std::string &old_word, std::string &word) {
    index_ = index;
    frequency_ = frequency;
    old_word_ = old_word;
    word_ = word;
  }

  ~BigramUpdateParameters() {}
};



class BytePairEncoding {

public:
  //std::hash_map<std::string, int> hm_vocabulary_;            // raw (words, frequencies)
  std::unordered_map<std::string, int> hm_vocabulary_;
  std::multimap<int, std::string> mm_vocabulary_;              // sorted (words, frequencies)
  std::vector<std::pair<std::string, int> > v_p_vocabulary_;   // vector sorted (words, frequencies)

  //std::hash_map<std::string, int> hm_bigram_stats_;
  std::unordered_map<std::string, int> hm_bigram_stats_;
  //std::hash_map<std::string, int> hm_bigram_big_stats_;
  std::unordered_map<std::string, int> hm_bigram_big_stats_;
  //std::hash_map<std::string, std::hash_map<int, int> > hm_bigram_indices_;
  std::unordered_map<std::string, std::unordered_map<int, int>> hm_bigram_indices_;

public:
  int vocabulary_size_ = 0;
  int min_frequency_ = 0;

public:
  BytePairEncoding() {}
  ~BytePairEncoding() {}

public:
  BasicMethod basic_method_;
  EncodingConversion encoding_conversion_;

public:
  void Train(const int &vocabulary_size, const int &min_frequency, const std::string &input_file_name, const std::string &output_file_name);

private:
  void TrainBpe(std::ofstream &out_file, std::ofstream &out_log);

private:
  void GetVocabulary(std::ifstream &in_file);
  void GetBigramVocabulary();
  void ReplaceBigram(std::string &max_frequency_bigram, std::vector<BigramUpdateParameters> &v_out_bigram_update);
  void UpdateBigramVocabulary(const std::string &max_frequency_bigram, std::vector<BigramUpdateParameters> &v_bigram_update);

private:
  void PruneBigramStats(const int &threshold);

private:
  //void GetMaxValueHashMap(const std::hash_map<std::string, int> &in_hash_map, std::string &out_key, int &out_max_value);
  void GetMaxValueHashMap(const std::unordered_map<std::string, int> &in_hash_map, std::string &out_key, int &out_max_value);
};

} // End of namespace neural_machine_translation




#endif


