/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#ifndef DECODER_SENTENCE_H_
#define DECODER_SENTENCE_H_

#include <string>
#include <map>

#include "utility_cu.h"
#include "global_configuration.h"
#include "decoder.h"


namespace neural_machine_translation {

class DecoderSentence {
private:
  GlobalConfiguration global_configuration_;
  AttentionConfiguration attention_configuration_;

private:
  BasicMethod basic_method_;

public:
  EnsembleFactory<precision> ensemble_decode_;

public:
  std::unordered_map<std::string, int> uno_source_mapping_word_int_;
  std::unordered_map<int, std::string> uno_target_mapping_int_word_;
  

public:
  bool Init(std::string &config);
  bool Process(const std::string &input_sentence, std::string &output_sentence);

private:
  bool IntegerizeSentence(const std::string &input_sentence, std::vector<int> &v_input_sentence_int);
  bool UnintSentence(std::string &sentence);

private:
  bool GeneralSettings(std::map<std::string, std::string> &m_parameters);
  bool ReadConfigFile(const std::string &config_file_name, std::map<std::string, std::string> &m_parameters);
  bool LoadParallelVocabulary(const std::string &nmt_model_file, int &target_vocab_size);
};

} // End of namespace neural_machine_translation


extern "C" {
  void python_decoder_init(char *msg);
  char* python_decoder_do_job(char *sentence);
}



#endif


