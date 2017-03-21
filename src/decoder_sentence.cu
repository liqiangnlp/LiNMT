/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#include "decoder_sentence.h"

namespace neural_machine_translation {

bool DecoderSentence::Init(std::string &config) {
  logger<<"\n$$ Init\n";

  std::map<std::string, std::string> m_parameters;
  ReadConfigFile(config, m_parameters);
  GeneralSettings(m_parameters);
  
  LoadParallelVocabulary(global_configuration_.model_names_[0], global_configuration_.target_vocab_size_);
  
  global_configuration_.PrintDecodingSentParameters();



  //EnsembleFactory<precision> ensemble_decode;
  ensemble_decode_.InitDecoderSentence(global_configuration_.model_names_, global_configuration_.hypotheses_number_, global_configuration_.beam_size_, \
                                      global_configuration_.min_decoding_ratio_, global_configuration_.penalty_, global_configuration_.longest_sentence_, \
                                      global_configuration_.print_decoding_information_mode_, global_configuration_.print_alignments_scores_mode_, \
                                      global_configuration_.decoder_output_file_, global_configuration_.gpu_indices_, \
                                      global_configuration_.max_decoding_ratio_, global_configuration_.target_vocab_size_, \
                                      global_configuration_.decoding_lp_alpha_, global_configuration_.decoding_cp_beta_, \
                                      global_configuration_.decoding_diversity_, global_configuration_.decoding_dump_sentence_embedding_mode_, \
                                      global_configuration_.decoding_sentence_embedding_file_name_, global_configuration_);

  return true;
}

bool DecoderSentence::Process(const std::string &input_sentence, std::string &output_sentence) {

  std::vector<int> v_input_sentence_int;
  IntegerizeSentence(input_sentence, v_input_sentence_int);

  ensemble_decode_.DecodeSentence(v_input_sentence_int, output_sentence);
  UnintSentence(output_sentence);

  return true;
}


bool DecoderSentence::IntegerizeSentence(const std::string &input_sentence, std::vector<int> &v_input_sentence_int) {

  std::vector<std::string> v_input_tmp;
  basic_method_.SplitWithString(input_sentence, " |||| ", v_input_tmp);
  std::string input_sentence_tmp = v_input_tmp[0];

  std::istringstream iss_src(input_sentence_tmp, std::istringstream::in);
  std::string word;
  while (iss_src >> word) {
    if (uno_source_mapping_word_int_.count(word) == 0) {
      v_input_sentence_int.push_back(uno_source_mapping_word_int_["UNK"]);
    } else {
      v_input_sentence_int.push_back(uno_source_mapping_word_int_[word]);
    }
  }

  std::reverse(v_input_sentence_int.begin(), v_input_sentence_int.end());
  return true;
}

bool DecoderSentence::UnintSentence(std::string &sentence) {

  std::istringstream iss(sentence, std::istringstream::in);
  std::vector<int> sentence_int;

  bool other_information_flag = false;
  std::string other_information = "";
  std::string word;
  while (iss >> word) {
    if (other_information_flag) {
      other_information += " " + word;
    } else if ("||||" != word) {
      sentence_int.push_back(std::stoi(word));
    } else {
      other_information_flag = true;
      other_information += " " + word;
    }
  }

  if (other_information.size() >= 4 &&
      ('|' == other_information.at(other_information.size() - 1)) && ('|' == other_information.at(other_information.size() - 2)) &&
      ('|' == other_information.at(other_information.size() - 3)) && ('|' == other_information.at(other_information.size() - 4))) {
    other_information += " ";
  }

  sentence = "";
  for (int i = 0; i < sentence_int.size(); ++i) {
    sentence += uno_target_mapping_int_word_[sentence_int[i]];
    if (i != sentence_int.size() - 1) {
      sentence += " ";
    }
  }
  sentence += other_information;

  return true;
}


bool DecoderSentence::GeneralSettings(std::map<std::string, std::string> &m_parameters) {

  std::string key = "--tmp-dir-location";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.tmp_location_ = m_parameters[key];
    if ("" != global_configuration_.tmp_location_) {
      if ('/' != global_configuration_.tmp_location_[global_configuration_.tmp_location_.size() - 1]) {
        global_configuration_.tmp_location_ += '/';
      }
    }
  }

  key = "--longest-sent";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.longest_sentence_ = std::atoi(m_parameters[key].c_str());
  }
  global_configuration_.longest_sentence_ += 4;

  key = "--beam-size";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.beam_size_ = std::atoi(m_parameters[key].c_str());
  }

  key = "--attention-width";
  if (m_parameters.find(key) != m_parameters.end()) {
    attention_configuration_.d_ = std::stoi(m_parameters[key].c_str());
  }



  key = "--print-decoding-info";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.print_decoding_information_mode_ = std::atoi(m_parameters[key].c_str());
    if (global_configuration_.print_decoding_information_mode_) {
      unk_replacement_mode__ = true;
      for (int i = 0; i < global_configuration_.beam_size_; ++i) {
        viterbi_alignments__.push_back(-1);
      }
      for (int i = 0; i < global_configuration_.beam_size_ * global_configuration_.longest_sentence_; ++i) {
        alignment_scores__.push_back(0);
      }
      p_host_align_indices__ = (int*)malloc((2 * attention_configuration_.d_ + 1) * global_configuration_.beam_size_ * sizeof(int));
      p_host_alignment_values__ = (precision*)malloc((2 * attention_configuration_.d_ + 1) * global_configuration_.beam_size_ * sizeof(precision));
    }
  }

  key = "--print-align-scores";
  if(m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.print_alignments_scores_mode_ = std::atoi(m_parameters[key].c_str());
  }

  boost::filesystem::path unique_path = boost::filesystem::unique_path();
  if ("" != global_configuration_.tmp_location_) {
    unique_path = boost::filesystem::path(global_configuration_.tmp_location_ + unique_path.string());
  }
  boost::filesystem::create_directories(unique_path);
  global_configuration_.unique_dir_ = unique_path.string();
  logger<<"\n$$ Directory Information\n"
        <<"   Tmp directory            : "<<global_configuration_.unique_dir_<<"\n";

  key = "--nmt-model";
  if (m_parameters.find(key) != m_parameters.end()) {
    basic_method_.Split(m_parameters[key], ' ', global_configuration_.model_names_);
  } else {
    logger<<"   Error: you must assign --nmt-model in config file!\n";
    exit(EXIT_FAILURE);
  }

  key = "--nbest";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.hypotheses_number_ = std::stoi(m_parameters[key].c_str());
  }

  key = "--multi-gpu";
  if (m_parameters.find(key) != m_parameters.end()) {
    logger<<"   Warning: multi-gpu is not written right now!\n";
    exit(EXIT_FAILURE);
  } else {
    for (int i = 0; i < global_configuration_.model_names_.size(); ++i) {
      global_configuration_.gpu_indices_.push_back(0);
    }
  }

  if (global_configuration_.beam_size_ <= 0) {
    logger<<"Error: --beam-size cannot be <= 0!\n";
    boost::filesystem::path tmp_path(unique_path);
    boost::filesystem::remove_all(tmp_path);
    exit(EXIT_FAILURE);
  }

  if (global_configuration_.penalty_ < 0) {
    logger<<"Error: --penalty cannot be < 0!\n";
    boost::filesystem::path tmp_path(unique_path);
    boost::filesystem::remove_all(tmp_path);
    exit(EXIT_FAILURE);
  }

  key = "--decoding-ratio";
  if (m_parameters.find("--decoding-ratio") != m_parameters.end()) {
    std::vector<std::string> v_decoding_ratio;
    basic_method_.Split(m_parameters["--decoding-ratio"], ' ', v_decoding_ratio);
    if (v_decoding_ratio.size() != 2) {
      logger<<"Error: only two inputs for --decoding-ratio, now is "<<v_decoding_ratio.size()<<"\n";
      boost::filesystem::path tmp_path(unique_path);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
    global_configuration_.min_decoding_ratio_ = std::atoi(v_decoding_ratio.at(0).c_str());
    global_configuration_.max_decoding_ratio_ = std::stoi(v_decoding_ratio.at(1).c_str());
    if (global_configuration_.min_decoding_ratio_ >= global_configuration_.max_decoding_ratio_) {
      logger<<"Error: min decoding ratio must be less than max decoding ratio!\n";
      boost::filesystem::path tmp_path(unique_path);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
  }

  key = "--lp-alpha";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.decoding_lp_alpha_ = std::atof(m_parameters[key].c_str());
  }

  key = "--cp-beta";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.decoding_cp_beta_ = std::atof(m_parameters[key].c_str());
  }

  key = "--diversity";
  if (m_parameters.find(key) != m_parameters.end()) {
    global_configuration_.decoding_diversity_ = std::atof(m_parameters[key].c_str());
  }


  global_configuration_.training_mode_ = false;
  global_configuration_.decode_mode_ = false;
  global_configuration_.decode_sentence_mode_ = true;
  global_configuration_.test_mode_ = false;
  global_configuration_.stochastic_generation_mode_ = false;
  global_configuration_.postprocess_unk_mode_ = false;
  global_configuration_.calculate_bleu_mode_ = false;
  global_configuration_.average_models_mode_ = false;
  global_configuration_.vocabulary_replacement_mode_ = false;
  global_configuration_.dump_word_embedding_mode_ = false;
  global_configuration_.train_bpe_mode_ = false;
  global_configuration_.segment_bpe_mode_ = false;
  global_configuration_.sequence_to_sequence_mode_ = true;
  return true;
}


bool DecoderSentence::ReadConfigFile(const std::string &config_file_name, std::map<std::string, std::string> &m_parameters) {

  std::ifstream in_config_file(config_file_name.c_str());
  if (!in_config_file) {
    logger<<"   ERROR: Config File "<<config_file_name<<" does not exist, exit!\n";
    exit (EXIT_FAILURE);
  }

  std::string line_of_config_file;
  while (std::getline(in_config_file, line_of_config_file)) {
    basic_method_.ClearIllegalChar(line_of_config_file);
    basic_method_.RmStartSpace(line_of_config_file);
    basic_method_.RmEndSpace(line_of_config_file);

    if (line_of_config_file == "" || *line_of_config_file.begin() == '#') {
      continue;
    } else if (line_of_config_file.find("param=\"") == line_of_config_file.npos || \
               line_of_config_file.find("value=\"") == line_of_config_file.npos) {
      continue;
    } else {
      std::string::size_type pos = line_of_config_file.find( "param=\"" );
      pos += 7;
      std::string key;
      for ( ; line_of_config_file[pos] != '\"' && pos < line_of_config_file.length(); ++pos) {
        key += line_of_config_file[pos];
      }
      if (line_of_config_file[ pos ] != '\"') {
        continue;
      }

      pos = line_of_config_file.find( "value=\"" );
      pos += 7;
      std::string value;

      for ( ; line_of_config_file[pos] != '\"' && pos < line_of_config_file.length(); ++pos ) {
        value += line_of_config_file[pos];
      }

      if (line_of_config_file[pos] != '\"') {
        continue;
      }

      if (m_parameters.find(key) == m_parameters.end()) {
        m_parameters.insert(make_pair(key, value));
      } else {
        m_parameters[key] = value;
      }
    }
  }
  in_config_file.close();
  return true;
}


bool DecoderSentence::LoadParallelVocabulary(const std::string &nmt_model_file, int &target_vocab_size) {
  std::ifstream in_nmt_model(nmt_model_file.c_str());
  if (!in_nmt_model) {
    logger<<"   Error: can not open "<<nmt_model_file<<" file!\n";
    exit(EXIT_FAILURE);
  }

    // get parameters of neural network model
  // 0: number of layers
  // 1: lstm size
  // 2: target vocab size
  // 3: source vocab size
  // 4: attention mode
  // 5: feed input mode
  // 6: multi source mode
  // 7: combine lstm mode
  // 8: char cnn mode
  std::string str;
  std::string word;
  std::vector<std::string> file_input_vector;
  std::getline(in_nmt_model, str);
  std::istringstream iss(str, std::istringstream::in);
  while (iss >> word) {
    file_input_vector.push_back(word);
  }

  // set target vocab size
  target_vocab_size = std::stoi(file_input_vector[2]);

  // now get the source mappings
  std::getline(in_nmt_model, str);     // get this line, since all equals
  while (std::getline(in_nmt_model, str)) {

    int tmp_index;
    if (str.size() > 3 && str[0] == '=' && str[1] == '=' && str[2] == '=') {
      break;  // done with target mapping
    }

    std::istringstream iss (str, std::istringstream::in);
    // first is index
    iss >> word;
    tmp_index = std::stoi(word);
    // second is word
    iss >> word;
    uno_source_mapping_word_int_[word] = tmp_index;
  }

  // now get the target mappings
  std::getline(in_nmt_model, str);     // get this line, since all equals
  while (std::getline(in_nmt_model, str)) {

    int tmp_index;
    if (str.size() > 3 && str[0] == '=' && str[1] == '=' && str[2] == '=') {
      break;  // done with target mapping
    }

    std::istringstream iss (str, std::istringstream::in);
    // first is index
    iss >> word;
    tmp_index = std::stoi(word);
    // second is word
    iss >> word;
    uno_target_mapping_int_word_[tmp_index] = word;
  }

  in_nmt_model.close();
  return true;
}

} // End of namespace neural_machine_translation


char *decoder_result__ = NULL;
neural_machine_translation::DecoderSentence decoder_sentence__;

void python_decoder_init(char *msg) {
  std::string configuration(msg);
  decoder_sentence__.Init(configuration);
}

char* python_decoder_do_job(char *sentence) {
  if (decoder_result__ != NULL) {
    delete[] decoder_result__;
  }

  std::string input_sentence(sentence);
  std::string output_sentence;
  decoder_sentence__.Process(input_sentence, output_sentence);

#ifdef WIN32
  decoder_result__ = new char[ output_sentence.size() + 1 ];
  strcpy_s(decoder_result__, output_sentence.size() + 1, output_sentence.c_str());
#else
  decoder_result__ = new char[ output_sentence.size() + 1 ];
  strncpy(decoder_result__, output_sentence.c_str(), output_sentence.size() + 1);
#endif

  return decoder_result__;
}





