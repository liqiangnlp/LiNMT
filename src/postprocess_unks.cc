/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/14/2016
 * Time  : 14:25
 *
 */

#include "postprocess_unks.h"

namespace neural_machine_translation {

bool PostProcessUnks::Init(std::string &config) {
  
  std::map<std::string, std::string> m_parameters;
  ReadConfigFile(config, m_parameters);
  GeneralSettings(m_parameters);
    
  LoadDicrionary(dict_file_name_);
  LoadStopwords(stopword_file_name_);
  LoadEndPunctuation(end_punctuation_file_name_);
  return true;
}


bool PostProcessUnks::Process(std::string &source_sentence, std::string &input_sentence, std::string &output_sentence) {
  Processing(source_sentence, input_sentence, output_sentence, remove_oov_mode_);
  return true;
}


bool PostProcessUnks::ReadConfigFile(const std::string &config_file_name, std::map<std::string, std::string> &m_parameters) {

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


bool PostProcessUnks::GeneralSettings(std::map<std::string, std::string> &m_parameters) {
  std::string key = "--phrase-table";
  if (m_parameters.find(key) != m_parameters.end()) {
    dict_file_name_ = m_parameters[key];
  }

  key = "--stopword";
  if (m_parameters.find(key) != m_parameters.end()) {
    stopword_file_name_ = m_parameters[key];
  }

  key = "--end-punct";
  if (m_parameters.find(key) != m_parameters.end()) {
    end_punctuation_file_name_ = m_parameters[key];
  }

  key = "--rm-oov-mode";
  if (m_parameters.find(key) != m_parameters.end()) {
    remove_oov_mode_ = std::atoi(m_parameters[key].c_str());
    remove_oov_mode_ = !remove_oov_mode_;
  }
  return true;
}


void PostProcessUnks::LoadDicrionary(std::string &unk_dict_file_name) {
  std::ifstream unk_dict_stream(unk_dict_file_name.c_str());
  if (!unk_dict_stream) {
    logger<<"   Error: can not open "<<unk_dict_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  logger<<"\n$$ Loading dict\n"
        <<"   Dict                     : "<<unk_dict_file_name<<"\n";

  int line_num = 0;
  std::string dict_entry;
  BasicMethod basic_method;
  while (std::getline(unk_dict_stream, dict_entry)) {
    ++line_num;
    std::vector<std::string> v_entry;
    basic_method.ClearIllegalChar(dict_entry);
    basic_method.SplitWithString(dict_entry, " ||| ", v_entry);
    if (3 != v_entry.size()) {
      logger<<"\n   Error: format error in "<<line_num<<" lines\n";
    }
    uno_dict_[v_entry[0]] = v_entry[1];

    if (line_num % 10000 == 0) {
      logger<<"\r   "<<line_num<<" entries";
    }
  }
  logger<<"\r   "<<line_num<<" entries\n";
  unk_dict_stream.clear();
  unk_dict_stream.close();
  return;
}


void PostProcessUnks::LoadStopwords(std::string &unk_stopword_file_name) {
  std::ifstream unk_stopword_stream(unk_stopword_file_name.c_str());
  if (!unk_stopword_stream) {
    logger<<"   Error: can not open "<<unk_stopword_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  logger<<"\n$$ Loading stopwords\n"
        <<"   Stopwords                : "<<unk_stopword_file_name<<"\n";

  int line_num = 0;
  std::string stopword_entry;
  //BasicMethod basic_method;
  while (std::getline(unk_stopword_stream, stopword_entry)) {
    ++line_num;
    std::vector<std::string> v_entry;
    basic_method_.ClearIllegalChar(stopword_entry);
    basic_method_.Split(stopword_entry, '\t', v_entry);
    if (2 != v_entry.size()) {
      logger<<"\n   Error: format error in "<<line_num<<" lines\n";
    }
    uno_stopwords_[v_entry[1]] = true;

    if (line_num % 10000 == 0) {
      logger<<"\r   "<<line_num<<" entries, "<<uno_stopwords_.size()<<" stopwords";
    }
  }
  logger<<"\r   "<<line_num<<" entries, "<<uno_stopwords_.size()<<" stopwords\n";
  unk_stopword_stream.clear();
  unk_stopword_stream.close();
  return;
}


void PostProcessUnks::LoadEndPunctuation(std::string &unk_stopword_file_name) {
  std::ifstream unk_stopword_stream(unk_stopword_file_name.c_str());
  if (!unk_stopword_stream) {
    logger<<"   Error: can not open "<<unk_stopword_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  logger<<"\n$$ Loading punctuations\n"
        <<"   Punctuations             : "<<unk_stopword_file_name<<"\n";

  int line_num = 0;
  std::string stopword_entry;
  //BasicMethod basic_method;
  while (std::getline(unk_stopword_stream, stopword_entry)) {
    ++line_num;
    std::vector<std::string> v_entry;
    basic_method_.ClearIllegalChar(stopword_entry);
    basic_method_.Split(stopword_entry, '\t', v_entry);
    if (2 != v_entry.size()) {
      logger<<"\n   Error: format error in "<<line_num<<" lines\n";
    }

    if ("ENDPUNCT" == v_entry[0]) {
      uno_end_punctuations_[v_entry[1]] = true;
    }

    if (line_num % 10000 == 0) {
      logger<<"\r   "<<line_num<<" entries, "<<uno_end_punctuations_.size()<<" punctuations";
    }
  }
  logger<<"\r   "<<line_num<<" entries, "<<uno_end_punctuations_.size()<<" punctuations\n";
  unk_stopword_stream.clear();
  unk_stopword_stream.close();
  return;
}


void PostProcessUnks::Processing(std::string &line_source, std::string &line_nmt_translation, \
                                 std::string &line_output, bool &output_oov_mode) {
  basic_method_.ClearIllegalChar(line_nmt_translation);
  basic_method_.ClearIllegalChar(line_source);

  std::vector<std::string> v_input_sentence;
  basic_method_.SplitWithString(line_source, " |||| ", v_input_sentence);
  line_source = v_input_sentence.at(0);

  std::unordered_map<int, std::string> uno_generalization_number;          // $number
  std::unordered_map<int, std::string> uno_generalization_time;            // $time
  std::unordered_map<int, std::string> uno_generalization_date;            // $date
  std::unordered_map<int, std::string> uno_generalization_person;          // $person
  std::unordered_map<int, std::string> uno_generalization_location;        // $location
  std::unordered_map<int, std::string> uno_generalization_organization;    // $organization
  std::unordered_map<int, std::string> uno_generalization_literal;         // $literal
  std::unordered_map<int, std::string> uno_generalization_userdict_1;      // $userdict1
  std::unordered_map<int, std::string> uno_generalization_userdict_2;      // $userdict2
  std::unordered_map<int, std::string> uno_generalization_userdict_3;      // $userdict3
  std::unordered_map<int, std::string> uno_generalization_userdict_4;      // $userdict4
  std::unordered_map<int, std::string> uno_generalization_userdict_5;      // $userdict5
  std::unordered_map<int, std::string> uno_generalization_userdict_6;      // $userdict6
  std::unordered_map<int, std::string> uno_generalization_userdict_7;      // $userdict7
  std::unordered_map<int, std::string> uno_generalization_userdict_8;      // $userdict8
  std::unordered_map<int, std::string> uno_generalization_userdict_9;      // $userdict9
  std::unordered_map<int, std::string> uno_generalization_userdict_10;     // $userdict10
  std::unordered_map<int, std::string> uno_generalization_dict;            // $NMT_*

  // have generalization, $number, $time, $date, $person, $location, $organization, $literal, $dict
  if (2 == v_input_sentence.size()) {
    if (v_input_sentence.at(1).size() < 3) {
      logger<<"\n   Warning: format error in generalization.\n";
    } else if (v_input_sentence.at(1).front() != '{' || v_input_sentence.at(1).back() != '}') {
      logger<<"\n   Warning: format error in generalization.\n";
    } else {
      std::string generalization = v_input_sentence.at(1).substr(1, v_input_sentence.at(1).size() - 2);
      std::vector<std::string> v_generalizations;
      basic_method_.SplitWithString(generalization, "}{", v_generalizations);
      for(std::vector<std::string>::iterator iter = v_generalizations.begin(); iter != v_generalizations.end(); ++iter) {
        std::vector<std::string> v_fields_tmp;
        basic_method_.SplitWithString(*iter, " ||| ", v_fields_tmp);
        if (5 != v_fields_tmp.size()) {
          logger<<"\n   Warning: format error in generalization.\n";
        } else {
          if ("$number" == v_fields_tmp.at(3)) {
            uno_generalization_number[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$time" == v_fields_tmp.at(3)) {
            uno_generalization_time[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$date" == v_fields_tmp.at(3)) {
            uno_generalization_date[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$psn" == v_fields_tmp.at(3)) {
            uno_generalization_person[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$loc" == v_fields_tmp.at(3)) {
            uno_generalization_location[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$org" == v_fields_tmp.at(3)) {
            uno_generalization_organization[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$literal" == v_fields_tmp.at(3)) {
            uno_generalization_literal[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict1" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_1[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict2" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_2[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict3" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_3[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict4" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_4[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict5" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_5[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict6" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_6[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict7" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_7[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict8" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_8[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict9" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_9[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$userdict10" == v_fields_tmp.at(3)) {
            uno_generalization_userdict_10[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } else if ("$dict" == v_fields_tmp.at(3)) {
            uno_generalization_dict[atoi(v_fields_tmp.at(0).c_str())] = v_fields_tmp.at(2);
          } 
        }
      }
    }
  }



  std::vector<std::string> v_translation_fields;
  basic_method_.SplitWithString(line_nmt_translation, " |||| ", v_translation_fields);
  if (5 > v_translation_fields.size()) {
    line_output = line_nmt_translation;
    return;
  }

  std::vector<std::string> v_source_words;
  std::vector<std::string> v_target_words;
  std::vector<int> v_target_align;
  std::vector<std::string> v_target_align_scores;
  basic_method_.Split(line_source, ' ', v_source_words);
  basic_method_.Split(v_translation_fields[0], ' ', v_target_words);
  basic_method_.Split(v_translation_fields[4], ' ', v_target_align_scores);

  std::istringstream iss_align(v_translation_fields[2], std::istringstream::in);
  int tmp_digit;
  while (iss_align >> tmp_digit) {
    v_target_align.push_back(tmp_digit);
  }


  if (v_target_words.size() != v_target_align.size()) {
    logger<<"\n   Warning: format error\n";
    line_output = line_nmt_translation;
    return;
  }

  line_output = "";
  for (int i = 0; i < v_target_words.size(); ++i) {
    if ("<UNK>" == v_target_words[i]) {
      
      if (v_source_words.size() > v_target_align[i]) {
        if (v_source_words[v_target_align[i]].size() > 5 && '$' == v_source_words[v_target_align[i]].at(0) && \
            'N' == v_source_words[v_target_align[i]].at(1) && 'M' == v_source_words[v_target_align[i]].at(2) && \
            'T' == v_source_words[v_target_align[i]].at(3) && '_' == v_source_words[v_target_align[i]].at(4)) {
          if (0 != uno_generalization_dict.count(v_target_align[i])) {
            line_output += " " + uno_generalization_dict[v_target_align[i]];
          } else {
            logger<<"\n   Warning: format error in generalization\n";
          }
        } else if (0 != uno_stopwords_.count(v_source_words[v_target_align[i]])) {
          int position = -1;
          FindUnStopword(v_target_align_scores[i], v_source_words, position);

          if (-1 == position) {
            line_output += " .";
          } else if (v_source_words.size() > position) {
            if (v_source_words[position].size() > 5 && '$' == v_source_words[position].at(0) && \
                'N' == v_source_words[position].at(1) && 'M' == v_source_words[position].at(2) && \
                'T' == v_source_words[position].at(3) && '_' == v_source_words[position].at(4)) {
              if (0 != uno_generalization_dict.count(position)) {
                line_output += " " + uno_generalization_dict[position];
              } else {
                logger<<"\n   Warning: format error in generalization\n";
              }
            } else if (0 != uno_dict_.count(v_source_words[position])) {
              line_output += " " + uno_dict_[v_source_words[position]];
            } else {
              if (output_oov_mode) {
                line_output += " <" + v_source_words[position] + ">";
              }
            }
          } else {
            logger<<"\n   Warning: format error\n";
          }

        } else if (0 != uno_dict_.count(v_source_words[v_target_align[i]])) {
          if ("$number" != v_source_words[v_target_align[i]] && \
              "$time" != v_source_words[v_target_align[i]] && \
              "$date" != v_source_words[v_target_align[i]] && \
              "$psn" != v_source_words[v_target_align[i]] && \
              "$loc" != v_source_words[v_target_align[i]] && \
              "$org" != v_source_words[v_target_align[i]] && \
              "$literal" != v_source_words[v_target_align[i]] && \
              "$userdict1" != v_source_words[v_target_align[i]] && \
              "$userdict2" != v_source_words[v_target_align[i]] && \
              "$userdict3" != v_source_words[v_target_align[i]] && \
              "$userdict4" != v_source_words[v_target_align[i]] && \
              "$userdict5" != v_source_words[v_target_align[i]] && \
              "$userdict6" != v_source_words[v_target_align[i]] && \
              "$userdict7" != v_source_words[v_target_align[i]] && \
              "$userdict8" != v_source_words[v_target_align[i]] && \
              "$userdict9" != v_source_words[v_target_align[i]] && \
              "$userdict10" != v_source_words[v_target_align[i]]) {
            line_output += " " + uno_dict_[v_source_words[v_target_align[i]]];
          } else {            
            int position = -1;
            FindUnStopword(v_target_align_scores[i], v_source_words, position);

            if (-1 == position) {
              line_output += " .";
            } else if (v_source_words.size() > position) {
              if (v_source_words[position].size() > 5 && '$' == v_source_words[position].at(0) && \
                  'N' == v_source_words[position].at(1) && 'M' == v_source_words[position].at(2) && \
                  'T' == v_source_words[position].at(3) && '_' == v_source_words[position].at(4)) {
                if (0 != uno_generalization_dict.count(position)) {
                  line_output += " " + uno_generalization_dict[position];
                } else {
                  logger<<"\n   Warning: format error in generalization\n";
                }
              } else if (0 != uno_dict_.count(v_source_words[position])) {
                line_output += " " + uno_dict_[v_source_words[position]];
              } else {
                if (output_oov_mode) {
                  line_output += " <" + v_source_words[position] + ">";
                }
              }
            } else {
              logger<<"\n   Warning: format error\n";
            }
          }
        } else {
          if (output_oov_mode) {
            line_output += " <" + v_source_words[v_target_align[i]] + ">";
          }
        }
      } else {
        logger<<"\n   Warning: format error\n";
      }

    } else if ("$number" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_number, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$number>",line_output);
    } else if ("$time" == v_target_words[i]) {

      TargetIsGeneralization(uno_generalization_time, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$time>",line_output);
    } else if ("$date" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_date, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$date>",line_output);
    } else if ("$psn" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_person, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$psn>",line_output);
    } else if ("$loc" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_location, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$loc>",line_output);
    } else if ("$org" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_organization, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$org>",line_output);
    } else if ("$literal" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_literal, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$literal>",line_output);
    } else if ("$userdict1" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_1, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict1>",line_output);
    } else if ("$userdict2" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_2, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict2>",line_output);
    } else if ("$userdict3" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_3, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict3>",line_output);
    } else if ("$userdict4" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_4, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict4>",line_output);
    } else if ("$userdict5" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_5, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict5>",line_output);
    } else if ("$userdict6" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_6, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict6>",line_output);
    } else if ("$userdict7" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_7, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict7>",line_output);
    } else if ("$userdict8" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_8, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict8>",line_output);
    } else if ("$userdict9" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_9, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict9>",line_output);
    } else if ("$userdict10" == v_target_words[i]) {
      TargetIsGeneralization(uno_generalization_userdict_10, v_source_words, v_target_align, v_target_align_scores, i, \
                             output_oov_mode, "<$userdict10>",line_output);
    } else {
      line_output += " " + v_target_words[i];
    } 
  }

  basic_method_.RmStartSpace(line_output);


  // post-process two consecutive identical words
  std::vector<std::string> v_translation_words;
  basic_method_.Split(line_output, ' ', v_translation_words);
  line_output = "";
  std::string last_word = "";
  for (int i = 0; i < v_translation_words.size(); ++i) {

    if (v_translation_words[i] == last_word) {
      continue;
    } else {
      if (i != 0) {
        line_output += " ";
      } 
      line_output += v_translation_words[i];
      last_word = v_translation_words[i];
    }
  }

  // post-process two consecutive identical bigrams
  v_translation_words.clear();
  basic_method_.Split(line_output, ' ', v_translation_words);
  if (4 <= v_translation_words.size()) {
    line_output = "";
    std::string last_bigram = v_translation_words.at(0) + " " + v_translation_words.at(1);
    line_output += last_bigram;
    int i;
    for (i = 3; i < v_translation_words.size(); ++i) {
      std::string current_bigram = v_translation_words.at(i - 1) + " " + v_translation_words.at(i);
      if (current_bigram != last_bigram) {
        line_output += " " + v_translation_words.at(i - 1);
        last_bigram = v_translation_words.at(i - 2) + " " + v_translation_words.at(i - 1);
      } else {
        ++i;
      }
    }

    if (v_translation_words.size() + 1 != i) {
      line_output += " " + v_translation_words.at(v_translation_words.size() - 1);
    }
  }


  // post-process two consecutive end punctuations
  v_translation_words.clear();
  basic_method_.Split(line_output, ' ', v_translation_words);
  line_output = "";
  bool last_end_punctuation = false;
  for (int i = 0; i < v_translation_words.size(); ++i) {
    if (0 != uno_end_punctuations_.count(v_translation_words[i])) {
      if (last_end_punctuation) {
        continue;
      } else {
        last_end_punctuation = true;
        if (i != 0) {
          line_output += " ";
        }
        line_output += v_translation_words[i];
      }
    } else {
      if (i != 0) {
        line_output += " ";
      }
      line_output += v_translation_words[i];
      last_end_punctuation = false;
    }
  }

  for (int i = 1; i < v_translation_fields.size(); ++i) {
    line_output += " |||| " + v_translation_fields[i];
  }
  return;
}


void PostProcessUnks::TargetIsGeneralization(std::unordered_map<int, std::string> &uno_generalization, std::vector<std::string> &v_source_words, \
                                             std::vector<int> &v_target_align, std::vector<std::string> &v_target_align_scores, \
                                             int &i, bool &output_oov_mode, std::string label, std::string &line_output) {
  if (0 != uno_generalization.count(v_target_align[i])) {
    line_output += " " + uno_generalization[v_target_align[i]];
    uno_generalization.erase(v_target_align[i]);
  } else {
    int position = -1;
    FindGeneralization(uno_generalization, v_target_align_scores[i], v_source_words, position);
    if (position != -1) {
      line_output += " " + uno_generalization[position];
      uno_generalization.erase(position);
    } else {
      if (output_oov_mode) {
        line_output += " " + label;
      }
    }
  }
}


void PostProcessUnks::FindGeneralization(std::unordered_map<int, std::string> &uno_generalization, std::string &align_scores, \
                                         std::vector<std::string> &v_source_words, int &position) {

  std::vector<std::string> v_scores;
  basic_method_.Split(align_scores, '/', v_scores);

  if (v_scores.size() != v_source_words.size()) {
    logger<<"\n   Warning: format error\n";
    return;
  }

  std::multimap<float, int> m_score_position;
  int i = 0;
  for (std::vector<std::string>::iterator iter = v_scores.begin(); iter != v_scores.end(); ++iter) {
    m_score_position.insert(std::make_pair(atof(iter->c_str()), i));
    ++i;
  }

  for (std::multimap<float, int>::reverse_iterator riter = m_score_position.rbegin(); riter != m_score_position.rend(); ++riter) {
    //if ("$number" == v_source_words.at(riter->second)) {
    if (0 != uno_generalization.count(riter->second)) {
       position = riter->second;
       break;
    } else {
      continue;
    }
  }
  return;
}



void PostProcessUnks::FindUnStopword(std::string &align_scores, std::vector<std::string> &v_source_words, int &position) {
  std::vector<std::string> v_scores;
  basic_method_.Split(align_scores, '/', v_scores);

  if (v_scores.size() != v_source_words.size()) {
    logger<<"\n   Warning: format error\n";
    return;
  }

  std::multimap<float, int> m_score_position;
  int i = 0;
  for (std::vector<std::string>::iterator iter = v_scores.begin(); iter != v_scores.end(); ++iter) {
    m_score_position.insert(std::make_pair(atof(iter->c_str()), i));
    ++i;
  }

  for (std::multimap<float, int>::reverse_iterator riter = m_score_position.rbegin(); riter != m_score_position.rend(); ++riter) {
    if (0 == uno_stopwords_.count(v_source_words.at(riter->second)) && \
        "$number" != v_source_words.at(riter->second) && \
        "$time" != v_source_words.at(riter->second) && \
        "$date" != v_source_words.at(riter->second) && \
        "$psn" != v_source_words.at(riter->second) && \
        "$loc" != v_source_words.at(riter->second) && \
        "$org" != v_source_words.at(riter->second) && \
        "$literal" != v_source_words.at(riter->second) && \
        "$userdict1" != v_source_words.at(riter->second) && \
        "$userdict2" != v_source_words.at(riter->second) && \
        "$userdict3" != v_source_words.at(riter->second) && \
        "$userdict4" != v_source_words.at(riter->second) && \
        "$userdict5" != v_source_words.at(riter->second) && \
        "$userdict6" != v_source_words.at(riter->second) && \
        "$userdict7" != v_source_words.at(riter->second) && \
        "$userdict8" != v_source_words.at(riter->second) && \
        "$userdict9" != v_source_words.at(riter->second) && \
        "$userdict10" != v_source_words.at(riter->second)) {
       position = riter->second;
       break;
    } else {
      continue;
    }
  }

  return;
}

} // end of neural_machine_translation namespace


char *postprocess_result__ = NULL;
neural_machine_translation::PostProcessUnks post_process_unks;

void python_unk_init(char *msg) {
  std::string configuration(msg);
  post_process_unks.Init(configuration);
}

char* python_unk_do_job(char *src_sentence, char *translation) {
  if (postprocess_result__ != NULL) {
    delete[] postprocess_result__;
  }

  std::string source_sentence(src_sentence);
  std::string translation_result(translation);
  std::string output_sentence;
  post_process_unks.Process(source_sentence, translation_result, output_sentence);
  std::vector<std::string> v_fields;
  neural_machine_translation::BasicMethod basic_method;
  basic_method.SplitWithString(output_sentence, " |||| ", v_fields);
  if (v_fields.size() > 1) {
    output_sentence = v_fields.at(0);
  }

#ifdef WIN32
  postprocess_result__ = new char[ output_sentence.size() + 1 ];
  strcpy_s(postprocess_result__, output_sentence.size() + 1, output_sentence.c_str());
#else
  postprocess_result__ = new char[ output_sentence.size() + 1 ];
  strncpy(postprocess_result__, output_sentence.c_str(), output_sentence.size() + 1);
#endif

  return postprocess_result__;
}




