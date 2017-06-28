/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 03/14/2017
 * Time  : 15:18
 *
 */

#include "byte_pair_encoding.h"

namespace neural_machine_translation {

std::wstring EncodingConversion::UTF8ToUnicode (const char* line) {
#ifdef _WIN32
  size_t size = MultiByteToWideChar(CP_UTF8, 0, line, -1, NULL, 0);
  wchar_t* wcstr = new wchar_t[size];
  if (!wcstr)
    return L"";
  MultiByteToWideChar(CP_UTF8, 0, line, -1, wcstr, size);
#else
  setlocale(LC_ALL, "zh_CN.UTF-8");
  size_t size = mbstowcs(NULL, line, 0);
  wchar_t* wcstr = new wchar_t[size + 1];
  if (!wcstr)
    return L"";
  mbstowcs(wcstr, line, size + 1);
#endif
  std::wstring final(wcstr);
  delete[] wcstr;

  return final;
}


std::string EncodingConversion::UnicodeToUTF8(std::wstring& line) {
#ifdef _WIN32
  size_t size = WideCharToMultiByte(CP_UTF8, 0, line.c_str(), -1, NULL, 0, NULL, NULL);
  char* mbstr = new char[size];
  if (!mbstr)
    return "";
  WideCharToMultiByte(CP_UTF8, 0, line.c_str(), -1, mbstr, size, NULL, NULL);
#else
  setlocale(LC_ALL, "zh_CN.UTF-8");
  size_t size = wcstombs(NULL, line.c_str(), 0);
  char* mbstr = new char[size + 1];
  if (!mbstr)
    return "";
  wcstombs(mbstr, line.c_str(), size+1);
#endif
  std::string final(mbstr);
  delete[] mbstr;
  return final;
}



void BytePairEncoding::Train(const int &vocabulary_size, const int &min_frequency, const std::string &input_file_name, const std::string &output_file_name) {
  vocabulary_size_ = vocabulary_size;
  min_frequency_ = min_frequency;
  
  std::ifstream in_file(input_file_name.c_str());
  if (!in_file) {
    logger<<"   Error: can not open "<<input_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  std::ofstream out_file(output_file_name.c_str());
  if (!out_file) {
    logger<<"   Error: can not open "<<output_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  std::string output_log_file_name = output_file_name + ".log";
  std::ofstream out_log(output_log_file_name.c_str());
  if (!out_log) {
    logger<<"   Error: can not open "<<output_log_file_name<<"\n";
    exit(EXIT_FAILURE);
  }


  GetVocabulary(in_file);
  GetBigramVocabulary();

  TrainBpe(out_file, out_log);

  in_file.close();
  out_file.close();
  out_log.close();
  return;
}


void BytePairEncoding::Segment(const std::string &input_codes_file_name, const std::string &input_file_name, const std::string &output_file_name) {
  std::ifstream in_codes_file(input_codes_file_name.c_str());
  if (!in_codes_file) {
    logger<<"   Error: can not open "<<input_codes_file_name<<"\n";
    exit(EXIT_FAILURE);
  }
  LoadCodes(in_codes_file);
  in_codes_file.close();


  std::ifstream in_file(input_file_name.c_str());
  if (!in_file) {
    logger<<"   Error: can not open "<<input_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  std::ofstream out_file(output_file_name.c_str());
  if (!out_file) {
    logger<<"   Error: can not open "<<output_file_name<<"\n";
    exit(EXIT_FAILURE);
  }

  /*
  std::string output_log_file_name = output_file_name + ".log";
  std::ofstream out_log(output_log_file_name.c_str());
  if (!out_log) {
    logger<<"   Error: can not open "<<output_log_file_name<<"\n";
    exit(EXIT_FAILURE);
  }
  */

  std::chrono::time_point<std::chrono::system_clock> start_total, end_total;
  std::chrono::duration<double> elapsed_seconds;
  start_total = std::chrono::system_clock::now();    // start time

  logger<<"\n$$ Segment\n";
  std::string input_sentence;
  int i = 0;
  while (std::getline(in_file, input_sentence)) {
    ++i;
    basic_method_.ClearIllegalChar(input_sentence);
    basic_method_.RmStartSpace(input_sentence);
    basic_method_.RmEndSpace(input_sentence);

    std::vector<std::string> v_string_generalization;
    basic_method_.SplitWithString(input_sentence, " |||| " ,v_string_generalization);

    std::string output_sentence;
    if (0 == v_string_generalization.size()) {
      output_sentence = "";
    } else {
      SegmentBpe(v_string_generalization.at(0), output_sentence);
      if (2 == v_string_generalization.size()) {
        std::string output_generalization;
        if (output_sentence == v_string_generalization.at(0)) {
          output_generalization = v_string_generalization.at(1);
        } else {
          ModifyGeneralization(output_sentence, v_string_generalization.at(1), output_generalization);
        }
        output_sentence += " |||| " + output_generalization;
      }
    }
    out_file<<output_sentence<<"\n"<<std::flush;
    end_total = std::chrono::system_clock::now();    // start time
    elapsed_seconds = end_total - start_total;

    if (i % 1000 == 0) {
      logger<<"\r   "<<i<<" sentences, "<<(float)i/elapsed_seconds.count()<<" sentences/s";
    }
  }
  logger<<"\r   "<<i<<" sentences, "<<(float)i/elapsed_seconds.count()<<" sentences/s\n";

  in_file.close();
  out_file.close();
  /*
  out_log.close();
  */
  return;
}


void BytePairEncoding::ModifyGeneralization(const std::string &segmented_string, const std::string &raw_generalization, std::string &modified_generalization) {
  //modified_generalization = "modified " + raw_generalization;
  std::vector<std::string> v_segmented_words;
  basic_method_.Split(segmented_string, ' ', v_segmented_words);
  std::string input_generalization(raw_generalization.substr(1, raw_generalization.size() - 2));
  //modified_generalization = input_generalization;
  std::vector<std::string> v_generalizations;
  basic_method_.SplitWithString(input_generalization, "}{", v_generalizations);
  int last_modified_position = -1;
  for (std::vector<std::string>::iterator iter = v_generalizations.begin(); iter != v_generalizations.end(); ++iter) {
    std::vector<std::string> v_generalizations_details;
    basic_method_.SplitWithString(*iter, " ||| ", v_generalizations_details);
    int i = std::atoi(v_generalizations_details.at(0).c_str());
    int j = std::atoi(v_generalizations_details.at(1).c_str());
    int steps = 0;
    for (int k = 0; k < v_segmented_words.size(); ++k) {
      if (v_segmented_words.at(k).size() > 2) {

        if (k >= i && k > last_modified_position && v_segmented_words.at(k) == v_generalizations_details.at(3)) {
          last_modified_position = k;
          break;
        }
        std::string tmp_string(v_segmented_words.at(k).substr(v_segmented_words.at(k).size() - 2, v_segmented_words.at(k).size() - 1));
        if ("@@" == tmp_string) {
          ++steps;
        }
      }
    }
    i += steps;
    j += steps;
    modified_generalization += "{" + basic_method_.IntToString(i) + " ||| "+ \
                               basic_method_.IntToString(j) + " ||| " + \
                               v_generalizations_details.at(2) + " ||| " + \
                               v_generalizations_details.at(3) + " ||| " + \
                               v_generalizations_details.at(4) + "}";

    
  }
  return;
}




void BytePairEncoding::TrainBpe(std::ofstream &out_file, std::ofstream &out_log) {
  std::string return_string;
  int max_value = 0;
  GetMaxValueHashMap(hm_bigram_stats_, return_string, max_value);

  hm_bigram_big_stats_ = hm_bigram_stats_;
  int threshold = max_value / 10;

  std::chrono::time_point<std::chrono::system_clock> start_total, end_total;
  std::chrono::duration<double> elapsed_seconds;
  start_total = std::chrono::system_clock::now();    // start time

  logger<<"\n$$ Action:\n";
  int i;
  for (i = 0; i < vocabulary_size_; ++i) {
    std::string max_frequency_bigram;
    if (hm_bigram_stats_.size() > 0) {
      GetMaxValueHashMap(hm_bigram_stats_, max_frequency_bigram, max_value);
    }

    if (hm_bigram_stats_.size() == 0 || (i > 0 && hm_bigram_stats_[max_frequency_bigram] < threshold)) {
      PruneBigramStats(threshold);
      hm_bigram_stats_.clear();
      hm_bigram_stats_ = hm_bigram_big_stats_;
      GetMaxValueHashMap(hm_bigram_stats_, max_frequency_bigram, max_value);
      threshold = hm_bigram_stats_[max_frequency_bigram] * i / (i + 10000);
      PruneBigramStats(threshold);

    }

    if (hm_bigram_stats_[max_frequency_bigram] < min_frequency_) {
      out_log<<"no pair has frequency >= "<<min_frequency_<<". Stopping\n"<<std::flush;
      break;
    }

    out_file<<max_frequency_bigram<<"\n"<<std::flush;
    std::string max_frequency_bigram_wo_space;
    basic_method_.RemoveAllSpace(max_frequency_bigram, max_frequency_bigram_wo_space);
    out_log<<"pair "<<i<<": "<<max_frequency_bigram<<" -> "<<max_frequency_bigram_wo_space<<" (frequency "<<max_value<<")\n"<<std::flush;
    

    std::vector<BigramUpdateParameters> v_bigram_update;
    ReplaceBigram(max_frequency_bigram, v_bigram_update);
    UpdateBigramVocabulary(max_frequency_bigram, v_bigram_update);
    hm_bigram_stats_[max_frequency_bigram] = 0;

    if (i % 100 == 0) {
      PruneBigramStats(threshold);
    }

    end_total = std::chrono::system_clock::now();
    // compute the final runtime
    elapsed_seconds = end_total - start_total;

    if (i % 1000 == 0) {
      logger<<"\r   "<<i<<" actions, "<<threshold<<" threshold, "<<(float)i/elapsed_seconds.count()<<" actions/s";
    }
  }
  logger<<"\r   "<<i<<" actions, "<<threshold<<" threshold, "<<(float)i/elapsed_seconds.count()<<" actions/s\n";
}



void BytePairEncoding::GetVocabulary(std::ifstream &in_file) {
  std::string line = "";
  int line_num = 0;

  logger<<"\n$$ Get vocabulary\n";
  while (std::getline(in_file, line)) {
    ++line_num;
    basic_method_.ClearIllegalChar(line);
    basic_method_.RmEndSpace(line);
    basic_method_.RmStartSpace(line);

    std::vector<std::string> v_words;
    basic_method_.Split(line, ' ', v_words);
    for (int i = 0; i < v_words.size(); ++i) {
      std::wstring words_unicode = encoding_conversion_.UTF8ToUnicode(v_words[i].c_str());
      std::wstring words_unicode_tmp;
      bool first_flag = true;
      std::string char_tmp;
      for (std::wstring::iterator iter = words_unicode.begin(); iter != words_unicode.end(); ++iter) {
        words_unicode_tmp = *iter;
        std::string words_tmp = encoding_conversion_.UnicodeToUTF8(words_unicode_tmp);
        if (!first_flag) {
          char_tmp += " ";
        } else {
          first_flag = false;
        }
        char_tmp += words_tmp;
      }
      char_tmp += " </w>";
      ++hm_vocabulary_[char_tmp];
    }
    if (line_num % 1000 == 0) {
      logger<<"\r   Process "<<line_num<<" lines";
    }
  }
  logger<<"\r   Process "<<line_num<<" lines\n";
  return;
}


void BytePairEncoding::GetBigramVocabulary() {
  for (std::unordered_map<std::string, int>::iterator iter = hm_vocabulary_.begin(); iter != hm_vocabulary_.end(); ++iter) {
    mm_vocabulary_.insert(make_pair(iter->second, iter->first));
  }

  for (std::multimap<int, std::string>::reverse_iterator riter = mm_vocabulary_.rbegin(); riter != mm_vocabulary_.rend(); ++riter) {
    v_p_vocabulary_.push_back(std::make_pair(riter->second, riter->first));
  }

  logger<<"\n$$ Get bigram vocabulary\n";
  int i;
  for (i = 0; i < v_p_vocabulary_.size(); ++i) {
    std::vector<std::string> char_tmp;
    basic_method_.Split(v_p_vocabulary_.at(i).first, ' ', char_tmp);
    if (char_tmp.size() < 2) {
      logger<<"   Warning: the size of char_tmp cannot less than 2!\n";
      logger<<"   char_tmp="<<v_p_vocabulary_.at(i).first<<"\n";
      continue;
    } else {
      std::string previous_char = char_tmp.at(0);
      for (int j = 1; j < char_tmp.size(); ++j) {
        std::string current_char = char_tmp.at(j);
        std::string bigram_char = previous_char + " " + current_char;
        hm_bigram_stats_[bigram_char] += v_p_vocabulary_.at(i).second;
        hm_bigram_indices_[bigram_char][i] += 1;
        previous_char = current_char;
      }
    }
    if (i % 10000 == 0) {
      logger<<"\r   Process "<<i<<" words";
    }
  }
  logger<<"\r   Process "<<i<<" words\n";
  return;
}


void BytePairEncoding::UpdateBigramVocabulary(const std::string &max_frequency_bigram, std::vector<BigramUpdateParameters> &v_bigram_update) {

  hm_bigram_stats_[max_frequency_bigram] = 0;
  hm_bigram_indices_[max_frequency_bigram].clear();

  std::vector<std::string> v_target_bigram;
  basic_method_.Split(max_frequency_bigram, ' ', v_target_bigram);
  if (v_target_bigram.size() != 2) {
    logger<<"   Error: format of max_frequency_bigram is incorrect!\n";
    exit(EXIT_FAILURE);
  }
  std::string target_bigram = v_target_bigram.at(0) + v_target_bigram.at(1);

  for(std::vector<BigramUpdateParameters>::iterator iter = v_bigram_update.begin(); \
      iter != v_bigram_update.end(); ++iter) {

    std::vector<std::string> v_old_words;
    basic_method_.Split(iter->old_word_, ' ', v_old_words);

    for(int i = 0; i < v_old_words.size(); ++i) {
      if (v_old_words.at(i) != v_target_bigram.at(0)) {
        continue;
      } else {
        if (i  < v_old_words.size() - 1 && v_old_words[i + 1] == v_target_bigram.at(1)) {
          if (i > 0) {
            std::string previous_string = v_old_words.at(i - 1) + " " + v_old_words.at(i);

            hm_bigram_stats_[previous_string] -= iter->frequency_;
            hm_bigram_indices_[previous_string][iter->index_] -= 1;
          }

          if (i < v_old_words.size() - 2) {
            if (v_old_words[i + 2] != v_target_bigram.at(0) || \
                i >= v_old_words.size() - 3 ||
                v_old_words[i + 3] != v_target_bigram.at(1)) {
              std::string next_string = v_old_words.at(i + 1) + " " + v_old_words.at(i + 2);

              hm_bigram_stats_[next_string] -= iter->frequency_;
              hm_bigram_indices_[next_string][iter->index_] -= 1;
            }
          }
          i += 1;
        }
      }
    }

    std::vector<std::string> v_words;
    basic_method_.Split(iter->word_, ' ', v_words);

    for(int i = 0; i < v_words.size(); ++i) {
      if (v_words.at(i) != target_bigram) {
        continue;
      } else {
        if (i > 0) {
          std::string previous_string = v_words.at(i - 1) + " " + v_words.at(i);
          hm_bigram_stats_[previous_string] += iter->frequency_;
          hm_bigram_indices_[previous_string][iter->index_] += 1;
        }

        if (i < v_words.size() - 1 && v_words.at(i + 1) != target_bigram) {
          std::string next_string = v_words.at(i) + " " + v_words.at(i + 1);
          hm_bigram_stats_[next_string] += iter->frequency_;
          hm_bigram_indices_[next_string][iter->index_] += 1;
        }
      }
    }
  }
  return;
}


void BytePairEncoding::ReplaceBigram(std::string &max_frequency_bigram, std::vector<BigramUpdateParameters> &v_out_bigram_update) {
  std::string target_bigram;
  basic_method_.RemoveAllSpace(max_frequency_bigram, target_bigram);

  target_bigram = " " + target_bigram + " ";
  std::string replacement = " " + max_frequency_bigram + " ";

  for(std::unordered_map<int, int>::iterator iter = hm_bigram_indices_[max_frequency_bigram].begin(); \
      iter != hm_bigram_indices_[max_frequency_bigram].end(); ++iter) {

    if (iter->second < 1) {
      continue;
    }

    std::pair<std::string, int> word_and_frequency = v_p_vocabulary_.at(iter->first);
    
    std::string word = word_and_frequency.first;
    std::string old_word = word;
    word = " " + word + " ";


    std::string::size_type pos;
    while ((pos = word.find(replacement)) != std::string::npos) {
      word.replace(pos, replacement.length(), target_bigram);
    }

    basic_method_.RmStartSpace(word);
    basic_method_.RmEndSpace(word);

    v_p_vocabulary_.at(iter->first).first = word;

    BigramUpdateParameters bigram_update_parameters(iter->first, v_p_vocabulary_.at(iter->first).second, old_word, word);
    v_out_bigram_update.push_back(bigram_update_parameters);
  }
  return;
}


void BytePairEncoding::PruneBigramStats(const int &threshold) {
  std::vector<std::string> v_delete_items;
  for (std::unordered_map<std::string, int>::iterator iter = hm_bigram_stats_.begin(); iter != hm_bigram_stats_.end(); ++iter) {
    if (iter->second < threshold) {
      v_delete_items.push_back(iter->first);
    }
  }

  for(std::vector<std::string>::iterator iter = v_delete_items.begin(); iter != v_delete_items.end(); ++iter) {
    if (hm_bigram_stats_[*iter] < 0) {
      hm_bigram_big_stats_[*iter] += hm_bigram_stats_[*iter];
    } else {
      hm_bigram_big_stats_[*iter] = hm_bigram_stats_[*iter];
    }
    hm_bigram_stats_.erase(*iter);
  }

  return;
}


/*
void BytePairEncoding::GetMaxValueHashMap(const std::hash_map<std::string, int> &in_hash_map, std::string &out_key, int &out_max_value) {
  out_max_value = -1;
  for (std::hash_map<std::string, int>::const_iterator iter = in_hash_map.begin(); iter != in_hash_map.end(); ++iter) {
    if (iter->second > out_max_value) {
      out_max_value = iter->second;
      out_key = iter->first;
    }
  }
  return;
}
*/

void BytePairEncoding::GetMaxValueHashMap(const std::unordered_map<std::string, int> &in_hash_map, std::string &out_key, int &out_max_value) {
  out_max_value = -1;
  for (std::unordered_map<std::string, int>::const_iterator iter = in_hash_map.begin(); iter != in_hash_map.end(); ++iter) {
    if (iter->second > out_max_value) {
      out_max_value = iter->second;
      out_key = iter->first;
    }
  }
  return;
}


void BytePairEncoding::LoadCodes(std::ifstream &in_codes_file) {
  std::string line;
  std::vector<std::string> v_code;
  while (std::getline(in_codes_file, line)) {
    basic_method_.ClearIllegalChar(line);
    basic_method_.RmStartSpace(line);
    basic_method_.RmEndSpace(line);
    v_code.push_back(line);
  }

  int i = v_code.size() - 1;
  for (std::vector<std::string>::reverse_iterator r_iter = v_code.rbegin(); r_iter != v_code.rend(); ++r_iter) {
    hm_codes_[*r_iter] = i;
    --i;
  }

  return;
}


void BytePairEncoding::SegmentBpe(const std::string &input_sentence, std::string &output_sentence) {
  std::vector<std::string> v_words;
  basic_method_.Split(input_sentence, ' ', v_words);

  output_sentence.clear();
  int i;
  for (i = 0; i < v_words.size(); ++i) {
    std::string output_string;
    EncodeBpe(v_words.at(i), output_string);
    if (0 != i) {
      output_sentence += " ";
    }
    output_sentence += output_string;
  }

  return;
}


void BytePairEncoding::EncodeBpe(const std::string &input_word, std::string &output_string) {
  
  output_string.clear();
  std::wstring w_input_word = encoding_conversion_.UTF8ToUnicode(input_word.c_str());
  std::wstring w_character;
  std::vector<std::string> v_chars;
  for (std::wstring::iterator iter = w_input_word.begin(); iter != w_input_word.end(); ++iter) {
    w_character = *iter;
    std::string character = encoding_conversion_.UnicodeToUTF8(w_character);
    v_chars.push_back(character);
  }
  v_chars.push_back("</w>");

  std::set<std::string> s_bigrams;
  GetBigramSet(v_chars, s_bigrams);

  while(true) {
    std::string bigram;
    GetMinBigram(s_bigrams, bigram);
    if (bigram == "") {
      break;
    }

    std::vector<std::string> v_bigram;
    basic_method_.Split(bigram, ' ', v_bigram);
    if (v_bigram.size() != 2) {
      std::cerr<<"Warning: v_bigram.size() must be equal to 2!\n";
      break;
    }

    std::vector<std::string> v_new_chars;
    for (int i = 0; i < v_chars.size(); ++i) {
      if (v_bigram.at(0) != v_chars.at(i)) {
        v_new_chars.push_back(v_chars.at(i));
        continue;
      } else {
        if (i < v_chars.size() - 1 && v_chars.at(i + 1) == v_bigram.at(1)) {
          v_new_chars.push_back(v_bigram.at(0) + v_bigram.at(1));
          ++i;
        } else {
          v_new_chars.push_back(v_chars.at(i));
        }
      }
    }

    v_chars.clear();
    v_chars = v_new_chars;
    if (1 == v_chars.size()) {
      break;
    } else {
      GetBigramSet(v_chars, s_bigrams);
    }
  }

  int i;
  for (i = 0; i < v_chars.size(); ++i) {
    if (v_chars.at(i) != "</w>") {
      if(0 != i) {
        output_string += "@@ ";
      }
      std::string::size_type pos;
      if ((pos = v_chars.at(i).find("</w>")) != std::string::npos) {
        std::string current_wo_eow = v_chars.at(i);
        current_wo_eow.replace(pos, 4, "");
        output_string += current_wo_eow;
      } else {
        output_string += v_chars.at(i);
      }
    } else {
      continue;
    }
  }

  return;
}


void BytePairEncoding::GetBigramSet(const std::vector<std::string> &v_characters, std::set<std::string> &s_bigrams) {
  if (v_characters.size() < 2) {
    std::cerr<<"Warning: v_characters size can not less than 2!\n";
    return;
  }
  s_bigrams.clear();
  std::string previous_char = v_characters.at(0);
  for (int i = 1; i < v_characters.size(); ++i) {
    std::string pair = previous_char + " " + v_characters.at(i);
    s_bigrams.insert(pair);
    previous_char = v_characters.at(i);
  }
  return;
}


void BytePairEncoding::GetMinBigram(const std::set<std::string> &s_bigrams, std::string &bigram) {
  int bigram_score = INT_MAX;
  for (std::set<std::string>::const_iterator iter = s_bigrams.begin(); iter != s_bigrams.end(); ++iter) {
    if (hm_codes_.find(*iter) != hm_codes_.end() && hm_codes_[*iter] < bigram_score) {
      bigram_score = hm_codes_[*iter];
      bigram = *iter;
    }
  }
  return;
}



} // End of namespace neural_machine_translation


char *bpe_segment_result__ = NULL;
neural_machine_translation::BytePairEncoding byte_pair_encoding__; 
//neural_machine_translation::DecoderSentence decoder_sentence__;

void python_bpe_segment_init(char *msg) {
  std::cerr<<"cpp::python_bpe_segment_init::msg="<<msg<<"\n"<<std::flush;
  std::ifstream in_codes_file(msg);
  if (!in_codes_file) {
    std::cerr<<"   Error: can not open "<<msg<<"\n"<<std::flush;
    exit(EXIT_FAILURE);
  }
  byte_pair_encoding__.LoadCodes(in_codes_file);
  in_codes_file.close();
}

char* python_bpe_segment_do_job(char *sentence) {
  if (bpe_segment_result__ != NULL) {
    delete[] bpe_segment_result__;
  }

  std::string input_sentence(sentence);
  byte_pair_encoding__.basic_method_.ClearIllegalChar(input_sentence);
  byte_pair_encoding__.basic_method_.RmStartSpace(input_sentence);
  byte_pair_encoding__.basic_method_.RmEndSpace(input_sentence);


  std::vector<std::string> v_string_generalization;
  byte_pair_encoding__.basic_method_.SplitWithString(input_sentence, " |||| " ,v_string_generalization);

  std::string output_sentence;
  if (0 == v_string_generalization.size()) {
    output_sentence = "";
  } else {
    byte_pair_encoding__.SegmentBpe(v_string_generalization.at(0), output_sentence);
    if (2 == v_string_generalization.size()) {
      std::string output_generalization;
      if (output_sentence == v_string_generalization.at(0)) {
        output_generalization = v_string_generalization.at(1);
      } else {
        byte_pair_encoding__.ModifyGeneralization(output_sentence, v_string_generalization.at(1), output_generalization);
      }
      output_sentence += " |||| " + output_generalization;
    }
  }

#ifdef WIN32
  bpe_segment_result__ = new char[ output_sentence.size() + 1 ];
  strcpy_s(bpe_segment_result__, output_sentence.size() + 1, output_sentence.c_str());
#else
  bpe_segment_result__ = new char[ output_sentence.size() + 1 ];
  strncpy(bpe_segment_result__, output_sentence.c_str(), output_sentence.size() + 1);
#endif

  return bpe_segment_result__;
}






