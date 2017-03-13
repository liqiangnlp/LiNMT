/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 08/08/2017
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
      logger<<"\r   Process "<<line_num<<"lines";
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
      std::cerr<<"   Error: the size of char_tmp cannot less than 2!\n";
      exit(EXIT_FAILURE);
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





} // End of namespace neural_machine_translation



