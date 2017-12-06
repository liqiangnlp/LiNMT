/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */



#include "input_file_preprocess.h"

#include "debug.h"


namespace neural_machine_translation {


////////////////////////////////////// InputFilePreprocess //////////////////////////////////////
////////////////////////////////////// InputFilePreprocess //////////////////////////////////////
////////////////////////////////////// InputFilePreprocess //////////////////////////////////////

bool InputFilePreprocess::PreprocessFilesTrainNeuralLM(int minibatch_size, int max_sent_cutoff, std::string target_file_name, \
                                                       std::string output_file_name, int &target_vocab_size, bool shuffle_flag, \
                                                       std::string model_output_file_name, int hiddenstate_size, int num_layers) {

  int visual_num_target_word_tokens = 0;            // all word tokens
  int visual_total_target_vocab_size = 0;           // vocab size
  int visual_num_single_target_words = 0;           // word count is 1 in corpus
  int visual_num_segment_pairs = 0;                 // number of sentences
  double visual_avg_target_seg_length = 0;          // avg sentence length
  int visual_target_longest_sentence = 0;

  int visual_num_tokens_thrown_away = 0;            // all word tokens thrown away

  target_input_.open(target_file_name.c_str());
  final_output_.open(output_file_name.c_str());


  // first stage is loading all data into RAM
  std::string target_string;
  std::string word;

  int target_length = 0;

  target_input_.clear();

  target_input_.seekg(0, std::ios::beg);
  while (std::getline(target_input_, target_string)) {
    ++target_length;
  }

  visual_num_segment_pairs = target_length;

  if (minibatch_size > target_length) {
    logger<<"Error: minibatch size cannot be greater than the file size\n";
    return false;
  }


  double visual_tmp_running_seg_length = 0;

  // filter any long sentences and get ready to shuffle
  target_input_.clear();
  target_input_.seekg(0, std::ios::beg);

  for (int i = 0; i < target_length; ++i) {
    std::vector<std::string> v_source_sentence;
    std::vector<std::string> v_target_sentence;

    std::getline(target_input_, target_string);

    std::istringstream iss_target(target_string, std::istringstream::in);

    while (iss_target >> word) {
      v_target_sentence.push_back(word);
    }

    if (!(v_source_sentence.size() + 1 >= max_sent_cutoff - 2 || v_target_sentence.size() + 1 >= max_sent_cutoff - 2)) {
      data_.push_back(CombineSentenceInformation(v_source_sentence, v_target_sentence));

      visual_tmp_running_seg_length += v_target_sentence.size();
      visual_num_target_word_tokens += v_target_sentence.size();

      if (visual_target_longest_sentence < v_target_sentence.size()) {
        visual_target_longest_sentence = v_target_sentence.size();
      }
    } else {
      visual_num_tokens_thrown_away += v_source_sentence.size() + v_target_sentence.size();
    }
  }

  visual_avg_target_seg_length = visual_tmp_running_seg_length / ((double)visual_num_segment_pairs);

  // shuffle the entire data
  if (shuffle_data_mode__) {
    std::random_shuffle(data_.begin(), data_.end());
  }


  // remove last sentences that do not fit in the minibatch
  //if (data_.size() % minibatch_size != 0) {
  //  int num_to_remove = data_.size() % minibatch_size;
  //  for (int i = 0; i < num_to_remove; ++i) {
  //    data_.pop_back();
  //  }
  //}

  if (data_.size() == 0) {
    logger<<"Error: file size is zero, could be wrong input file or all lines are above max sentence length\n";
    return false;
  }

  // sort the data based on minibatch
  CompareNeuralMT compare_neural_mt;
  int current_index = 0;
  while (current_index < data_.size()) {
    if (current_index + minibatch_size * minibatch_mult_ <= data_.size()) {
      std::sort(data_.begin() + current_index, data_.begin() + current_index + minibatch_size * minibatch_mult_, compare_neural_mt);
      current_index += minibatch_size * minibatch_mult_;
    } else {
      std::sort(data_.begin() + current_index, data_.end(), compare_neural_mt);
      break;
    }
  }


  // now get counts for mappings
  
  for (int i = 0; i < data_.size(); ++i) {

    // for target sentences
    for (int j = 0; j < data_[i].v_target_sentence_.size(); ++j) {
      if (data_[i].v_target_sentence_[j] != "<UNK>") {
        if (0 == uno_target_counts_.count(data_[i].v_target_sentence_[j])) {
          uno_target_counts_[data_[i].v_target_sentence_[j]] = 1;
        } else {
          uno_target_counts_[data_[i].v_target_sentence_[j]] += 1;
        }
      }
    }
  }

  // now use heap to get the highest source and target mappings
  if (-1 == target_vocab_size) {
    target_vocab_size = uno_target_counts_.size() + 3;
  }

  visual_total_target_vocab_size = uno_target_counts_.size();

  target_vocab_size = std::min(target_vocab_size, (int)uno_target_counts_.size() + 3);

#ifdef DEBUG_NEWCHECKPOINT_1
  std::cerr<<"   source_vocab_size: "<<source_vocab_size<<"\n"
           <<"   target_vocab_size: "<<target_vocab_size<<"\n"
           <<std::flush;
#endif

  // output the model information to first line of output weights file
  std::ofstream output_model;
  output_model.open(model_output_file_name.c_str());
  output_model<<num_layers<<" "<<hiddenstate_size<<" "<<target_vocab_size<<"\n";

  std::priority_queue<MappingPair, std::vector<MappingPair>, MappingPairCompareFunctor> pq_target_map_heap;

  // push heap for the target side
  for (auto it = uno_target_counts_.begin(); it != uno_target_counts_.end(); ++it) {
    pq_target_map_heap.push(MappingPair(it->first, it->second));
    if (1 == it->second) {
      ++ visual_num_single_target_words;
    }
  }
  

  
  output_model<<"==========================================================\n";

  // output target vocab
  uno_target_mapping_["<START>"] = 0;
  uno_target_mapping_["<EOF>"] = 1;
  uno_target_mapping_["<UNK>"] = 2;
  output_model<<0<<" <START>\n"
              <<1<<" <EOF>\n"
              <<2<<" <UNK>\n";

  for (int i = 3; i < target_vocab_size; ++i) {
    uno_target_mapping_[pq_target_map_heap.top().word_] = i;
    output_model<<i<<" "<<pq_target_map_heap.top().word_<<"\n";
    pq_target_map_heap.pop();
  }

  output_model<<"==========================================================\n";

  // now integerize
  for (int i = 0; i < data_.size(); ++i) {
    std::vector<int> v_target_int;

    // for the target sentence
    for (int j = 0; j < data_[i].v_target_sentence_.size(); ++j) {
      if (0 == uno_target_mapping_.count(data_[i].v_target_sentence_[j])) {
        v_target_int.push_back(uno_target_mapping_["<UNK>"]);
      } else {
        v_target_int.push_back(uno_target_mapping_[data_[i].v_target_sentence_[j]]);  
      }
    }

    data_[i].v_target_sentence_.clear();
    data_[i].v_target_sentence_int_i_ = v_target_int;
    data_[i].v_target_sentence_int_o_ = v_target_int;
    // insert <START> to the first position
    data_[i].v_target_sentence_int_i_.insert(data_[i].v_target_sentence_int_i_.begin(), 0);
    // push back <EOF>
    data_[i].v_target_sentence_int_o_.push_back(1);
  }

  // now pad based on minibatch
  current_index = 0;
  while (current_index < data_.size()) {
    int max_target_minibatch = 0;

    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {
      if (data_[i].v_target_sentence_int_i_.size() > max_target_minibatch) {
        max_target_minibatch = data_[i].v_target_sentence_int_i_.size();
      }
    }

    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {
     
      // NMT is: while (data_[i].v_target_sentence_int_i_.size() < max_target_minibatch)
      // bug ??????????
      while (data_[i].v_target_sentence_int_i_.size() <= max_target_minibatch) {
        data_[i].v_target_sentence_int_i_.push_back(-1);
        data_[i].v_target_sentence_int_o_.push_back(-1);
      }
    }
    current_index += minibatch_size;
  }


  // now add in all -1's to make the last minibatch complete
  int num_extra_to_add = minibatch_size - data_.size() % minibatch_size;
  if(num_extra_to_add == minibatch_size) {
      num_extra_to_add = 0;
  }

  int target_sent_len = data_.back().v_target_sentence_int_i_.size();
  for (int i = 0; i < num_extra_to_add; ++i) {
    std::vector<std::string> v_src_sentence;
    std::vector<std::string> v_tgt_sentence;

    data_.push_back(CombineSentenceInformation(v_src_sentence, v_tgt_sentence));

    std::vector<int> v_tgt_int_m1;
    for (int j = 0; j < target_sent_len; ++j) {
      v_tgt_int_m1.push_back(-1);
    }

    data_.back().v_target_sentence_int_i_ = v_tgt_int_m1;
    data_.back().v_target_sentence_int_o_ = v_tgt_int_m1;
  }



  // now output to the file
  for (int i = 0; i < data_.size(); ++i) {

    final_output_<<"\n";
    final_output_<<"\n";

    // output the v_target_sentence_int_i_
    for (int j = 0; j < data_[i].v_target_sentence_int_i_.size(); ++j) {
      final_output_<<data_[i].v_target_sentence_int_i_[j];
      if (j != data_[i].v_target_sentence_int_i_.size()) {
        final_output_<<" ";
      }
    }
    final_output_<<"\n";

    for (int j = 0; j < data_[i].v_target_sentence_int_o_.size(); ++j) {
      final_output_<<data_[i].v_target_sentence_int_o_[j];
      if (j != data_[i].v_target_sentence_int_o_.size()) {
        final_output_<<" ";
      }
    }
    final_output_<<"\n";
  }
  final_output_.close();
  target_input_.close();

  // print file stats
  logger<<"\n$$ Target Train File Information\n"
        <<"   Number of target word tokens : "<<visual_num_target_word_tokens<<"\n"
        <<"   Traget vocabulary size (before <unk>) : "<<visual_total_target_vocab_size<<"\n"
        <<"   Number of target singleton word types : "<<visual_num_single_target_words<<"\n"
        <<"   Number of sentences : "<<visual_num_segment_pairs<<"\n"
        <<"   Average sentence length : "<<visual_avg_target_seg_length<<"\n"
        <<"   Longest target segment (after removing long sentences for training) : "<<visual_target_longest_sentence<<"\n"
        <<"   Total word tokens thrown out due to sentence cutoff (source + target) : "<<visual_num_tokens_thrown_away<<"\n";

  return true;
}


bool InputFilePreprocess::PreprocessFilesTrainNeuralMT(int minibatch_size, int max_sent_cutoff, \
                                  std::string source_file_name, std::string target_file_name, std::string output_file_name, \
                                  int &source_vocab_size, int &target_vocab_size, \
                                  bool shuffle_mode, std::string model_output_file_name, \
                                  int hiddenstate_size, int num_layers, bool unk_replace_mode, int unk_align_range, bool attention_mode) {

#ifdef DEBUG_DROPOUT
  std::cerr << "************NCP1 In *InputFilePreprocess* *PreprocessFilesTrainNeuralMT*\n" << std::flush;
  std::cerr<<"  minibatch_size: "<<minibatch_size<<"\n"
           <<"  max_sent_cutoff: "<<max_sent_cutoff<<"\n"
           <<"  source_file_name: "<<source_file_name<<"\n"
           <<"  target_file_name: "<<target_file_name<<"\n"
           <<"  output_file_name: "<<output_file_name<<"\n"
           <<"  source_vocab_size: "<<source_vocab_size<<"\n"
           <<"  target_vocab_size: "<<target_vocab_size<<"\n"
           <<"  shuffle_mode: "<<shuffle_mode<<"\n"
           <<"  model_output_file_name: "<<model_output_file_name<<"\n"
           <<"  hiddenstate_size: "<<hiddenstate_size<<"\n"
           <<"  num_layers: "<<num_layers<<"\n"
           <<"  unk_replace_mode: "<<unk_replace_mode<<"\n"
           <<"  unk_align_range: "<<unk_align_range<<"\n"
           <<"  attention_mode: "<<attention_mode<<"\n"<<std::flush;
#endif


  int visual_num_source_word_tokens = 0;
  int visual_total_source_vocab_size = 0;
  int visual_num_single_source_words = 0;
  double visual_avg_source_seg_length = 0;
  int visual_source_longest_sentence = 0;

  int visual_num_target_word_tokens = 0;
  int visual_total_target_vocab_size = 0;
  int visual_num_single_target_words = 0;
  double visual_avg_target_seg_length = 0;
  int visual_target_longest_sentence = 0;

  int visual_num_segment_pairs = 0;
  int visual_num_tokens_thrown_away = 0;

  source_input_.open(source_file_name.c_str());
  target_input_.open(target_file_name.c_str());
  final_output_.open(output_file_name.c_str());


  // first stage is loading all data into RAM
  std::string source_string;
  std::string target_string;
  std::string word;

  int source_length = 0;
  int target_length = 0;

  source_input_.clear();
  target_input_.clear();

  source_input_.seekg(0, std::ios::beg);
  while (std::getline(source_input_, source_string)) {
    ++source_length;
  }

  target_input_.seekg(0, std::ios::beg);
  while (std::getline(target_input_, target_string)) {
    ++target_length;
  }

  visual_num_segment_pairs = target_length;

  // do check to be sure the two files are the same length
  if (source_length != target_length) {
    logger<<"Error: Input files are not the same length\n";
    return false;
  }

  if (minibatch_size > source_length) {
    logger<<"Error: minibatch size cannot be greater than the file size\n";
    return false;
  }


  // filter any long sentences and get ready to shuffle
  source_input_.clear();
  target_input_.clear();
  source_input_.seekg(0, std::ios::beg);
  target_input_.seekg(0, std::ios::beg);

  for (int i = 0; i < source_length; ++i) {
    std::vector<std::string> v_source_sentence;
    std::vector<std::string> v_target_sentence;

    std::getline(source_input_, source_string);
    std::getline(target_input_, target_string);

    std::istringstream iss_source(source_string, std::istringstream::in);
    std::istringstream iss_target(target_string, std::istringstream::in);

    while (iss_source >> word) {
      v_source_sentence.push_back(word);
    }

    while (iss_target >> word) {
      v_target_sentence.push_back(word);
    }

    if (!(v_source_sentence.size() + 1 >= max_sent_cutoff - 2 || v_target_sentence.size() + 1 >= max_sent_cutoff - 2)) {
      data_.push_back(CombineSentenceInformation(v_source_sentence, v_target_sentence));
      visual_avg_source_seg_length += v_source_sentence.size();
      visual_avg_target_seg_length += v_target_sentence.size();
      visual_num_source_word_tokens += v_source_sentence.size();
      visual_num_target_word_tokens += v_target_sentence.size();

      if (visual_source_longest_sentence < v_source_sentence.size()) {
        visual_source_longest_sentence = v_source_sentence.size();
      }
      if (visual_target_longest_sentence < v_target_sentence.size()) {
        visual_target_longest_sentence = v_target_sentence.size();
      }
    } else {
      visual_num_tokens_thrown_away += v_source_sentence.size() + v_target_sentence.size();
    }
  }
  visual_avg_source_seg_length = visual_avg_source_seg_length / ((double)visual_num_segment_pairs);
  visual_avg_target_seg_length = visual_avg_target_seg_length / ((double)visual_num_segment_pairs);


#ifdef DEBUG_DROPOUT
  std::cerr << "   shuffle_data_mode: " << shuffle_data_mode__ << "\n";
#endif

  // shuffle the entire data
  if (shuffle_data_mode__) {
    std::random_shuffle(data_.begin(), data_.end());
  }


  // remove last sentences that do not fit in the minibatch
  if (data_.size() % minibatch_size != 0) {
    int num_to_remove = data_.size() % minibatch_size;
    for (int i = 0; i < num_to_remove; ++i) {
      data_.pop_back();
    }
  }

  if (data_.size() == 0) {
    logger<<"Error: file size is zero, could be wrong input file or all lines are above max sentence length\n";
    return false;
  }

  // sort the data based on minibatch
  CompareNeuralMT compare_neural_mt;
  int current_index = 0;
  while (current_index < data_.size()) {
    if (current_index + minibatch_size * minibatch_mult_ <= data_.size()) {
      std::sort(data_.begin() + current_index, data_.begin() + current_index + minibatch_size * minibatch_mult_, compare_neural_mt);
      current_index += minibatch_size * minibatch_mult_;
    } else {
      std::sort(data_.begin() + current_index, data_.end(), compare_neural_mt);
      break;
    }
  }


  // now get counts for mappings
  for (int i = 0; i < data_.size(); ++i) {
    // for source sentences
    for (int j = 0; j < data_[i].v_source_sentence_.size(); ++j) {
      if (data_[i].v_source_sentence_[j] != "<UNK>") {
        if (0 == uno_source_counts_.count(data_[i].v_source_sentence_[j])) {
          uno_source_counts_[data_[i].v_source_sentence_[j]] = 1;
        } else {
          uno_source_counts_[data_[i].v_source_sentence_[j]] += 1;
        }
      }   
    }

    // for target sentences
    for (int j = 0; j < data_[i].v_target_sentence_.size(); ++j) {
      if (data_[i].v_target_sentence_[j] != "<UNK>") {
        if (0 == uno_target_counts_.count(data_[i].v_target_sentence_[j])) {
          uno_target_counts_[data_[i].v_target_sentence_[j]] = 1;
        } else {
          uno_target_counts_[data_[i].v_target_sentence_[j]] += 1;
        }
      }
    }
  }

  // now use heap to get the highest source and target mappings
  if (-1 == source_vocab_size) {
    source_vocab_size = uno_source_counts_.size() + 1;
  }
  if (-1 == target_vocab_size) {
    if (!unk_replace_mode) {
      target_vocab_size = uno_target_counts_.size() + 3;
    } else {
      target_vocab_size = uno_target_counts_.size() + 3 + 1 + unk_align_range * 2;
    }
  }

  visual_total_source_vocab_size = uno_source_counts_.size();
  visual_total_target_vocab_size = uno_target_counts_.size();

  if (!unk_replace_mode) {
    source_vocab_size = std::min(source_vocab_size, (int)uno_source_counts_.size() + 1);
    target_vocab_size = std::min(target_vocab_size, (int)uno_target_counts_.size() + 3);
  }

#ifdef DEBUG_NEWCHECKPOINT_1
  std::cerr<<"   source_vocab_size: "<<source_vocab_size<<"\n"
           <<"   target_vocab_size: "<<target_vocab_size<<"\n"
           <<std::flush;
#endif

  // output the model information to first line of output weights file
  std::ofstream output_model;
  output_model.open(model_output_file_name.c_str());
  output_model<<num_layers<<" "<<hiddenstate_size<<" "<<target_vocab_size<<" "<<source_vocab_size<<"\n";

  std::priority_queue<MappingPair, std::vector<MappingPair>, MappingPairCompareFunctor> pq_source_map_heap;
  std::priority_queue<MappingPair, std::vector<MappingPair>, MappingPairCompareFunctor> pq_target_map_heap;

  // push heap for the source side
  for (auto it = uno_source_counts_.begin(); it != uno_source_counts_.end(); ++it) {
    pq_source_map_heap.push(MappingPair(it->first, it->second));
    if (1 == it->second) {
      ++ visual_num_single_source_words;
    }
  }

  // push heap for the target side
  for (auto it = uno_target_counts_.begin(); it != uno_target_counts_.end(); ++it) {
    pq_target_map_heap.push(MappingPair(it->first, it->second));
    if (1 == it->second) {
      ++ visual_num_single_target_words;
    }
  }
  

  if (!unk_replace_mode) {
    // do not replace <UNK>
    output_model<<"==========================================================\n";

    // output source vocab
    //uno_source_mapping_["<START>"] = 0;
    uno_source_mapping_["<UNK>"] = 0;
    //output_model<<0<<" <START>\n"
    //            <<1<<" <UNK>\n";
    output_model<<0<<" <UNK>"<<"\n";

    for (int i = 1; i < source_vocab_size; ++i) {
      uno_source_mapping_[pq_source_map_heap.top().word_] = i;
      output_model<<i<<" "<<pq_source_map_heap.top().word_<<"\n";
      pq_source_map_heap.pop();
    }
    output_model<<"==========================================================\n";

    // output target vocab
    uno_target_mapping_["<START>"] = 0;
    uno_target_mapping_["<EOF>"] = 1;
    uno_target_mapping_["<UNK>"] = 2;
    output_model<<0<<" <START>\n"
                <<1<<" <EOF>\n"
                <<2<<" <UNK>\n";

    for (int i = 3; i < target_vocab_size; ++i) {
      uno_target_mapping_[pq_target_map_heap.top().word_] = i;
      output_model<<i<<" "<<pq_target_map_heap.top().word_<<"\n";
      pq_target_map_heap.pop();
    }

    output_model<<"==========================================================\n";
  } else {
    // replace <UNK>
    output_model<<"==========================================================\n";

    // output source vocab
    //uno_source_mapping_["<START>"] = 0;
    uno_source_mapping_["<UNK>"] = 0;
    //output_model<<0<<" <START>\n"
    //            <<1<<" <UNK>\n";
    output_model<<0<<" <UNK>\n";

    for (int i = 1; i < source_vocab_size; ++i) {
      uno_source_mapping_[pq_source_map_heap.top().word_] = i;
      output_model<<i<<" "<<pq_source_map_heap.top().word_<<"\n";
      pq_source_map_heap.pop();
    }
    output_model << "==========================================================\n";

    // output target vocab
    uno_target_mapping_["<START>"] = 0;
    uno_target_mapping_["<EOF>"] = 1;
    uno_target_mapping_["<UNK>NULL"] = 2;
    output_model<<0<<" <START>\n"
                <<1<<" <EOF>\n"
                <<2<<" <UNK>NULL\n";

    int current_index = 3;
    for (int i = -unk_align_range; i < unk_align_range + 1; ++i) {
      uno_target_mapping_["<UNK>" + std::to_string(i)] = current_index;
      output_model<<current_index<<" "<<"<UNK>" + std::to_string(i)<<"\n";
      ++current_index;
    }

    for (int i = current_index; i < target_vocab_size; ++i) {
      if (0 == uno_target_mapping_.count(pq_target_map_heap.top().word_)) {
        uno_target_mapping_[pq_target_map_heap.top().word_] = i;
        output_model<<i<<" "<<pq_target_map_heap.top().word_<<"\n";
      }
      pq_target_map_heap.pop();
    }
    output_model<<"==========================================================\n";
  }

  // now integerize
  for (int i = 0; i < data_.size(); ++i) {
    std::vector<int> v_source_int;
    std::vector<int> v_target_int;

    // for the source sentence
    for (int j = 0; j < data_[i].v_source_sentence_.size(); ++j) {
      if (0 == uno_source_mapping_.count(data_[i].v_source_sentence_[j])) {
        v_source_int.push_back(uno_source_mapping_["<UNK>"]);
      } else {
        v_source_int.push_back(uno_source_mapping_[data_[i].v_source_sentence_[j]]);
      }
    }

    std::reverse(v_source_int.begin(), v_source_int.end());
    data_[i].v_source_sentence_.clear();
    data_[i].v_source_sentence_int_ = v_source_int;
    // insert <START> to the first position
    //data_[i].v_source_sentence_int_.insert(data_[i].v_source_sentence_int_.begin(), 0);

    while (data_[i].v_minus_two_source_.size() != data_[i].v_source_sentence_int_.size()) {
      data_[i].v_minus_two_source_.push_back(-2);
    }

    // for the target sentence
    for (int j = 0; j < data_[i].v_target_sentence_.size(); ++j) {
      if (0 == uno_target_mapping_.count(data_[i].v_target_sentence_[j])) {
        
        if (0 == uno_target_mapping_.count("<UNK>")) {
          v_target_int.push_back(uno_target_mapping_["<UNK>NULL"]);
        } else {
          v_target_int.push_back(uno_target_mapping_["<UNK>"]);
        }
      } else {
        v_target_int.push_back(uno_target_mapping_[data_[i].v_target_sentence_[j]]);  
      }
    }

    data_[i].v_target_sentence_.clear();
    data_[i].v_target_sentence_int_i_ = v_target_int;
    data_[i].v_target_sentence_int_o_ = v_target_int;
    // insert <START> to the first position
    data_[i].v_target_sentence_int_i_.insert(data_[i].v_target_sentence_int_i_.begin(), 0);
    // push back <EOF>
    data_[i].v_target_sentence_int_o_.push_back(1);
  }

  // now pad based on minibatch
  current_index = 0;
  while (current_index < data_.size()) {
    int max_source_minibatch = 0;
    int max_target_minibatch = 0;

    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {
      if (data_[i].v_source_sentence_int_.size() > max_source_minibatch) {
        max_source_minibatch = data_[i].v_source_sentence_int_.size();
      }
      if (data_[i].v_target_sentence_int_i_.size() > max_target_minibatch) {
        max_target_minibatch = data_[i].v_target_sentence_int_i_.size();
      }
    }

    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {

      while (data_[i].v_source_sentence_int_.size() < max_source_minibatch) {
        data_[i].v_source_sentence_int_.insert(data_[i].v_source_sentence_int_.begin(), -1);
        data_[i].v_minus_two_source_.insert(data_[i].v_minus_two_source_.begin(), -1);
      }

      while (data_[i].v_target_sentence_int_i_.size() < max_target_minibatch) {
        data_[i].v_target_sentence_int_i_.push_back(-1);
        data_[i].v_target_sentence_int_o_.push_back(-1);
      }
    }
    current_index += minibatch_size;
  }

  // now output to the file
  for (int i = 0; i < data_.size(); ++i) {

    // output the v_source_sentence_int_
    for (int j = 0; j < data_[i].v_source_sentence_int_.size(); ++j) {
      final_output_<<data_[i].v_source_sentence_int_[j];
      if (j != data_[i].v_source_sentence_int_.size()) {
        final_output_<<" ";
      }
    }
    final_output_<<"\n";

    // output the v_minus_two_source
    for (int j = 0; j < data_[i].v_minus_two_source_.size(); ++j) {
      final_output_<<data_[i].v_minus_two_source_[j];
      if (j != data_[i].v_minus_two_source_.size()) {
        final_output_<<" ";
      }
    }
    final_output_ << "\n";

    // output the v_target_sentence_int_i_
    for (int j = 0; j < data_[i].v_target_sentence_int_i_.size(); ++j) {
      final_output_<<data_[i].v_target_sentence_int_i_[j];
      if (j != data_[i].v_target_sentence_int_i_.size()) {
        final_output_<<" ";
      }
    }
    final_output_<<"\n";

    for (int j = 0; j < data_[i].v_target_sentence_int_o_.size(); ++j) {
      final_output_<<data_[i].v_target_sentence_int_o_[j];
      if (j != data_[i].v_target_sentence_int_o_.size()) {
        final_output_<<" ";
      }
    }
    final_output_<<"\n";
  }
  final_output_.close();
  source_input_.close();
  target_input_.close();

  // print file stats
  logger<<"\n$$ Source & target training file\n"
        <<"   Number of src/tgt tokens : "<<visual_num_source_word_tokens<<" / "<<visual_num_target_word_tokens<<"\n"
        <<"   Number of src/tgt vocab  : "<<visual_total_source_vocab_size<<" / "<<visual_total_target_vocab_size<<"\n"
        //<<"   Number of source singleton word types : "<<visual_num_single_source_words<<"\n"
        <<"   Average src/tgt length   : "<<visual_avg_source_seg_length<<" / "<<visual_avg_target_seg_length<<"\n"
        <<"   Longest src/tgt length   : "<<visual_source_longest_sentence<<" / "<<visual_target_longest_sentence<<"\n"
        <<"   Src&tgt sentences        : "<<visual_num_segment_pairs<<"\n"
        <<"   Src&tgt tokens thrown out: "<<visual_num_tokens_thrown_away<<"\n";
  return true;
}

bool InputFilePreprocess::IntegerizeFileNeuralLM(std::string output_weights_name, std::string target_file_name, \
                                                 std::string tmp_output_name, int max_sent_cutoff, int minibatch_size, \
                                                 bool dev_flag, int &hiddenstate_size, int &target_vocab_size, int &num_layers) {

  int visual_num_target_word_tokens = 0;        // number of word tokens
  int visual_target_longest_sent = 0;           // number of words in longest sentences

  int visual_num_segment_pairs = 0;             // number of sentences

  int visual_num_tokens_thrown_away = 0;        // total words thrown away

  std::ifstream weights_file;
  weights_file.open(output_weights_name.c_str());

  std::vector<std::string> v_file_input;
  std::string str;
  std::string word;

  std::getline(weights_file, str);
  std::istringstream iss(str, std::istringstream::in);
  while (iss >> word) {
    v_file_input.push_back(word);
  }

  //if (v_file_input.size() != 4) {
  //  std::cerr<<"Error: neural network file format has been corrupted\n";
  //  exit(EXIT_FAILURE);
  //}

  num_layers = std::stoi(v_file_input[0]);
  hiddenstate_size = std::stoi(v_file_input[1]);
  target_vocab_size = std::stoi(v_file_input[2]);

  // now get the mappings
  // skip all equals *=====...
  std::getline(weights_file, str);
  while (std::getline(weights_file, str)) {
    int tmp_index;
    if (str.size() > 3 && '=' == str[0] && '=' == str[1] && '=' == str[2]) {
      // done with target mapping
      break;
    }

    std::istringstream iss(str, std::istringstream::in);
    iss>>word;
    tmp_index = std::stoi(word);
    iss>>word;
    uno_target_mapping_[word] = tmp_index;
  }

  // we have the mappings, integerize the file
  std::ofstream final_output;
  final_output.open(tmp_output_name.c_str());
  std::ifstream target_input;
  target_input.open(target_file_name.c_str());

  // first get the number of lines the files and check to be sure they are the same
  int target_length = 0;
  std::string target_string;

  target_input.clear();


  target_input.seekg(0, std::ios::beg);
  while (std::getline(target_input, target_string)) {
    ++target_length;
  }

  visual_num_segment_pairs = target_length;

  target_input.clear();
  target_input.seekg(0, std::ios::beg);

  for (int i = 0; i < target_length; ++i) {
    std::vector<std::string> v_source_sentence;
    std::vector<std::string> v_target_sentence;

    std::getline(target_input, target_string);

    std::istringstream iss_target(target_string, std::istringstream::in);

    while (iss_target>>word) {
      v_target_sentence.push_back(word);
    }

    if (!(v_target_sentence.size() + 1 >= max_sent_cutoff - 2)) {
      data_.push_back(CombineSentenceInformation(v_source_sentence, v_target_sentence));

      visual_num_target_word_tokens += v_target_sentence.size();

      if (visual_target_longest_sent < v_target_sentence.size()) {
        visual_target_longest_sent = v_target_sentence.size();
      }
    } else {
      visual_num_tokens_thrown_away += v_source_sentence.size() + v_target_sentence.size();
    }
  }

  if (0 == data_.size()) {
    logger<<"Error: file size is zero, could be wrong input file or all lines are above max sentence length\n";
    exit(EXIT_FAILURE);
  }

  CompareNeuralMT compare_neural_lm;
  int curr_index = 0;

  if (minibatch_size != 1) {
    while (curr_index < data_.size()) {
      if (curr_index + minibatch_size * minibatch_mult_ <= data_.size()) {
        std::sort(data_.begin() + curr_index, data_.begin() + curr_index + minibatch_size * minibatch_mult_, compare_neural_lm);
        curr_index += minibatch_size * minibatch_mult_;
      } else {
        std::sort(data_.begin() + curr_index, data_.end(), compare_neural_lm);
        break;
      }
    }
  }

  // now integerize
  for (int i = 0; i < data_.size(); ++i) {
    std::vector<int> v_target_int;

    // for the target side
    for (int j = 0; j < data_[i].v_target_sentence_.size(); ++j) {
      if (0 == uno_target_mapping_.count(data_[i].v_target_sentence_[j])) {
        v_target_int.push_back(uno_target_mapping_["<UNK>"]);
      } else {
        v_target_int.push_back(uno_target_mapping_[data_[i].v_target_sentence_[j]]);
      }
    }
    data_[i].v_target_sentence_.clear();
    data_[i].v_target_sentence_int_i_ = v_target_int;
    data_[i].v_target_sentence_int_o_ = v_target_int;
    data_[i].v_target_sentence_int_i_.insert(data_[i].v_target_sentence_int_i_.begin(), 0);
    data_[i].v_target_sentence_int_o_.push_back(1);
  }

  // now pad
  int current_index = 0;
  while (current_index < data_.size()) {
    int max_target_minibatch = 0;
    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {
      if (data_[i].v_target_sentence_int_i_.size() > max_target_minibatch) {
        max_target_minibatch = data_[i].v_target_sentence_int_i_.size();
      }
    }

    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {
      // this is neural mt, bug ????????
      //while (data_[i].v_target_sentence_int_i_.size() < max_target_minibatch) {
      while (data_[i].v_target_sentence_int_i_.size() <= max_target_minibatch) {
        data_[i].v_target_sentence_int_i_.push_back(-1);
        data_[i].v_target_sentence_int_o_.push_back(-1);
      }
    }
    current_index += minibatch_size;
  }

  int num_extra_to_add = minibatch_size - data_.size() % minibatch_size;
  if (num_extra_to_add == minibatch_size) {
    num_extra_to_add = 0;
  }
  int target_sent_len = data_.back().v_target_sentence_int_i_.size();
  for (int i = 0; i < num_extra_to_add; ++i) {
    std::vector<std::string> v_src_sentence;
    std::vector<std::string> v_tgt_sentence;
    data_.push_back(CombineSentenceInformation(v_src_sentence, v_tgt_sentence));

    std::vector<int> v_tgt_int_m1;
    for (int j = 0; j < target_sent_len; ++j) {
      v_tgt_int_m1.push_back(-1);
    }

    data_.back().v_target_sentence_int_i_ = v_tgt_int_m1;
    data_.back().v_target_sentence_int_o_ = v_tgt_int_m1;
  }

  // output validation file
  for (int i = 0; i < data_.size(); ++i) {
 
    final_output<<"\n";
    final_output<<"\n";

    // output v_target_sentence_int_i
    for (int j = 0; j < data_[i].v_target_sentence_int_i_.size(); ++j) {
      final_output<<data_[i].v_target_sentence_int_i_[j];
      if (j != data_[i].v_target_sentence_int_i_.size()) {
        final_output<<" ";
      }
    }
    final_output<<"\n";

    // output v_target_sentence_int_o
    for (int j = 0; j < data_[i].v_target_sentence_int_o_.size(); ++j) {
      final_output<<data_[i].v_target_sentence_int_o_[j];
      if (j != data_[i].v_target_sentence_int_o_.size()) {
        final_output<<" ";
      }
    }
    final_output<<"\n";
  }

  weights_file.close();
  final_output.close();
  target_input.close();

  logger<<"\n$$ Target Dev File Information\n"
        <<"   Number of target word tokens : "<<visual_num_target_word_tokens<<"\n"
        <<"   Number of sentences : "<<visual_num_segment_pairs<<"\n"
        <<"   Longest sentence (after removing long sentences for training) : "<<visual_target_longest_sent<<"\n"
        <<"   Total word tokens thrown out due to sentence cutoff : "<<visual_num_tokens_thrown_away<<"\n";
  return true;
}



bool InputFilePreprocess::IntegerizeFileNeuralMT(std::string output_weights_name, std::string source_file_name, std::string target_file_name, \
    std::string tmp_output_name, int max_sent_cutoff, int minibatch_size, int &hiddenstate_size, \
    int &source_vocab_size, int &target_vocab_size, int &num_layers, bool attention_mode, bool multi_source_mode, std::string multi_source_file, std::string tmp_output_name_ms, std::string ms_mapping_file) {

#ifdef DEBUG_NEWCHECKPOINT_1
  std::cerr << "************ In *InputFilePreprocess* *IntegerizeFileNeuralMT*\n" << std::flush;
  std::cerr<<"   output_weights_name: "<<output_weights_name<<"\n"
           <<"   source_file_name: "<<source_file_name<<"\n"
           <<"   target_file_name: "<<target_file_name<<"\n"
           <<"   tmp_output_name: "<<tmp_output_name<<"\n"
           <<"   max_sent_cutoff: "<<max_sent_cutoff<<"\n"
           <<"   minibatch_size: "<<minibatch_size<<"\n"
           <<"   hiddenstate_size: "<<hiddenstate_size<<"\n"
           <<"   source_vocab_size: "<<source_vocab_size<<"\n"
           <<"   target_vocab_size: "<<target_vocab_size<<"\n"
           <<"   num_layers: "<<num_layers<<"\n"
           <<"   attention_mode: "<<attention_mode<<"\n"
           <<std::flush;
#endif


  int visual_num_source_word_tokens = 0;
  int visual_source_longest_sent = 0;

  int visual_num_target_word_tokens = 0;
  int visual_target_longest_sent = 0;

  int visual_num_segment_pairs = 0;

  int visual_num_tokens_thrown_away = 0;

  std::ifstream weights_file;
  weights_file.open(output_weights_name.c_str());

  std::vector<std::string> v_file_input;
  std::string str;
  std::string word;

  std::getline(weights_file, str);
  std::istringstream iss(str, std::istringstream::in);
  while (iss >> word) {
    v_file_input.push_back(word);
  }

  //if (v_file_input.size() != 4) {
  //  std::cerr<<"Error: neural network file format has been corrupted\n";
  //  exit(EXIT_FAILURE);
  //}

  num_layers = std::stoi(v_file_input[0]);
  hiddenstate_size = std::stoi(v_file_input[1]);
  target_vocab_size = std::stoi(v_file_input[2]);
  source_vocab_size = std::stoi(v_file_input[3]);

  // now get the mappings
  // skip all equals *=====...
  std::getline(weights_file, str);
  while (std::getline(weights_file, str)) {
    int tmp_index;
    if (str.size() > 3 && '=' == str[0] && '=' == str[1] && '=' == str[2]) {
      // done with source mapping
      break;
    }

    std::istringstream iss(str, std::istringstream::in);
    iss>>word;
    tmp_index = std::stoi(word);
    iss>>word;
    uno_source_mapping_[word] = tmp_index;
  }

  while (std::getline(weights_file, str)) {
    int tmp_index;
    if (str.size() > 3 && '=' == str[0] && '=' == str[1] && '=' == str[2]) {
      // done with target mapping
      break;
    }

    std::istringstream iss(str, std::istringstream::in);
    iss>>word;
    tmp_index = std::stoi(word);
    iss>>word;
    uno_target_mapping_[word] = tmp_index;
  }

  /*
    here for multi-source, not write ...
  */



  // we have the mappings, integerize the file
  std::ofstream final_output;
  final_output.open(tmp_output_name.c_str());
  std::ifstream source_input;
  source_input.open(source_file_name.c_str());
  std::ifstream target_input;
  target_input.open(target_file_name.c_str());

  /*
    here for multi-source, not write ...
  */


  // first get the number of lines the files and check to be sure they are the same
  int source_length = 0;
  int target_length = 0;
  std::string source_string;
  std::string target_string;

  source_input.clear();
  target_input.clear();

  source_input.seekg(0, std::ios::beg);
  while (std::getline(source_input, source_string)) {
    ++source_length;
  }

  /*
    here for multi-source, not write ...
  */

  target_input.seekg(0, std::ios::beg);
  while (std::getline(target_input, target_string)) {
    ++target_length;
  }

  visual_num_segment_pairs = target_length;

  // do check to be sure the two files are the same length
  if (source_length != target_length) {
    logger<<"Error: input files are not the same length\n";
    exit(EXIT_FAILURE);
  }

  /*
    here for multi-source, not write ...
  */

  source_input.clear();
  source_input.seekg(0, std::ios::beg);

  target_input.clear();
  target_input.seekg(0, std::ios::beg);

  for (int i = 0; i < source_length; ++i) {
    std::vector<std::string> v_source_sentence;
    std::vector<std::string> v_target_sentence;
    std::getline(source_input, source_string);

    /*
      here for multi-source, not write ...
    */

    std::getline(target_input, target_string);

    std::istringstream iss_source(source_string, std::istringstream::in);
    std::istringstream iss_target(target_string, std::istringstream::in);

    while (iss_source>>word) {
      v_source_sentence.push_back(word);
    }
    while (iss_target>>word) {
      v_target_sentence.push_back(word);
    }

    if (!(v_source_sentence.size() + 1 >= max_sent_cutoff - 2 || v_target_sentence.size() + 1 >= max_sent_cutoff - 2)) {
      data_.push_back(CombineSentenceInformation(v_source_sentence, v_target_sentence));

      visual_num_source_word_tokens += v_source_sentence.size();
      visual_num_target_word_tokens += v_target_sentence.size();

      if (visual_source_longest_sent < v_source_sentence.size()) {
        visual_source_longest_sent = v_source_sentence.size();
      }

      if (visual_target_longest_sent < v_target_sentence.size()) {
        visual_target_longest_sent = v_target_sentence.size();
      }
    } else {
      visual_num_tokens_thrown_away += v_source_sentence.size() + v_target_sentence.size();
    }
  }

  if (minibatch_size != 1) {
    if (shuffle_data_mode__) {
      std::random_shuffle(data_.begin(), data_.end());
    }
  }

  // sort the data based on minibatch
  if (minibatch_size != 1) {
    CompareNeuralMT compare_neural_mt;
    int curr_index = 0;
    while (curr_index < data_.size()) {
      if (curr_index + minibatch_size * minibatch_mult_ <= data_.size()) {
        std::sort(data_.begin() + curr_index, data_.begin() + curr_index + minibatch_size * minibatch_mult_, compare_neural_mt);
        curr_index += minibatch_size * minibatch_mult_;
      } else {
        std::sort(data_.begin() + curr_index, data_.end(), compare_neural_mt);
        break;
      }
    }
  }

  if (data_.size() % minibatch_size != 0) {
    int num_to_remove = data_.size() % minibatch_size;
    for (int i = 0; i < num_to_remove; ++i) {
      data_.pop_back();
    }
  }


  if (0 == data_.size()) {
    logger<<"Error: file size is zero, could be wrong input file or all lines are above max sentence length\n";
    exit(EXIT_FAILURE);
  }

  // now integerize
  for (int i = 0; i < data_.size(); ++i) {
    std::vector<int> v_source_int;
    std::vector<int> v_target_int;

    // for the source side
    for (int j = 0; j < data_[i].v_source_sentence_.size(); ++j) {
      if (0 == uno_source_mapping_.count(data_[i].v_source_sentence_[j])) {
        v_source_int.push_back(uno_source_mapping_["<UNK>"]);
      }
      else {
        v_source_int.push_back(uno_source_mapping_[data_[i].v_source_sentence_[j]]);
      }
    }

    std::reverse(v_source_int.begin(), v_source_int.end());
    data_[i].v_source_sentence_.clear();
    data_[i].v_source_sentence_int_ = v_source_int;
    //data_[i].v_source_sentence_int_.insert(data_[i].v_source_sentence_int_.begin(), 0);

    while (data_[i].v_minus_two_source_.size() != data_[i].v_source_sentence_int_.size()) {
      data_[i].v_minus_two_source_.push_back(-2);
    }

    int max_iter = 0;
    max_iter = data_[i].v_target_sentence_.size();

    // for the target side
    for (int j = 0; j < max_iter; ++j) {
      if (0 == uno_target_mapping_.count(data_[i].v_target_sentence_[j])) {
        if (uno_target_mapping_.count("<UNK>") == 0) {
          v_target_int.push_back(uno_target_mapping_["<UNK>NULL"]);
        } else {
          v_target_int.push_back(uno_target_mapping_["<UNK>"]);
        }
      } else {
        v_target_int.push_back(uno_target_mapping_[data_[i].v_target_sentence_[j]]);
      }
    }
    data_[i].v_target_sentence_.clear();
    data_[i].v_target_sentence_int_i_ = v_target_int;
    data_[i].v_target_sentence_int_o_ = v_target_int;
    data_[i].v_target_sentence_int_i_.insert(data_[i].v_target_sentence_int_i_.begin(), 0);
    data_[i].v_target_sentence_int_o_.push_back(1);
  }

  // now pad
  int current_index = 0;
  while (current_index < data_.size()) {
    int max_source_minibatch = 0;
    int max_target_minibatch = 0;
    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {
      if (data_[i].v_source_sentence_int_.size() > max_source_minibatch) {
        max_source_minibatch = data_[i].v_source_sentence_int_.size();
      }
      if (data_[i].v_target_sentence_int_i_.size() > max_target_minibatch) {
        max_target_minibatch = data_[i].v_target_sentence_int_i_.size();
      }
    }

    for (int i = current_index; i < std::min((int)data_.size(), current_index + minibatch_size); ++i) {
      while (data_[i].v_source_sentence_int_.size() < max_source_minibatch) {
        data_[i].v_source_sentence_int_.insert(data_[i].v_source_sentence_int_.begin(), -1);
        data_[i].v_minus_two_source_.insert(data_[i].v_minus_two_source_.begin(), -1);
      }

      while (data_[i].v_target_sentence_int_i_.size() < max_target_minibatch) {
        data_[i].v_target_sentence_int_i_.push_back(-1);
        data_[i].v_target_sentence_int_o_.push_back(-1);
      }
    }
    current_index += minibatch_size;
  }

  // output validation file
  for (int i = 0; i < data_.size(); ++i) {
    // output v_source_sentence_int
    for (int j = 0; j < data_[i].v_source_sentence_int_.size(); ++j) {
      final_output<<data_[i].v_source_sentence_int_[j];
      if (j != data_[i].v_source_sentence_int_.size()) {
        final_output<<" ";
      }
    }
    final_output<<"\n";

    // output v_minus_two_source
    for (int j = 0; j < data_[i].v_minus_two_source_.size(); ++j) {
      final_output<<data_[i].v_minus_two_source_[j];
      if (j != data_[i].v_minus_two_source_.size()) {
        final_output<<" ";
      }
    }
    final_output<<"\n";

    // output v_target_sentence_int_i
    for (int j = 0; j < data_[i].v_target_sentence_int_i_.size(); ++j) {
      final_output<<data_[i].v_target_sentence_int_i_[j];
      if (j != data_[i].v_target_sentence_int_i_.size()) {
        final_output<<" ";
      }
    }
    final_output<<"\n";

    // output v_target_sentence_int_o
    for (int j = 0; j < data_[i].v_target_sentence_int_o_.size(); ++j) {
      final_output<<data_[i].v_target_sentence_int_o_[j];
      if (j != data_[i].v_target_sentence_int_o_.size()) {
        final_output<<" ";
      }
    }
    final_output<<"\n";
  }

  weights_file.close();
  final_output.close();
  source_input.close();
  target_input.close();

  // print source file stats
  logger<<"\n$$ Source & target development file\n"
        <<"   Number of src/tgt tokens : "<<visual_num_source_word_tokens<<" / "<<visual_num_target_word_tokens<<"\n"
        <<"   Longest src/tgt length   : "<<visual_source_longest_sent<<" / "<<visual_target_longest_sent<<"\n"
        <<"   Src&tgt sentences        : "<<visual_num_segment_pairs<<"\n"
        <<"   Src&tgt tokens thrown out: "<<visual_num_tokens_thrown_away<<"\n";
  return true;
}


bool InputFilePreprocess::IntegerizeFileDecoding(std::string output_weights_name, std::string source_file_name, std::string tmp_output_name, \
                                                 int max_sent_cutoff, int &target_vocab_size, bool multi_source_model, \
                                                 std::string multi_source_mapping_file) {

  data_.clear();

  int visual_num_source_word_tokens = 0;
  int visual_num_segment_pairs = 0;
  int visual_source_longest_sentence = 0;
  int visual_num_tokens_thrown_away = 0;

  std::ifstream weights_file;
  weights_file.open(output_weights_name.c_str());

  // for multi-source only
  std::ifstream multi_source_weight_file;
  if (multi_source_model) {
    multi_source_weight_file.open(multi_source_mapping_file.c_str());
  }

  std::vector<std::string> file_input_vector;
  std::string str;
  std::string word;

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
  std::getline(weights_file, str);
  std::istringstream iss(str, std::istringstream::in);
  while (iss >> word) {
    file_input_vector.push_back(word);
  }

  // set target vocab size
  target_vocab_size = std::stoi(file_input_vector[2]);

  if (!multi_source_model) {
    // now get the source mappings
    std::getline(weights_file, str);     // get this line, since all equals
    while (std::getline(weights_file, str)) {
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
      uno_source_mapping_[word] = tmp_index;
    }
  } else {
    std::getline(multi_source_weight_file, str);
    while (std::getline(multi_source_weight_file, str)) {
      int tmp_index;
      if (str.size() > 3 && str[0] == '=' && str[1] == '=' && str[2] == '=') {
        break;
      }

      std::istringstream iss(str, std::istringstream::in);
      iss >> word;
      tmp_index = std::stoi(word);
      iss >> word;
      uno_source_mapping_[word] = tmp_index;
    }
  }

  // now that we have the mappigs, integerize the file
  std::ofstream final_output;
  final_output.open(tmp_output_name.c_str());
  std::ifstream source_input;
  source_input.open(source_file_name.c_str());

  // first get the number of lines in the files and check to be sure they are the same
  int source_len = 0;
  std::string src_str;

  source_input.clear();

  source_input.seekg(0, std::ios::beg);
  while (std::getline(source_input, src_str)) {
    ++source_len;
  }

  visual_num_segment_pairs = source_len;

  source_input.clear();
  source_input.seekg(0, std::ios::beg);

  for (int i = 0; i < source_len; ++i) {
    std::vector<std::string> v_source_sentence;
    std::vector<std::string> v_target_sentence;
    std::getline(source_input, src_str);

    std::vector<std::string> v_input_tmp;
    basic_method_.SplitWithString(src_str, " |||| ", v_input_tmp);
    src_str = v_input_tmp[0];


    std::istringstream iss_src(src_str, std::istringstream::in);

    while (iss_src >> word) {
      v_source_sentence.push_back(word);
    }

    // reverse source sentence, 1 2 3 4 => 4 3 2 1
    std::reverse(v_source_sentence.begin(), v_source_sentence.end());

    if (!(v_source_sentence.size() + 1 >= max_sent_cutoff - 2)) {
      data_.push_back(CombineSentenceInformation(v_source_sentence, v_target_sentence));
      visual_num_source_word_tokens += v_source_sentence.size();
      if (visual_source_longest_sentence < v_source_sentence.size()) {
          visual_source_longest_sentence = v_source_sentence.size();
      }
    } else {
      visual_num_tokens_thrown_away += v_source_sentence.size() + v_target_sentence.size();
    }
  }

  // now integerize
  for (int i = 0; i < data_.size(); ++i) {
    std::vector<int> v_source_int;
    for (int j = 0; j < data_[i].v_source_sentence_.size(); ++j) {
      if (uno_source_mapping_.count(data_[i].v_source_sentence_[j]) == 0) {
        v_source_int.push_back(uno_source_mapping_["<UNK>"]);
      } else {
        v_source_int.push_back(uno_source_mapping_[data_[i].v_source_sentence_[j]]);
      }
    }
    data_[i].v_source_sentence_.clear();
    data_[i].v_source_sentence_int_ = v_source_int;
  }

  for (int i = 0; i < data_.size(); ++i) {
    for (int j = 0; j < data_[i].v_source_sentence_int_.size(); ++j) {
      final_output<<data_[i].v_source_sentence_int_[j];
      if (j != data_[i].v_source_sentence_int_.size()) {
        final_output<<" ";
      }
    }
    final_output<<"\n";
  }

  weights_file.close();
  final_output.close();
  source_input.close();

  if(!multi_source_model) {
    logger<<"\n$$ Source Decoding File Information\n";
  } else {
    logger<<"\n$$ Source Decoding File Information (another encoder)\n";
  }
  logger<<"   Input file               : "<<source_file_name<<"\n"
        <<"   Output digits file       : "<<tmp_output_name<<"\n"
        <<"   Number of src tokens     : "<<visual_num_source_word_tokens<<"\n"
        <<"   Longest src length       : "<<visual_source_longest_sentence<<"\n"
        <<"   Src sentences            : "<<visual_num_segment_pairs<<"\n"
        <<"   Src tokens thrown out    : "<<visual_num_tokens_thrown_away<<"\n";
  return true;
}



void InputFilePreprocess::UnintFile(std::string output_weights_name, std::string unint_file, std::string output_final_name, bool sequence_to_sequence_mode, bool decoder_mode) {
  
  std::ifstream weights_file;
  weights_file.open(output_weights_name.c_str());
  weights_file.clear();
  weights_file.seekg(0, std::ios::beg);

  std::string str;
  std::string word;

  std::getline(weights_file, str);
  std::getline(weights_file, str);

  if (sequence_to_sequence_mode) {
    while (std::getline(weights_file, str)) {
      if (str.size() > 3 && str[0] == '=' && str[1] == '=' && str[2] == '=') {
        break;
      }
    }
  }

  while (std::getline(weights_file, str)) {
    int tmp_index;

    if (str.size() > 3 && str[0] == '=' && str[1] == '=' && str[2] == '=') {
      break;
    }

    std::istringstream iss(str, std::istringstream::in);
    iss >> word;
    tmp_index = std::stoi(word);
    iss >> word;
    
    uno_target_reverse_mapping_[tmp_index] = word;

  }

  weights_file.close();

  std::ifstream unint;
  unint.open(unint_file.c_str());

  std::ofstream final_output;
  final_output.open(output_final_name.c_str());

  while (std::getline(unint, str)) {
    std::istringstream iss(str, std::istringstream::in);
    std::vector<int> sentence_int;

    if (decoder_mode) {
      if (str[0] == '=' || str[0] == ' ' || str.size() == 0) {
        final_output<<str<<"\n";
        continue;
      }
    }

    bool other_information_flag = false;
    std::string other_information = "";
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


    for (int i = 0; i < sentence_int.size(); ++i) {
      final_output<<uno_target_reverse_mapping_[sentence_int[i]];
      if (i != sentence_int.size() - 1) {
        final_output<<" ";
      }
    }
    final_output<<other_information<<"\n";
  }

  final_output.close();
  unint.close();
}



}