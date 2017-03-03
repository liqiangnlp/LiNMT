/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#include "file_helper.h"

namespace neural_machine_translation {

// CHECK: OK //
// Constructor
FileHelper::FileHelper(std::string file_name, int minibatch_size, int &num_lines_in_file, int longest_sentence, \
                       int source_vocab_size, int target_vocab_size, int &total_words, bool truncated_softmax_mode, \
                       int shortlist_size, int sampled_size, CharCnnConfiguration &char_cnn_config, std::string char_file) {

#ifdef DEBUG_DROPOUT_5
    std::cerr<<"\n************CP5 In *FileHelper* *Constructor*\n"<<std::flush;
    std::cerr<<"   file_name: "<<file_name<<"\n"
             <<"   minibatch_size: "<<minibatch_size<<"\n"
             <<"   num_lines_in_file: "<<num_lines_in_file<<"\n"
             <<"   longest_sentence: "<<longest_sentence<<"\n"
             <<"   source_vocab_size: "<<source_vocab_size<<"\n"
             <<"   target_vocab_size: "<<target_vocab_size<<"\n"
             <<"   total_words: "<<total_words<<"\n"
             <<"   truncated_softmax_mode: "<<truncated_softmax_mode<<"\n"
             <<"   shortlist_size: "<<shortlist_size<<"\n"
             <<"   sampled_size: "<<sampled_size<<"\n"
             <<"   char_file: "<<char_file<<"\n"
             <<"   char_cnn_mode: "<<char_cnn_config.char_cnn_mode_<<"\n"
             <<std::flush;
#endif


  file_name_ = file_name;
  minibatch_size_ = minibatch_size;
  
  // Open the stream to the file
  input_file_.open(file_name_.c_str(), std::ifstream::in);
  source_vocab_size_ = source_vocab_size;
  target_vocab_size_ = target_vocab_size;

  cc_util::GetFileStats(num_lines_in_file, total_words, input_file_, total_target_words_);
  nums_lines_in_file_ = num_lines_in_file;

  // GPU allocation
  max_sentence_length_ = longest_sentence;

  p_host_input_vocab_indices_source_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));
  p_host_output_vocab_indices_source_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));

  p_host_input_vocab_indices_target_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));
  p_host_output_vocab_indices_target_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));

  p_host_input_vocab_indices_source_tmp_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));
  p_host_output_vocab_indices_source_tmp_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));

  p_host_input_vocab_indices_target_tmp_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));
  p_host_output_vocab_indices_target_tmp_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));


  p_host_input_vocab_indices_source_wgrad_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));
  p_host_input_vocab_indices_target_wgrad_ = (int *)malloc(minibatch_size * longest_sentence * sizeof(int));

  if (source_vocab_size != -1) {
    p_bitmap_source_ = new bool[source_vocab_size * sizeof(bool)];
  } else {
    p_bitmap_source_ = new bool[2 * sizeof(bool)];
  }

  p_bitmap_target_ = new bool[target_vocab_size * sizeof(bool)];

  truncated_softmax_mode_ = truncated_softmax_mode;
  sampled_size_ = sampled_size;
  shortlist_size_ = shortlist_size;
  p_host_sampled_indices_ = (int *)malloc(sampled_size * sizeof(int));
  p_host_batch_information_ = (int *)malloc(2 * minibatch_size * sizeof(int));

  // char_cnn is not written
}


// CHECK: OK //
FileHelper::~FileHelper() {
  delete [] p_bitmap_source_;
  delete [] p_bitmap_target_;

  free(p_host_input_vocab_indices_source_);
  free(p_host_output_vocab_indices_source_);

  free(p_host_input_vocab_indices_target_);
  free(p_host_output_vocab_indices_target_);

  free(p_host_input_vocab_indices_source_tmp_);
  free(p_host_output_vocab_indices_source_tmp_);
  free(p_host_input_vocab_indices_target_tmp_);
  free(p_host_output_vocab_indices_target_tmp_);

  free(p_host_input_vocab_indices_source_wgrad_);
  free(p_host_input_vocab_indices_target_wgrad_);

  free(p_host_batch_information_);

  input_file_.close();
}

// CHECK: OK //
// Read in the next minibatch from the file, returns bool, true is same epoch, false if now need to start new epoch
bool FileHelper::ReadMinibatch() {

#ifdef DEBUG_CHECKPOINT_6
  std::cerr<<"\n************CP6 In *FileHelper* *ReadMinibatch*\n"<<std::flush;
  std::cerr<<"   minibatch_size_: "<<minibatch_size_<<"\n"<<std::flush;
#endif

  //int max_sentence_length_source = 0;
  //int max_sentence_length_target = 0;
  bool same_epoch_flag = true;

  // for throughput calculation
  words_in_minibatch_ = 0;

  // for gpu file input
  int current_tmp_source_input_index = 0;
  int current_tmp_source_output_index = 0;
  int current_tmp_target_input_index = 0;
  int current_tmp_target_output_index = 0;

  // char_cnn is not written

  // now load in the minibatch
  for (int i = 0; i < minibatch_size_; ++i) {
    if (current_line_in_file_ > nums_lines_in_file_) {
      input_file_.clear();
      input_file_.seekg(0, std::ios::beg);
      current_line_in_file_ = 1;
      same_epoch_flag = false;

      // char_cnn is not written

      break;
    }

    std::string tmp_input_source;
    std::string tmp_output_source;
    std::getline(input_file_, tmp_input_source);
    std::getline(input_file_, tmp_output_source);

    std::string tmp_input_target;
    std::string tmp_output_target;
    std::getline(input_file_, tmp_input_target);
    std::getline(input_file_, tmp_output_target);

    // process the source
    std::istringstream iss_input_source(tmp_input_source, std::istringstream::in);
    std::istringstream iss_output_source(tmp_output_source, std::istringstream::in);
    std::string word;

    int input_source_length = 0;
    while (iss_input_source>>word) {
      p_host_input_vocab_indices_source_tmp_[current_tmp_source_input_index] = std::stoi(word);
      input_source_length += 1;
      current_tmp_source_input_index += 1;
    }

    int output_source_length = 0;
    while (iss_output_source>>word) {
      p_host_output_vocab_indices_source_tmp_[current_tmp_source_output_index] = std::stoi(word);
      output_source_length += 1;
      current_tmp_source_output_index += 1;
    }

    words_in_minibatch_ += input_source_length;
    //max_sentence_length_source = input_source_length;
    // ABOVE CHECK: OK //


    // process the target
    std::istringstream iss_input_target(tmp_input_target, std::istringstream::in);
    std::istringstream iss_output_target(tmp_output_target, std::istringstream::in);

    int input_target_length = 0;
    while (iss_input_target>>word) {
      p_host_input_vocab_indices_target_tmp_[current_tmp_target_input_index] = std::stoi(word);
      current_tmp_target_input_index += 1;
      input_target_length += 1;
    }

    int output_target_length = 0;
    while (iss_output_target>>word) {
      p_host_output_vocab_indices_target_tmp_[current_tmp_target_output_index] = std::stoi(word);
      current_tmp_target_output_index += 1;
      output_target_length += 1;
    }

    current_source_length_ = input_source_length;
    current_target_length_ = input_target_length;

    words_in_minibatch_ += input_target_length;
    //max_sentence_length_target = input_target_length;

    current_line_in_file_ += 4;
  }
  // ABOVE CHECK: OK //

  if (current_line_in_file_ > nums_lines_in_file_) {
    current_line_in_file_ = 1;
    input_file_.clear();
    input_file_.seekg(0, std::ios::beg);
    same_epoch_flag = false;

    // char_cnn is not written

  }

  // reset for GPU
  words_in_minibatch_ = 0;

  // Get vocab indices in correct memory layout on the host
  // for the source side
  for (int i = 0; i < minibatch_size_; ++i) {
    int stats_source_length = 0;
    for (int j = 0; j < current_source_length_; ++j) {
      
      // stuff for getting the individual source lengths in the minibatch
      if (p_host_input_vocab_indices_source_tmp_[j + current_source_length_ * i] != -1) {
        stats_source_length += 1;
      }

      p_host_input_vocab_indices_source_[i + j * minibatch_size_] = p_host_input_vocab_indices_source_tmp_[j + current_source_length_ * i];
      p_host_output_vocab_indices_source_[i + j * minibatch_size_] = p_host_output_vocab_indices_source_tmp_[j + current_source_length_ * i];

      if (p_host_input_vocab_indices_source_[i + j * minibatch_size_] != -1) {
        words_in_minibatch_ += 1;
      }
    }

    p_host_batch_information_[i] = stats_source_length;
    p_host_batch_information_[i + minibatch_size_] = current_source_length_ - stats_source_length;
  }
  // ABOVE CHECK: OK //


  // for the target side
  for (int i = 0; i < minibatch_size_; ++i) {
    for (int j = 0; j < current_target_length_; ++j) {
      p_host_input_vocab_indices_target_[i + j * minibatch_size_] = p_host_input_vocab_indices_target_tmp_[j + current_target_length_ * i];
      p_host_output_vocab_indices_target_[i + j * minibatch_size_] = p_host_output_vocab_indices_target_tmp_[j + current_target_length_ * i];
      if (p_host_output_vocab_indices_target_[i + j * minibatch_size_] != -1) {
        words_in_minibatch_ += 1;
      }
    }
  }
  // ABOVE CHECK: OK //

  // Now preprocess the data on the host before sending it to the gpu
  PreprocessInputWgrad();

#ifdef DEBUG_DROPOUT
  std::cerr << "   truncated_softmax_mode_: " << truncated_softmax_mode_ << "\n" << std::flush;
#endif

  if(truncated_softmax_mode_) {
    PreprocessOutputTruncatedSoftmax();
  }
    
  return same_epoch_flag;
}

// CHECK: OK //
// This returns the length of the special sequence for the w grad
void FileHelper::PreprocessInputWgrad() {

  // zero out bitmaps at begining
  ZeroBitmaps();

  // for the source side
  for (int i = 0; i < minibatch_size_ * current_source_length_; ++i) {

    if (-1 == p_host_input_vocab_indices_source_[i]) {
      p_host_input_vocab_indices_source_wgrad_[i] = -1;
    } else if (false == p_bitmap_source_[p_host_input_vocab_indices_source_[i]]) {
      p_bitmap_source_[p_host_input_vocab_indices_source_[i]] = true;
      p_host_input_vocab_indices_source_wgrad_[i] = p_host_input_vocab_indices_source_[i];
    } else {
      p_host_input_vocab_indices_source_wgrad_[i] = -1;
    }
  }

  // for the target side
  for (int i = 0; i < minibatch_size_ * current_target_length_; ++i) {
    if (p_host_input_vocab_indices_target_[i] >= target_vocab_size_) {
      std::cerr<<"Error bigger than max target size\n";
      std::cerr<<"Traget sentence length: "<<current_target_length_<<"\n";
      std::cerr<<p_host_input_vocab_indices_target_[i]<<" "<<target_vocab_size_<<"\n";
      exit(EXIT_FAILURE);
    } else if (p_host_input_vocab_indices_target_[i] < -1) {
      std::cerr<<"Error bigger than max target size\n";
      std::cerr<<"Traget sentence length: "<<current_target_length_<<"\n";
      std::cerr<<p_host_input_vocab_indices_target_[i]<<" "<<target_vocab_size_<<"\n";
      exit(EXIT_FAILURE);
    }

    if (-1 == p_host_input_vocab_indices_target_[i]) {
      p_host_input_vocab_indices_target_wgrad_[i] = -1;
    } else if (false == p_bitmap_target_[p_host_input_vocab_indices_target_[i]]) {
      p_bitmap_target_[p_host_input_vocab_indices_target_[i]] = true;
      p_host_input_vocab_indices_target_wgrad_[i] = p_host_input_vocab_indices_target_[i];
    } else {
      p_host_input_vocab_indices_target_wgrad_[i] = -1;
    }
  }
  // ABOVE CHECK: OK //


  // for the source side
  // now go and put all -1's at far right and number in far left
  length_source_wgrad_ = -1;
  int left_index = 0;
  int right_index = minibatch_size_ * current_source_length_ - 1;
  while (left_index < right_index) {

    if (-1 == p_host_input_vocab_indices_source_wgrad_[left_index]) {
      if (-1 != p_host_input_vocab_indices_source_wgrad_[right_index]) {

        int tmp_swap = p_host_input_vocab_indices_source_wgrad_[left_index];
        p_host_input_vocab_indices_source_wgrad_[left_index] = p_host_input_vocab_indices_source_wgrad_[right_index];
        p_host_input_vocab_indices_source_wgrad_[right_index] = tmp_swap;
        left_index++;
        right_index--;
        continue;

      } else {
        right_index--;
        continue;
      }
    } 
    left_index++;
  }

  if (-1 != p_host_input_vocab_indices_source_wgrad_[left_index]) {
    left_index++;
  }
  length_source_wgrad_ = left_index;
  // ABOVE CHECK: OK //


  // for the target side
  // Now go and put all -1's at far right and number in far left
  length_target_wgrad_ = -1;
  left_index = 0;
  right_index = minibatch_size_ * current_target_length_ - 1;

  while (left_index < right_index) {
    if (-1 == p_host_input_vocab_indices_target_wgrad_[left_index]) {
      if (-1 != p_host_input_vocab_indices_target_wgrad_[right_index]) {

        int tmp_swap = p_host_input_vocab_indices_target_wgrad_[left_index];
        p_host_input_vocab_indices_target_wgrad_[left_index] = p_host_input_vocab_indices_target_wgrad_[right_index];
        p_host_input_vocab_indices_target_wgrad_[right_index] = tmp_swap;
        left_index++;
        right_index--;
        continue;

      } else {
        right_index--;
        continue;
      }
    }
    left_index++;
  }

  if (-1 != p_host_input_vocab_indices_target_wgrad_[left_index]) {
    left_index++;
  }
  length_target_wgrad_ = left_index;
}


// CHECK: OK //
// can change to memset for speed if needed
void FileHelper::ZeroBitmaps() {
  // for the p_bitmap_source_
  for (int i = 0; i < source_vocab_size_; ++i) {
    p_bitmap_source_[i] = false;  
  }

  // for the p_bitmap_target_
  for (int i = 0; i < target_vocab_size_; ++i) {
    p_bitmap_target_[i] = false;
  }
}


// CHECK: OK //
void FileHelper::PreprocessOutputTruncatedSoftmax() {
  ZeroBitmaps();
  resevoir_mapping_.clear();

  int current_index = 0;
  for (int i = 0; i < minibatch_size_ * current_target_length_; ++i) {

    if (false == p_bitmap_target_[p_host_output_vocab_indices_target_[i]] && p_host_output_vocab_indices_target_[i] >= shortlist_size_) {
      p_bitmap_target_[p_host_output_vocab_indices_target_[i]] = true;
      p_host_sampled_indices_[current_index] = p_host_output_vocab_indices_target_[i];
      current_index += 1;
    }
  }

  len_unique_words_trunc_softmax_ = current_index;

  if (current_index > sampled_size_) {
    std::cerr<<"Error: the sample size of the truncated softmax is too small\n";
    std::cerr<<"More unique words in the minibatch that there are sample slots\n";
    exit(EXIT_FAILURE);
  }

  current_index = 0;
  int num_to_sample = sampled_size_ - len_unique_words_trunc_softmax_;
  boost::uniform_real<> distribution(0, 1);

  for (int i = shortlist_size_; i < target_vocab_size_; ++i) {

    if (false == p_bitmap_target_[i]) {
      // fill the resevoir
      if (current_index < num_to_sample) {
        p_host_sampled_indices_[len_unique_words_trunc_softmax_ + current_index] = i;
        current_index++;
      } else {
        int rand_num = (int)(current_index * distribution(generator__));

        if (rand_num < num_to_sample) {
          p_host_sampled_indices_[len_unique_words_trunc_softmax_ + rand_num] = i;
        }
        current_index++;
      }
    }
    if (len_unique_words_trunc_softmax_ + current_index >= sampled_size_) {
      break;
    }
  }

  // get the mappings
  for (int i = 0; i < sampled_size_; ++i) {
    resevoir_mapping_[p_host_sampled_indices_[i]] = i;
  }

  for (int i = 0; i < minibatch_size_ * current_target_length_; ++i) {
    if (p_host_output_vocab_indices_target_[i] >= shortlist_size_ && p_host_output_vocab_indices_target_[i] != -1) {
      p_host_output_vocab_indices_target_[i] = resevoir_mapping_.at(p_host_output_vocab_indices_target_[i]);
    }
  }
}



}





