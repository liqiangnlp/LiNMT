/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#include "global_configuration.h"
#include "debug.h"

namespace neural_machine_translation {

void GlobalConfiguration::ParseCommandLine(int argc, char **argv) {

  // files for keeping the user input, 
  // s-to-s (1st source, 2nd target, 3rd output weights name), 
  // s (1st target, 2nd output weights name)
  std::vector<std::string> training_files;  

  // files for force decoding, 
  // s-to-s (1st source, 2nd target, 3rd neural network model, 4th output), 
  // s (1st target, 2nd neural network model, 3rd output)
  std::vector<std::string> test_files;      

  // stuff for adaptive learning rate schedule
  // if s-to-s, 1st is source dev, 2nd is target dev
  // if s, 1st is target dev
  std::vector<std::string> adaptive_learning_rate;

  // lower and upper range for parameter initialization
  std::vector<precision> lower_upper_range;

  // for the 'decoding' flag, 3 arguments must be entered for 'decoding' at least, 
  // 1 kbest, 2 NMT model, 3 output
  std::vector<std::string> kbest_files;

  // for the decoding-sentence, 3 arguments must be entered for decoding-sentence at least,
  // 1 kbest, 2 NMT model, 3 output
  std::vector<std::string> v_decoding_sentences;

  // for the --postprocess-unk, 4 arguments must be entered for --postprocess-unk
  // 1 dict, 2 1best 3 1best-unk 4 outoov-mode
  std::vector<std::string> v_postprocess_unk;

  // for stoic gen, 1st neural netowrk model, 2nd output
  std::vector<std::string> stochastic_gen_files;

  // truncated softmax
  std::vector<std::string> truncated_information;

  // for decoding ratios
  std::vector<precision> decoding_ratio;

  // for continuing to train
  std::vector<std::string> continue_train;

  // for multi gpu training
  std::vector<int> gpu_indices;

  std::vector<precision> clip_cell_values;

  std::vector<double> nce_values;

  // for multisource
  std::vector<std::string> multi_source;

  // for char-mt
  std::vector<int> char_mt_vector;

  // for bleu score
  // 1 1best, 2 dev-format file, 3 number of references, 4 output
  std::vector<std::string> v_bleu_score;

  std::vector<std::string> v_average_models;

  std::vector<std::string> v_vocabulary_replacement;

  std::vector<std::string> v_words_embeddings;

  std::vector<std::string> v_parameters_of_bpe;

  
  namespace p_options = boost::program_options;
  p_options::options_description description("Options");
  AddOptions(description, training_files, continue_train, test_files, \
             kbest_files, v_decoding_sentences, v_postprocess_unk, decoding_ratio, gpu_indices, lower_upper_range, \
             adaptive_learning_rate, clip_cell_values, v_bleu_score, v_average_models, v_vocabulary_replacement, \
             v_words_embeddings, v_parameters_of_bpe);

  boost::program_options::variables_map v_map;

  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, description), v_map);
    boost::program_options::notify(v_map);
    
    NormalSettingsAndChecking(description, v_map, lower_upper_range, clip_cell_values);

    // training or continue training
    if (v_map.count("tune") || v_map.count("continue-tune")) {
      TuningAndContinueTuning(v_map, training_files, continue_train, gpu_indices, lower_upper_range, \
                              adaptive_learning_rate, multi_source, truncated_information);
      return;
    } else {
      TuningErrorSettings(v_map);
    }

    if (v_map.count("decoding")) {
      Decoding(v_map, kbest_files, decoding_ratio, gpu_indices);
      return;
    }

    if (v_map.count("decoding-sentence")) {
      DecodingSentence(v_decoding_sentences);
      return;
    }

    if (v_map.count("force-decoding")) {
      ForceDecoding(v_map, test_files, gpu_indices);
      return;
    }

    if (v_map.count("postprocess-unk")) {
      PostProcessUnk(v_map, v_postprocess_unk);
    }

    if (v_map.count("bleu")) {
      CalculateBleuScore(v_map, v_bleu_score);
    }

    if (v_map.count("average-models")) {
      AverageModels(v_map, v_average_models);
    }

    if (v_map.count("vocab-replacement")) {
      ReplaceVocab(v_map, v_vocabulary_replacement);
    }

    if (v_map.count("word-embedding")) {
      DumpWordEmbedding(v_map, v_words_embeddings);
    }

    if (v_map.count("bpe-train")) {
      TrainBytePairEncoding(v_map, v_parameters_of_bpe);
    }

    if (v_map.count("bpe-segment")) {
      SegmentBytePairEncoding(v_map, v_parameters_of_bpe);
    }

  } catch(boost::program_options::error& e) {
      logger<<"Error: "<<e.what()<<"\n";
      exit(EXIT_FAILURE);
  }
}


void GlobalConfiguration::NormalSettingsAndChecking(boost::program_options::options_description &description, \
                                                    boost::program_options::variables_map &v_map, \
                                                    std::vector<precision> &lower_upper_range, std::vector<precision> &clip_cell_values) {

  // see if the user specified the help flag
  if (v_map.count("help")) {
    std::cerr<<description<<"\n";
    exit(EXIT_FAILURE);
  }

  if (!(v_map.count("tune") || v_map.count("continue-tune") || \
        v_map.count("decoding") || v_map.count("decoding-sentence") || \
        v_map.count("force-decoding") || v_map.count("postprocess-unk") || \
        v_map.count("bleu") || v_map.count("average-models") || \
        v_map.count("vocab-replacement") || v_map.count("word-embedding") || \
        v_map.count("stoch-gen") || v_map.count("bpe-train") || \
        v_map.count("bpe-segment"))) {
    logger<<"Please use \n"
          <<"   $./NiuTrans.NMT --help\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("log")) {
    output_log_mode_ = true;
  }
  logger.InitOutputLogger(output_log_file_name_, output_log_mode_);
  logger<<std::fixed<<std::setprecision(3);

  if (!v_map.count("help")) {
    OptionsSetByUser(v_map);
  }

  if (v_map.count("average-models") || v_map.count("postprocess-unk") || \
	  v_map.count("bleu") || v_map.count("vocab-replacement") || \
      v_map.count("word-embedding") || v_map.count("bpe-train") || \
      v_map.count("bpe-segment") || v_map.count("decoding-sentence")) {
    return;
  }

  if (v_map.count("random-seed")) {
    random_seed_mode_ = true;
  }

  if (v_map.count("tmp-dir-location")) {
    if (tmp_location_ != "") {
      if (tmp_location_[tmp_location_.size() - 1] != '/') {
        tmp_location_ += '/';
      }
    }
  }

  if (v_map.count("shuffle")) {
    shuffle_data_mode__ = shuffle_mode_;
  }

    
  // error checks to be sure only once of these options is set
  if (v_map.count("tune") && v_map.count("decoding")) {
    logger<<"Error: you cannot --tune and --decoding at the same time\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("tune") && v_map.count("decoding-sentence")) {
    logger<<"Error: you cannot --tune and --decoding-sentence at the same time\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("tune") && v_map.count("force-decoding")) {
    logger<<"Error: you cannot --tune and --force-decoding at the same time\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("force-decoding") && v_map.count("decoding")) {
    logger<<"Error: you cannot --force-decoding and --decoding at the same time\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("force-decoding") && v_map.count("decoding-sentence")) {
    logger<<"Error: you cannot --force-decoding and --decoding-sentence at the same time\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("decoding") && v_map.count("decoding-sentence")) {
    logger<<"Error: you cannot --decoding and --decoding-sentence at the same time\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("parameter-range")) {
    lower__ = lower_upper_range[0];
    upper__ = lower_upper_range[1];
  }

  if (v_map.count("continue-tune")) {
    continue_train_mode__ = true;
  } else {
    continue_train_mode__ = false;
  }

  if (v_map.count("clip-cell")) {
    if (clip_cell_values.size() != 2) {
      logger<<"Error: clip-cell must have exactly two arguement\n";
      exit(EXIT_FAILURE);
    }
    cell_clip_mode__ = true;
    cell_clip_threshold__ = clip_cell_values[0];
    error_clip_threshold__ = clip_cell_values[1];
  }

  // This is because it is really 4 less
  longest_sentence_ += 4;

  return;
}



void GlobalConfiguration::TuningAndContinueTuning(boost::program_options::variables_map &v_map, std::vector<std::string> &training_files, \
                                                  std::vector<std::string> &continue_train, std::vector<int> &gpu_indices, \
                                                  std::vector<precision> &lower_upper_range, std::vector<std::string> &adaptive_learning_rate, \
                                                  std::vector<std::string> &multi_source, std::vector<std::string> &truncated_information) {

  FinetuneSettings();
    
    
  if (v_map.count("multi-source")) {
    if (multi_source.size() != 2) {
      logger<<"Error: only two arguements for the multi-source flag\n";
      exit(EXIT_FAILURE);
    }

    multi_source_params_.multi_source_mode_ = true;
    multi_source_params_.file_name_ = multi_source[0];
    multi_source_params_.source_model_name_ = multi_source[1];
  }

  BasicErrorChecking(v_map);


  ClipGradientsSettings(v_map);


  if (v_map.count("nce")) {
    nce_mode_ = true;
    softmax_mode_ = false;
    //print_partition_function_mode__ = true;
  }

  if (v_map.count("replace-unk")) {
    unk_replace_mode_ = true;
  }

  // boost filesystem headers
  boost::filesystem::path unique_path = boost::filesystem::unique_path();

  if (v_map.count("tmp-dir-location")) {
    unique_path = boost::filesystem::path(tmp_location_ + unique_path.string());
  }

  logger<<"\n$$ Directory Information\n";
  logger<<"   Tmp directory            : "<<unique_path.string()<<"\n";
  boost::filesystem::create_directories(unique_path);
  unique_dir_ = unique_path.string();

  train_file_name_ = unique_dir_ + "/train.txt";
  logger<<"   Training file name       : "<<train_file_name_<<"\n";


  // number of layers, error checking is done when initializing model
  if (v_map.count("multi-gpu")) {
    gpu_indices_ = gpu_indices;
  }

  if (v_map.count("continue-tune")) {
    ContinueTuning(v_map, continue_train);
  } else {
    Tuning(v_map, training_files);
  }

  ParameterRangeSettings(v_map, lower_upper_range);

  LearningMethodSettings(v_map, adaptive_learning_rate);

  TruncatedSoftmaxSettings(v_map, truncated_information);

  // put in the first line of the model file with the correct information
  // format:
  //    0: num_layers
  //    1: lstm_size
  //    2: target_vocab_size
  //    3: source_vocab_size
  //    4: attention_model
  //    5: feed_input
  //    6: multi_source
  //    7: combine_lstm
  //    8: char_cnn

  cc_util::AddModelInformation(layers_number_, lstm_size_, target_vocab_size_, source_vocab_size_, \
                               attention_configuration_.attention_model_mode_, attention_configuration_.feed_input_mode_, \
                               multi_source_params_.multi_source_mode_, multi_source_params_.lstm_combine_mode_, \
                               false, output_weight_file_);

  training_mode_ = true;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  return;
}


void GlobalConfiguration::ContinueTuning(boost::program_options::variables_map &v_map, std::vector<std::string> &continue_train) {

  if (v_map.count("nlm")) {
    // continue-tune nlm
    if (continue_train.size() != 2) {
      logger<<"Error: two arguements to be supplied to the --continue-tune\n"
            <<"  1 train file name,  2 neural-lm file\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    attention_configuration_.attention_model_mode_ = false;
    target_file_name_ = continue_train[0];
    input_weight_file_ = continue_train[1];
    output_weight_file_ = continue_train[1];
    sequence_to_sequence_mode_ = false;
    load_model_train_mode_ = true;
    load_model_name_ = input_weight_file_;

    InputFilePreprocess input_file_preprocess;
    input_file_preprocess.IntegerizeFileNeuralLM(input_weight_file_, target_file_name_, train_file_name_, longest_sentence_, minibatch_size_, \
                                                 true, lstm_size_, target_vocab_size_, layers_number_);

  } else {
    // continue-tune nmt
    if (continue_train.size() != 3) {
      logger<<"Error: three arguements to be supplied to the continue-tune flag\n"
            <<"  1. source train file  2. target train file  3. neural-mt model\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);     
    }

    sequence_to_sequence_mode_ = true;
    source_file_name_ = continue_train[0];
    target_file_name_ = continue_train[1];
    input_weight_file_ = continue_train[2];
    output_weight_file_ = continue_train[2];
    load_model_train_mode_ = true;
    load_model_name_ = input_weight_file_;
    logger<<"   Load model name          : "<<load_model_name_<<"\n";

    if (source_file_name_ == target_file_name_) {
      logger<<"Error: do not use the same file for source and target data\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    InputFilePreprocess input_file_preprocess;

    // multi-source is not written
    input_file_preprocess.IntegerizeFileNeuralMT(input_weight_file_, source_file_name_, target_file_name_, train_file_name_, \
                                                 longest_sentence_, minibatch_size_, lstm_size_, source_vocab_size_, target_vocab_size_, \
                                                 layers_number_, attention_configuration_.attention_model_mode_, \
                                                 multi_source_params_.multi_source_mode_, multi_source_params_.file_name_, \
                                                 multi_source_params_.int_file_name_, multi_source_params_.source_model_name_);
          
  }

  std::ifstream tmp_if_stream(input_weight_file_.c_str());
  std::string tmp_str;
  std::string tmp_word;
  std::getline(tmp_if_stream, tmp_str);
  std::istringstream my_ss(tmp_str, std::istringstream::in);
  std::vector<std::string> tmp_model_params;
  while (my_ss >> tmp_word) {
    tmp_model_params.push_back(tmp_word);
  }

  if (tmp_model_params.size() != 9) {
    logger << "Error: the model file is not in the correct format for --force-decoding\n";
    exit(EXIT_FAILURE);
  }

  layers_number_ = std::stoi(tmp_model_params[0]);
  lstm_size_ = std::stoi(tmp_model_params[1]);
  target_vocab_size_ = std::stoi(tmp_model_params[2]);
  source_vocab_size_ = std::stoi(tmp_model_params[3]);
  attention_configuration_.attention_model_mode_ = std::stoi(tmp_model_params[4]);
  attention_configuration_.feed_input_mode_ = std::stoi(tmp_model_params[5]);
  multi_source_params_.multi_source_mode_ = std::stoi(tmp_model_params[6]);
  multi_source_params_.lstm_combine_mode_ = std::stoi(tmp_model_params[7]);
  char_cnn_config_.char_cnn_mode_ = std::stoi(tmp_model_params[8]);
}


void GlobalConfiguration::Tuning(boost::program_options::variables_map &v_map, std::vector<std::string> &training_files) {

  // training
  if (v_map.count("layers-number")) {
    if (layers_number_ <= 0) {
      logger<<"Error: you must have 1 layer at least for your model\n";
      exit(EXIT_FAILURE);
    }
  }

  if (v_map.count("nlm")) {

    // train the neural-lm model
    if (training_files.size() != 2) {
      logger<<"Error: two arguements to be supplied to the --tune flag\n"
            <<"       1 train data file name, 2 output nlm model\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    attention_configuration_.attention_model_mode_ = false;
    sequence_to_sequence_mode_ = false;
    target_file_name_ = training_files[0];
    output_weight_file_ = training_files[1];

    InputFilePreprocess input_helper;

    if (v_map.count("vocab-mapping-file")) {
      ensemble_train_mode_ = true;
    }

    bool success_flag = true;
    if (!ensemble_train_mode_) {
      success_flag = input_helper.PreprocessFilesTrainNeuralLM(minibatch_size_, longest_sentence_, target_file_name_, \
                                                               train_file_name_, target_vocab_size_, shuffle_mode_, \
                                                               output_weight_file_, lstm_size_, layers_number_);
    } else {
      // ensemble training
    }

    // clean up if error
    if (!success_flag) {
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

  } else {

    // train the neural-mt model
    if (training_files.size() != 3) {
      logger<<(int)training_files.size()<<"\n";
      logger<<"Error: three arguments to be supplied to --tune for the nmt model\n"\
              "       1 source train data, 2 target train data, 3 output nmt model\n";

      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    sequence_to_sequence_mode_ = true;
    source_file_name_ = training_files[0];
    target_file_name_ = training_files[1];
    output_weight_file_ = training_files[2];

    if (source_file_name_ == target_file_name_) {
      logger<<"Error: do not use the same file for source and target data\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    // if ensemble training
    if (v_map.count("vocab-mapping-file")) {
      ensemble_train_mode_ = true;
    }

    InputFilePreprocess input_helper;

    bool success_flag = true;

    // char_cnn is not written

    if (multi_source_params_.multi_source_mode_) {
      // multi-source
      ;
    } else if (!ensemble_train_mode_) {
      // normal training
      success_flag = input_helper.PreprocessFilesTrainNeuralMT(minibatch_size_, longest_sentence_, source_file_name_, target_file_name_, \
                                                               train_file_name_, source_vocab_size_, target_vocab_size_, shuffle_mode_, \
                                                               output_weight_file_, lstm_size_, layers_number_, unk_replace_mode_, \
                                                               unk_aligned_width_, attention_configuration_.attention_model_mode_);
    } else {
      // ensemble training
      ;
    }

    // clean up if error
    if (!success_flag) {
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
  }
}


void GlobalConfiguration::BasicErrorChecking(boost::program_options::variables_map &v_map) {

  // some basic error checks to parameters
  if (learning_rate_ <= 0) {
    logger<<"Error: you cannot have a learning rate <= 0\n";
    exit(EXIT_FAILURE);
  }

  if (minibatch_size_ <= 0) {
    logger<<"Error: you cannot have a minibatch of size <= 0\n";
    exit(EXIT_FAILURE);
  }

  if (lstm_size_ <= 0) {
    logger<<"Error: you cannot have a hiddenstate of size <=0\n";
    exit(EXIT_FAILURE);
  }

  if (source_vocab_size_ <= 0) {
    if (source_vocab_size_ != -1) {
      logger<<"Error: you cannot have a source vocab size <=0\n";
      exit(EXIT_FAILURE);
    } 
  }

  if (target_vocab_size_ <= 0) {
    if (target_vocab_size_ != -1) {
      logger<<"Error: you cannot have a target vocab size <=0\n";
      exit(EXIT_FAILURE);
    }
  }

  if (norm_clip_ <= 0) {
    logger<<"Error: you cannot have a norm clip <=0\n";
    exit(EXIT_FAILURE);
  }

  if (epochs_number_ <= 0) {
    logger<<"Error: you cannot have a number of epches <=0\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("dropout")) {
    dropout_mode_ = true;

    if (dropout_rate_ < 0 || dropout_rate_ > 1) {
      logger<<"Error: dropout rate must be between 0 and 1\n";
      exit(EXIT_FAILURE);
    }
  }

}


void GlobalConfiguration::FinetuneSettings() {

  if (finetune_mode_) {
    logger<<"\n$$ Finetune mode\n";
    train_source_input_embedding_mode__ = false;
    train_target_input_embedding_mode__ = false;
    train_target_output_embedding_mode__ = false;
  } 

}


void GlobalConfiguration::LearningMethodSettings(boost::program_options::variables_map &v_map, std::vector<std::string> &adaptive_learning_rate) {
  
  if (v_map.count("fixed-halve-lr-full")) {
    stanford_learning_rate_ = true;
  }

  if (v_map.count("fixed-halve-lr")) {
    google_learning_rate_ = true;
    if (epoch_to_start_halving_ <= 0) {
      logger<<"Error: cannot halve learning rate until 1st epoch\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
  }

  if (v_map.count("adaptive-halve-lr")) {

    learning_rate_schedule_ = true;
    if (v_map.count("nlm")) {
      // nlm model
      if (adaptive_learning_rate.size() != 1) {
        logger<<"Error: adaptive-halve-lr takes one argument\n 1. target dev file name\n";
        boost::filesystem::path tmp_path(unique_dir_);
        boost::filesystem::remove_all(tmp_path);
        exit(EXIT_FAILURE);
      }
      dev_target_file_name_ = adaptive_learning_rate[0];
      test_file_name_ = unique_dir_ + "/validation.txt";

      InputFilePreprocess input_helper;
      input_helper.IntegerizeFileNeuralLM(output_weight_file_, dev_target_file_name_, test_file_name_, \
                                          longest_sentence_, minibatch_size_, true, lstm_size_, target_vocab_size_, layers_number_);

    } else {
      // nmt model
      if (adaptive_learning_rate.size() != 2 && !multi_source_params_.multi_source_mode_) {
        logger<<"Error: adaptive-halve-lr takes two arguments\n"
              <<"       1 source dev file name, 2 target dev file name\n";
        boost::filesystem::path tmp_path(unique_dir_);
        boost::filesystem::remove_all(tmp_path);
        exit(EXIT_FAILURE);
      }

      if (adaptive_learning_rate.size() != 3 && multi_source_params_.multi_source_mode_) {
        logger<<"Error: adaptive-halve-lr takes three arguements with multi-source\n"\
                "       1 source dev file name, 2 target dev file name, 3 other source dev file name\n";
        boost::filesystem::path tmp_path(unique_dir_);
        boost::filesystem::remove_all(tmp_path);
        exit(EXIT_FAILURE);
      }

      if (multi_source_params_.multi_source_mode_) {
        multi_source_params_.test_file_name_ = adaptive_learning_rate[2];
      }

      dev_source_file_name_ = adaptive_learning_rate[0];
      dev_target_file_name_ = adaptive_learning_rate[1];
      test_file_name_ = unique_dir_ + "/validation.txt";
      multi_source_params_.int_file_name_test_ = unique_dir_ + multi_source_params_.int_file_name_test_;

      if (dev_source_file_name_ == dev_target_file_name_) {
        logger<<"Error: do not use the same file for source and target data\n";
        boost::filesystem::path tmp_path(unique_dir_);
        boost::filesystem::remove_all(tmp_path);
        exit(EXIT_FAILURE);
      }

      InputFilePreprocess input_helper;
      input_helper.IntegerizeFileNeuralMT(output_weight_file_, dev_source_file_name_, dev_target_file_name_, test_file_name_, \
                                          longest_sentence_, minibatch_size_, lstm_size_, source_vocab_size_, target_vocab_size_, \
                                          layers_number_, attention_configuration_.attention_model_mode_, multi_source_params_.multi_source_mode_, \
                                          multi_source_params_.test_file_name_, multi_source_params_.int_file_name_test_, \
                                          multi_source_params_.source_model_name_);
    }

    if (v_map.count("best-model")) {
      best_model_mode_ = true;
    }
  }
}


void GlobalConfiguration::ParameterRangeSettings(boost::program_options::variables_map &v_map, std::vector<precision> &lower_upper_range) {
  if (v_map.count("parameter-range")) {
    if (lower_upper_range.size() != 2) {
      logger<<"Error: you must have two inputs to parameter-range\n 1. lower bound\n 2. upper bound\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    lower__ = lower_upper_range[0];
    upper__ = lower_upper_range[1];
    if (lower__ >= upper__) {
      logger<<"Error: the lower parameter range cannot be greater than the upper range\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
  }
}


void GlobalConfiguration::TruncatedSoftmaxSettings(boost::program_options::variables_map &v_map, std::vector<std::string> &truncated_information) {
  if (v_map.count("truncated-softmax")) {
    shortlist_size_ = std::stoi(truncated_information[0]);
    sampled_size_ = std::stoi(truncated_information[1]);
    truncated_softmax_mode_ = true;
    if (shortlist_size_ + sampled_size_ > target_vocab_size_) {
      logger<<"Error: you cannot have shortlist size + sampled size >= target vocab size\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
  }
}


void GlobalConfiguration::ClipGradientsSettings(boost::program_options::variables_map &v_map) {

  if (v_map.count("matrix-clip-gradients")) {
    global_grad_clip_mode__ = false;
    clip_gradient_mode_ = true;
    individual_grad_clip_mode__ = false;
  }

  if (v_map.count("whole-clip-gradients")) {
    global_grad_clip_mode__ = true;
    clip_gradient_mode_ = false;
    individual_grad_clip_mode__ = false;
  }

  if (v_map.count("ind-clip-gradients")) {
    global_grad_clip_mode__ = false;
    clip_gradient_mode_ = false;
    individual_grad_clip_mode__ = true;
  } 

}



void GlobalConfiguration::TuningErrorSettings(boost::program_options::variables_map &v_map) {
  if (v_map.count("layers-number")) {
    logger<<"Error: layers-number should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("dropout")) {
    logger<<"Error: dropout should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("learning-rate")) {
    logger<<"Error: learning-rate should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("random-seed")) {
    logger<<"Error: random-seed should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("hidden-size")) {
    logger<<"Error: hidden-size should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("nce")) {
    logger<<"Error: nce should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("attention-mode")) {
    logger<<"Error: attention-mode should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("attention-width")) {
    logger<<"Error: attention-width should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("feed-input")) {
    logger<<"Error: feed-input should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("source-vocab-size")) {
    logger<<"Error: source-vocab-size should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("target-vocab-size")) {
    logger<<"Error: target-vocab-size should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("parameter-range")) {
    logger<<"Error: parameter-range should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("number-epochs")) {
    logger<<"Error: number-epochs should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("matrix-clip-gradients")) {
    logger<<"Error: matrix-clip-gradients should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("whole-clip-gradients")) {
    logger<<"Error: whole-clip-gradients should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("adaptive-halve-lr")) {
    logger<<"Error: adaptive-halve-lr should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("clip-cell")) {
    logger<<"Error: clip-cell should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("adaptive-decrease-factor")) {
    logger<<"Error: adaptive-decrease-factor should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("fixed-halve-lr")) {
    logger<<"Error: fixed-halve-lr should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("fixed-halve-lr-full")) {
    logger<<"Error: fixed-halve-lr-full should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("screen-print-rate")) {
    logger<<"Error: screen-print-rate should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }

  if (v_map.count("best-model")) {
    logger<<"Error: best-model should only be used during --tune or --continue-tune\n";
    exit(EXIT_FAILURE);
  }  
}



void GlobalConfiguration::Decoding(boost::program_options::variables_map &v_map, std::vector<std::string> &kbest_files, \
                                   std::vector<precision> &decoding_ratio, std::vector<int> &gpu_indices) {
  logger<<"\n$$ Decoding mode\n";

  if (print_decoding_information_mode_) {

    unk_replacement_mode__ = true;
    for (int i = 0; i < beam_size_; ++i) {
      viterbi_alignments__.push_back(-1);
    }
    for (int i = 0; i < beam_size_ * longest_sentence_; ++i) {
      alignment_scores__.push_back(0);
    }
    p_host_align_indices__ = (int*)malloc((2 * attention_configuration_.d_ + 1) * beam_size_ * sizeof(int));
    p_host_alignment_values__ = (precision*)malloc((2 * attention_configuration_.d_ + 1) * beam_size_ * sizeof(precision));
  }

  if (kbest_files.size() < 3) {
    logger<<"Error: at least 3 arguements must be entered for --decoding\n"\
            "       1 k-best, 2 NMT model, 3 kbest file\n";
    exit(EXIT_FAILURE);
  }

  if (decode_user_files_additional_.size() == 0) {
    for (int i = 0; i < decode_user_files_.size(); ++i) {
      decode_user_files_additional_.push_back("NULL");    
    }
  }

  if (model_names_multi_source_.size() == 0) {
    for (int i = 0; i < decode_user_files_.size(); ++i) {
      model_names_multi_source_.push_back("NULL");
    }
  }

  boost::filesystem::path unique_path = boost::filesystem::unique_path();
  if (v_map.count("tmp-dir-location")) {
    unique_path = boost::filesystem::path(tmp_location_ + unique_path.string());
  }

  boost::filesystem::create_directories(unique_path);
  unique_dir_ = unique_path.string();
  logger<<"\n$$ Directory Information\n"
        <<"   Tmp directory            : "<<unique_dir_<<"\n";

  // for ensembles, get the rnn models
  for (int i = 1; i < kbest_files.size() - 1; ++i) {
    model_names_.push_back(kbest_files[i]);
    std::string tmp_path = unique_dir_ + "/kbest_tmp_" + std::to_string(i - 1);

    decode_tmp_files_.push_back(tmp_path);
    tmp_path = unique_dir_ + "/kbest_tmp_additional_" + std::to_string(i - 1);

    decode_tmp_files_additional_.push_back(tmp_path);
  }

  if (model_names_.size() != decode_user_files_.size() || model_names_.size() != model_names_multi_source_.size()) {
    logger<<"Error: the same number of inputs must be specified as models\n";
    exit(EXIT_FAILURE);
  }
  
  hypotheses_number_ = std::stoi(kbest_files[0]);
  decoder_output_file_ = kbest_files.back() + "-int";
  decoder_final_file_ = kbest_files.back();

  InputFilePreprocess input_helper;

  for (int i = 0; i < decode_tmp_files_.size(); ++i) {
    input_helper.IntegerizeFileDecoding(model_names_[i], decode_user_files_[i], decode_tmp_files_[i], \
                                        longest_sentence_, target_vocab_size_, false, "NULL");

    if (decode_user_files_additional_[i] != "NULL") {
      input_helper.IntegerizeFileDecoding(model_names_[i], decode_user_files_additional_[i], decode_tmp_files_additional_[i], \
                                          longest_sentence_, target_vocab_size_, true, model_names_multi_source_[i]);
    }
  }

  // each model must be specified a gpu, default all is gpu 0
  if (v_map.count("multi-gpu")) {
    if (gpu_indices.size() != model_names_.size()) {
      logger<<"Error: for decoding, each model must be specified a gpu\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
    gpu_indices_ = gpu_indices;
  }
  else {
    for (int i = 0; i < model_names_.size(); ++i) {
      gpu_indices_.push_back(0);
    }
  }

  if (beam_size_ <= 0) {
    logger<<"Error: beam size cannot be <= 0\n";
    boost::filesystem::path tmp_path(unique_dir_);
    boost::filesystem::remove_all(tmp_path);
    exit(EXIT_FAILURE);
  }

  if (penalty_ < 0) {
    logger<<"Error: penalty cannot be less than zero\n";
    boost::filesystem::path tmp_path(unique_dir_);
    boost::filesystem::remove_all(tmp_path);
    exit(EXIT_FAILURE);
  }

  if (decoding_dump_sentence_embedding_mode_) {
    if ("" == decoding_sentence_embedding_file_name_) {
      logger<<"Error: --sentence-embedding-file cannot be empty\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
  }

  if (v_map.count("dump-lstm")) {
    dump_lstm_mode_ = true;
  }

  if (v_map.count("decoding-ratio")) {
    if (decoding_ratio.size() != 2) {
      logger<<"Error: only two inputs for --decoding-ratio, now is "<<decoding_ratio.size()<<"\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
    min_decoding_ratio_ = decoding_ratio[0];
    max_decoding_ratio_ = decoding_ratio[1];
    if (min_decoding_ratio_ >= max_decoding_ratio_) {
      logger<<"Error: min decoding ratio must be less than max_decoding_ratio\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }
  }

  training_mode_ = false;
  decode_mode_ = true;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  sequence_to_sequence_mode_ = true;
  return;  
}


void GlobalConfiguration::DecodingSentence(std::vector<std::string> &v_decoding_sentences) {

  logger<<"\n$$ Decoding sentence mode\n";

  if (v_decoding_sentences.size() != 3) {
    logger<<"Error: --decoding-sentence takes three arguements.\n"
          <<" <config> <input> <output>\n";
    exit(EXIT_FAILURE);
  }

  decode_sentence_config_file_ = v_decoding_sentences.at(0);
  decode_sentence_input_file_ = v_decoding_sentences.at(1);
  decode_sentence_output_file_ = v_decoding_sentences.at(2);

  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = true;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  sequence_to_sequence_mode_ = true;
  return;
}



void GlobalConfiguration::ForceDecoding(boost::program_options::variables_map &v_map, std::vector<std::string> &test_files, std::vector<int> &gpu_indices) {

  force_decode_mode__ = true;

  if (v_map.count("multi-gpu")) {
    gpu_indices_ = gpu_indices;
  }

  boost::filesystem::path unique_path = boost::filesystem::unique_path();

  if (v_map.count("tmp-dir-location")) {
    unique_path = boost::filesystem::path(tmp_location_ + unique_path.string());
  }

  logger<<"$$ Directory Information\n"
        <<"   Tmp directory: "<<unique_path<<"\n";

  boost::filesystem::create_directories(unique_path);
  unique_dir_ = unique_path.string();

  test_file_name_ = unique_dir_ + "/validation.txt";

  if (v_map.count("nlm")) {

    // Neural LM
    if (test_files.size() != 3) {
      logger<<"Error: --force-decoding takes three arguements.\n"
            <<" 1 input file,  2 neural model,  3 output file\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    attention_configuration_.attention_model_mode_ = false;
    target_file_name_ = test_files[0];
    input_weight_file_ = test_files[1];
    output_force_decode_ = test_files[2];
    sequence_to_sequence_mode_ = false;

    InputFilePreprocess input_file_preprocess;
    input_file_preprocess.IntegerizeFileNeuralLM(input_weight_file_, target_file_name_, test_file_name_, longest_sentence_, \
                                                 minibatch_size_, false, lstm_size_, target_vocab_size_, layers_number_);
  } else {
    // Neural MT
    if (test_files.size() != 4) {
      logger<<"Error: --force-decoding takes four arguements:\n"
            <<"  1 source input file,  2 target input file,  3 neural model,  4 output file\n";

      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit(EXIT_FAILURE);
    }

    sequence_to_sequence_mode_ = true;
    source_file_name_ = test_files[0];
    target_file_name_ = test_files[1];
    input_weight_file_ = test_files[2];
    output_force_decode_ = test_files[3];

    attention_configuration_.tmp_alignment_file_name_ = unique_dir_ + "/alignments.txt";

    if (source_file_name_ == target_file_name_) {
      logger<<"Error: do not use the same file for source and target data\n";
      boost::filesystem::path tmp_path(unique_dir_);
      boost::filesystem::remove_all(tmp_path);
      exit (EXIT_FAILURE);
    }

    // multi-source is not written

    // char_cnn is not written
    InputFilePreprocess input_file_preprocess;
    input_file_preprocess.IntegerizeFileNeuralMT(input_weight_file_, source_file_name_, target_file_name_, test_file_name_, \
                                                 longest_sentence_, 1, lstm_size_, source_vocab_size_, target_vocab_size_, layers_number_, \
                                                 attention_configuration_.attention_model_mode_, multi_source_params_.multi_source_mode_, \
                                                 multi_source_params_.test_file_name_, multi_source_params_.int_file_name_test_, \
                                                 multi_source_params_.source_model_name_);
    minibatch_size_ = 1;
  }

  std::ifstream tmp_if_stream(input_weight_file_.c_str());
  std::string tmp_str;
  std::string tmp_word;
  std::getline(tmp_if_stream, tmp_str);
  std::istringstream my_ss(tmp_str, std::istringstream::in);
  std::vector<std::string> tmp_model_params;
  while (my_ss>>tmp_word) {
    tmp_model_params.push_back(tmp_word);
  }

  if (tmp_model_params.size() != 9) {
    logger<<"Error: the model file is not in the correct format for --force-decoding\n";
    exit(EXIT_FAILURE);
  }

  layers_number_ = std::stoi(tmp_model_params[0]);
  lstm_size_ = std::stoi(tmp_model_params[1]);
  target_vocab_size_ = std::stoi(tmp_model_params[2]);
  source_vocab_size_ = std::stoi(tmp_model_params[3]);
  attention_configuration_.attention_model_mode_ = std::stoi(tmp_model_params[4]);
  attention_configuration_.feed_input_mode_ = std::stoi(tmp_model_params[5]);
  multi_source_params_.multi_source_mode_ = std::stoi(tmp_model_params[6]);
  multi_source_params_.lstm_combine_mode_ = std::stoi(tmp_model_params[7]);
  char_cnn_config_.char_cnn_mode_ = std::stoi(tmp_model_params[8]);

  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = true;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  return;
}


void GlobalConfiguration::PostProcessUnk(boost::program_options::variables_map &v_map, std::vector<std::string> &v_postprocess_unk) {
  logger<<"\n$$ Postprocess unks\n";

  if (v_postprocess_unk.size() != 4) {
    logger<<"Error: --postprocess-unk takes four arguements.\n"
          <<" 1 config, 2 src,  3 1best, 4 output\n";
    exit(EXIT_FAILURE);
  }

  unk_config_file_name_ = v_postprocess_unk.at(0);
  unk_source_file_name_ = v_postprocess_unk.at(1);
  unk_1best_file_name_ = v_postprocess_unk.at(2);
  unk_output_file_name_ = v_postprocess_unk.at(3);

  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = true;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  return;
}


void GlobalConfiguration::CalculateBleuScore(boost::program_options::variables_map &v_map, std::vector<std::string> &v_bleu_score) {

  logger<<"\n$$ Calculate BLEU score\n";

  if (v_bleu_score.size() != 5) {
    logger<<"Error: --bleu takes five arguements.\n"
          <<" 1 1best,  2 dev,  3 nref, 4 rmoov, 5 output\n";
    exit(EXIT_FAILURE);
  }

  bleu_one_best_file_name_ = v_bleu_score[0];
  bleu_dev_file_name_ = v_bleu_score[1];
  bleu_number_references_ = std::stoi(v_bleu_score[2].c_str());
  bleu_remove_oov_mode_ = std::stoi(v_bleu_score[3].c_str());
  bleu_output_file_name_ = v_bleu_score[4];

  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = true;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  return;
}


void GlobalConfiguration::AverageModels(boost::program_options::variables_map &v_map, std::vector<std::string> &v_average_models) {

  logger<<"\n$$ Average models\n";

  if (v_average_models.size() < 3) {
    logger<<"Error: --average-models takes three arguements at least.\n"
          <<" <mode-1> ... <model-n> <output>\n";
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < v_average_models.size() - 1; ++i) {
    v_average_models_input_file_.push_back(v_average_models.at(i));
  }
  average_models_output_file_ = v_average_models.back();

  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = true;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  return;
}

void GlobalConfiguration::ReplaceVocab(boost::program_options::variables_map &v_map, std::vector<std::string> &v_vocabulary_replacement) {

  logger<<"\n$$ Vocabulary Replacement\n";
  if (v_vocabulary_replacement.size() != 3) {
    logger<<"Error: --vocab-replacement takes three arguements.\n"
          <<" <config> <model> <output>\n";
    exit(EXIT_FAILURE);
  }

  vocab_replace_config_file_name_ = v_vocabulary_replacement.at(0);
  vocab_replace_model_file_name_ = v_vocabulary_replacement.at(1);
  vocab_replace_output_file_name_ = v_vocabulary_replacement.at(2);

  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = true;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  return;
}



void GlobalConfiguration::DumpWordEmbedding(boost::program_options::variables_map &v_map, std::vector<std::string> &v_words_embeddings) {
  logger<<"\n$$ Dump Words Embeddings\n";
  if (v_words_embeddings.size() != 3) {
    logger<<"Error: --word-embedding takes three arguements.\n"
          <<" <model> <src-word> <tgt-word>\n";
    exit(EXIT_FAILURE);
  }

  word_embedding_model_file_name_ = v_words_embeddings.at(0);
  word_embedding_source_vocab_file_name_ = v_words_embeddings.at(1);
  word_embedding_target_vocab_file_name_ = v_words_embeddings.at(2);

  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = true;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = false;
  return;
}


void GlobalConfiguration::TrainBytePairEncoding(boost::program_options::variables_map &v_map, std::vector<std::string> &v_parameters_of_bpe) {
  logger<<"\n$$ Train Byte Pair Encoding Model\n";
  if (v_parameters_of_bpe.size() != 4) {
    logger<<"Error: --bpe-train takes four arguements.\n"
          <<" <vocab-size> <min-freq> <input> <output>\n";
    exit(EXIT_FAILURE);
  }

  bpe_vocabulary_size_ = std::stoi(v_parameters_of_bpe.at(0).c_str());
  bpe_min_frequency_ = std::stoi(v_parameters_of_bpe.at(1).c_str());
  bpe_input_file_name_ = v_parameters_of_bpe[2];
  bpe_output_file_name_ = v_parameters_of_bpe[3];
  
  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = true;
  segment_bpe_mode_ = false;
  return;
}



void GlobalConfiguration::SegmentBytePairEncoding(boost::program_options::variables_map &v_map, std::vector<std::string> &v_parameters_of_bpe) {

  logger<<"\n$$ BPE Segment\n";
  if (v_parameters_of_bpe.size() != 3) {
    logger<<"Error: --bpe-segment takes four arguements.\n"
          <<" <codes> <input> <output>\n";
    exit(EXIT_FAILURE);
  }

  bpe_input_codes_file_name_ = v_parameters_of_bpe[0];
  bpe_input_file_name_ = v_parameters_of_bpe[1];
  bpe_output_file_name_ = v_parameters_of_bpe[2];
  
  training_mode_ = false;
  decode_mode_ = false;
  decode_sentence_mode_ = false;
  test_mode_ = false;
  stochastic_generation_mode_ = false;
  postprocess_unk_mode_ = false;
  calculate_bleu_mode_ = false;
  average_models_mode_ = false;
  vocabulary_replacement_mode_ = false;
  dump_word_embedding_mode_ = false;
  train_bpe_mode_ = false;
  segment_bpe_mode_ = true;
  return;
}



void GlobalConfiguration::OptionsSetByUser(boost::program_options::variables_map &v_map) {
  logger<<"\n$$ Options set by the user\n";
    
  // now try to loop over all the boost program options
  for (auto it = v_map.begin(); it != v_map.end(); ++it) {
    logger<<"   "<<it->first<<": ";
    auto& value = it->second.value();
    if (auto v = boost::any_cast<int>(&value)) {
      logger<<*v<<"\n";
    } else if (auto v = boost::any_cast<bool>(&value)) {
      logger<<*v<<"\n";
    } else if (auto v = boost::any_cast<float>(&value)) {
      logger<<*v<<"\n";
    } else if (auto v = boost::any_cast<double>(&value)) {
      logger<<*v<<"\n";
    } else if (auto v = boost::any_cast<std::string>(&value)) {
      logger<<*v<<"\n";
    } else if (std::vector<std::string> *v = boost::any_cast<std::vector<std::string>>(&value)) {
      std::vector<std::string> vv = *v;
      for (int i = 0; i < vv.size(); ++i) {
        logger<<" "<<vv[i]<<" ";
      }
      logger<<"\n";
    } else if (std::vector<precision> *v = boost::any_cast<std::vector<precision>>(&value)) {
      std::vector<precision> vv = *v;
      for (int i = 0; i < vv.size(); ++i) {
        logger<<" "<<vv[i]<<" ";
      }
      logger<<"\n";
    } else if (std::vector<int> *v = boost::any_cast<std::vector<int>>(&value)) {
      std::vector<int> vv = *v;
      for (int i = 0; i < vv.size(); ++i) {
        logger<<" "<<vv[i]<<" ";
      }
      logger<<"\n";
    } else {
      logger<<"Not Printable\n";
    }
  }
}


void GlobalConfiguration::AddOptions(boost::program_options::options_description &description, std::vector<std::string> &training_files, \
                                     std::vector<std::string> &continue_train, std::vector<std::string> &test_files, \
                                     std::vector<std::string> &kbest_files, std::vector<std::string> &v_decoding_sentences, \
                                     std::vector<std::string> &v_postprocess_unk, std::vector<precision> &decoding_ratio, \
                                     std::vector<int> &gpu_indices, std::vector<precision> &lower_upper_range, \
                                     std::vector<std::string> &adaptive_learning_rate, std::vector<precision> &clip_cell_values, \
                                     std::vector<std::string> &v_bleu_score, std::vector<std::string> &v_average_models, \
                                     std::vector<std::string> &v_vocabulary_replacement, std::vector<std::string> &v_words_embeddings, \
                                     std::vector<std::string> &v_parameters_of_bpe) {
  namespace p_options = boost::program_options;
  description.add_options() 
    ("help", "NiuTrans.NMT Usage\n")
    ("tune", p_options::value<std::vector<std::string> > (&training_files)->multitoken(), "Tune a NMT/NLM model\n"\
     " NMT: <srcfile> <tgtfile> <model>\n"\
     " NLM: <tgtfile> <model>")
    ("continue-tune", p_options::value<std::vector<std::string> > (&continue_train)->multitoken(), "Resume tuning\n"\
     " NMT: <srcfile> <tgtfile> <model>\n"\
     " NLM: <tgtfile> <model>")
    ("decoding", p_options::value<std::vector<std::string> >(&kbest_files)->multitoken(), "Decoding sentences in files\n"\
    " <paths> <m1> ... <mn> <output>")
    ("decoding-sentence", p_options::value<std::vector<std::string> >(&v_decoding_sentences)->multitoken(), "Decoding sentence by sentence\n"\
    " <config> <input> <output>")
    ("force-decoding", p_options::value<std::vector<std::string> >(&test_files)->multitoken(), "Force decoding\n"\
    " NMT: <src> <tgt> <model> <output>\n"\
    " NLM: <tgt> <model> <output>")
    ("average-models", p_options::value<std::vector<std::string> >(&v_average_models)->multitoken(), "Average n models\n"\
    " <model 1> ... <model n> <output>")
    ("vocab-replacement", p_options::value<std::vector<std::string> >(&v_vocabulary_replacement)->multitoken(), "Replace vocabulary\n"\
    " <replaced-words> <model> <output>")
    ("postprocess-unk", p_options::value<std::vector<std::string> >(&v_postprocess_unk)->multitoken(), "Postprocess unks\n"\
    " <config> <src> <1best> <output>")
    ("word-embedding", p_options::value<std::vector<std::string> >(&v_words_embeddings)->multitoken(), "Dump words embeddings\n"\
    " <model> <src-words> <tgt-words>")
    ("bleu", p_options::value<std::vector<std::string> >(&v_bleu_score)->multitoken(), "Calculate BLEU score\n"\
    " <1best> <dev> <nref> <rmoov> <output>\n")
    //("stoch-gen,g", p_options::value<std::vector<std::string> >(&stochastic_gen_files)->multitoken(), "Do random generation for a sequence model, such as a language model\n"\
      " FORMAT: <neural network model> <output file name>")
    ("nlm", "Train the NLM model\n DEFAULT: False\n")
    ("finetune", p_options::value<bool>(&finetune_mode_), "Finetune an existing model\n DEFAULT: False")
    ("source-vocab-size", p_options::value<int>(&source_vocab_size_), "Set source vocab size\n DEFAULT: Size of unique words")
    ("target-vocab-size", p_options::value<int>(&target_vocab_size_), "Set target vocab size\n DEFAULT: Size of unique words")
    ("hidden-size", p_options::value<int>(&lstm_size_), "Size of lstm nodes\n DEFAULT: 100")
    ("layers-number", p_options::value<int>(&layers_number_), "Number of layers\n DEFAULT: 1")
    ("shuffle", p_options::value<bool>(&shuffle_mode_), "Shuffle the training data\n DEFAULT: True")
    ("minibatch-size", p_options::value<int>(&minibatch_size_), "Minibatch size\n DEFAULT: 8")
    ("learning-rate", p_options::value<precision>(&learning_rate_), "Learning rate\n DEFAULT: 0.5")
    ("parameter-range", p_options::value<std::vector<precision> >(&lower_upper_range)->multitoken(), "Range of initializated parameters\n"\
    " DEFAULT: -0.08 0.08")
    ("whole-clip-gradients", p_options::value<precision>(&norm_clip_), "Gradient clip threshold\n DEFAULT: 5")
    //("matrix-clip-gradients,c", p_options::value<precision>(&norm_clip_), "Grad clip threshold\n DEFAULT: 5")
    //("ind-clip-gradients,i", p_options::value<precision>(&individual_norm_clip_threshold__), "Set gradient clipping threshold for individual elements\n DEFAULT: 0.1")
    ("multi-gpu", p_options::value<std::vector<int> >(&gpu_indices)->multitoken(), "Tune a model on multi-gpus\n DEFAULT: 0 ... 0")
    ("number-epochs", p_options::value<int>(&epochs_number_), "Number of epochs\n DEFAULT: 10")
    ("attention-mode", p_options::value<bool>(&attention_configuration_.attention_model_mode_), "Use attention or not\n DEFAULT: False")
    ("attention-width", p_options::value<int>(&attention_configuration_.d_), "Attention width\n DEFAULT: 10")
    ("feed-input", p_options::value<bool>(&attention_configuration_.feed_input_mode_), "Use feed input or not\n DEFAULT: False")
    ("dropout", p_options::value<precision>(&dropout_rate_), "The probability of keeping a node\n <dropout rate> DEFAULT: not used")
    ("adaptive-halve-lr", p_options::value<std::vector<std::string> > (&adaptive_learning_rate)->multitoken(), "Adaptive learning\n"\
     " NMT: <source dev> <target dev>\n"\
     " NLM: <target dev>")
    ("adaptive-decrease-factor", p_options::value<precision>(&decrease_factor_), "To be used with adaptive-halve-lr\n DEFAULT: 0.5")
    ("fixed-halve-lr", p_options::value<int>(&epoch_to_start_halving_), "Midway halve the learning rate")
    ("fixed-halve-lr-full", p_options::value<int>(&epoch_to_start_halving_full_), "Full halve the learning rate")
    ("best-model", p_options::value<std::string>(&best_model_file_name_), "Output the best model\n <output file>")
    ("save-all-models", p_options::value<bool>(&dump_every_best__), "Save all best models\n DEFAULT: False")
    ("vocab-mapping-file", p_options::value<std::string>(&ensemble_train_file_name_), "Tune models with the same vocabulary\n"\
     " <model>")    
    ("random-seed", p_options::value<int>(&random_seed_int_), "Specify a random seed")
    ("clip-cell", p_options::value<std::vector<precision>>(&clip_cell_values)->multitoken(), "Specify cell clip & error threshold in bpp\n" \
     " <cell threshold> <error threshold>\n Recommended: <50> <1000>\n DEFAULT: not used")
    ("screen-print-rate", p_options::value<int>(&screen_print_rate_), "How many minibatches to print information\n DEFAULT: 20\n")
    ("decoded-files", p_options::value<std::vector<std::string> >(&decode_user_files_)->multitoken(), "File to be decoded"\
     "\n <file1> ... <filen>")
    ("beam-size",p_options::value<int>(&beam_size_), "Beam size\n DEFAULT: 12")
    ("penalty", p_options::value<precision> (&penalty_), "Penalty\n DEFAULT: 0")
    ("decoding-ratio", p_options::value<std::vector<precision>>(&decoding_ratio)->multitoken(), "Restrict the output length\n DEFAULT: 0.5 1.5")
    ("lp-alpha", p_options::value<precision>(&decoding_lp_alpha_), "Alpha of length normalization\n DEFAULT: 0.65")
    ("cp-beta", p_options::value<precision>(&decoding_cp_beta_), "Beta of penalty\n DEFAULT: 0.20")
    ("diversity", p_options::value<precision>(&decoding_diversity_), "Diversity\n DEFAULT: 0.10")
    ("dump-sentence-embedding", p_options::value<bool>(&decoding_dump_sentence_embedding_mode_), "Dump sentence embedding\n DEFAULT: False")
    ("sentence-embedding-file", p_options::value<std::string>(&decoding_sentence_embedding_file_name_), "Sentence embedding file\n DEFAULT: sentence-embedding.txt")
    ("print-decoding-info", p_options::value<bool>(&print_decoding_information_mode_), "Output decoding information\n DEFAULT: False")
    ("print-align-scores", p_options::value<bool>(&print_alignments_scores_mode_), "Output all alignments scores\n DEFAULT: False\n")
    ("longest-sent", p_options::value<int>(&longest_sentence_), "Maximum sentence length\n DEFAULT: 100")
    ("tmp-dir-location", p_options::value<std::string>(&tmp_location_), "Specify the tmp location\n DEFAULT: ./")
    ("log", p_options::value<std::string>(&output_log_file_name_), "Print out informations\n <file>\n")
    ("bpe-train", p_options::value<std::vector<std::string> >(&v_parameters_of_bpe)->multitoken(), "Train BPE model\n <vocab-size> <min-freq> <input> <output>")
    ("bpe-segment", p_options::value<std::vector<std::string> >(&v_parameters_of_bpe)->multitoken(), "BPE segment\n <codes> <input> <output>");  
}


void GlobalConfiguration::PrintParameters() {
  if (training_mode_) {
    logger<<"\n$$ Training Information\n"
		  <<"   Minibatch size           : "<<minibatch_size_<<"\n"
          <<"   Number of epochs         : "<<epochs_number_<<"\n"
          <<"   Learning rate            : "<<learning_rate_<<"\n";

    if (clip_gradient_mode_) {
      logger<<"   Gradient clipping threshold per matrix: "<<norm_clip_<<"\n";
    }

    if (individual_gradient_clip_mode_) {
      logger<<"   Gradient clipping threshold per element: "<<individual_norm_clip_threshold_<<"\n";
    }

    if (dropout_mode_) {
      logger<<"   Dropout mode             : TRUE\n"
            <<"   Dropout rate             : "<<dropout_rate_<<"\n\n";
    } else {
      logger<<"   Dropout mode             : FALSE\n\n";
    }

    logger<<"\n$$ Model Information\n";
    if (sequence_to_sequence_mode_) {
      logger<<"   Model                    : NMT\n";
    } else {
      logger<<"   Model                    : Neural LM\n";    
    }

    logger<<"   Source vocabulary size   : "<<source_vocab_size_<<"\n"
          <<"   Target vocabulary size   : "<<target_vocab_size_<<"\n"
          <<"   LSTM size                : "<<lstm_size_<<"\n"
          <<"   Number of layers         : "<<layers_number_<<"\n";

    if (attention_configuration_.attention_model_mode_) {
      logger<<"   Attention mode           : TRUE\n"
            <<"   Attention D              : "<<attention_configuration_.d_<<"\n";
      if (attention_configuration_.feed_input_mode_) {
        logger<<"   Feed input mode          : TRUE\n";
      } else {
        logger<<"   Feed input mode          : FALSE\n";
      }
    } else {
      logger<<"   Attention mode           : FALSE\n";
    }

    if (unk_replace_mode_) {
      logger<<"   <UNK> replacement        : TRUE\n";
    } else {
      logger<<"   <UNK> replacement        : FALSE\n";
    }

    if (nce_mode_) {
      logger<<"   NCE objective            : TRUE\n";
      logger<<"   Number of noise samples  : "<<negative_samples_number_<<"\n";
    } else {
      logger<<"   MLE objective            : TRUE\n";
    }
  }


  if (stochastic_generation_mode_) {
    logger<<"\n$$ Stochastic Generation Information\n"
          <<"   Number of tokens         : "<<stochastic_generation_length_<<"\n"
          <<"   Temperature              : "<<stochastic_generation_temperature_<<"\n";
  }
}


void GlobalConfiguration::PrintDecodingParameters() {
  logger<<"\n$$ Decoding Information\n"
        <<"   Kbest number             : "<<hypotheses_number_<<"\n"
        <<"   Beam size                : "<<beam_size_<<"\n"
        <<"   Target vocab size        : "<<target_vocab_size_<<"\n"
        <<"   Penalty                  : "<<penalty_<<"\n"
        <<"   Decoding ratio           : "<<min_decoding_ratio_<<" "<<max_decoding_ratio_<<"\n"
        <<"   Length normalization     : "<<decoding_lp_alpha_<<"\n"
        <<"   Penalty beta             : "<<decoding_cp_beta_<<"\n"
        <<"   Longest sentence         : "<<longest_sentence_<<"\n"
        <<"   Print information        : "<<print_decoding_information_mode_<<"\n"
        <<"   Print alignment scores   : "<<print_alignments_scores_mode_<<"\n"
        <<"   Integerized results      : "<<decoder_output_file_<<"\n"
        <<"   Final results            : "<<decoder_final_file_<<"\n";
  return;
}


void GlobalConfiguration::PrintDecodingSentParameters() {
  logger<<"\n$$ Decoding Information\n"
        <<"   Kbest number             : "<<hypotheses_number_<<"\n"
        <<"   Beam size                : "<<beam_size_<<"\n"
        <<"   Target vocab size        : "<<target_vocab_size_<<"\n"
        <<"   Penalty                  : "<<penalty_<<"\n"
        <<"   Decoding ratio           : "<<min_decoding_ratio_<<" "<<max_decoding_ratio_<<"\n"
        <<"   Length normalization     : "<<decoding_lp_alpha_<<"\n"
        <<"   Penalty beta             : "<<decoding_cp_beta_<<"\n"
        <<"   Longest sentence         : "<<longest_sentence_<<"\n"
        <<"   Print information        : "<<print_decoding_information_mode_<<"\n"
        <<"   Print alignment scores   : "<<print_alignments_scores_mode_<<"\n";
  return;
}

}