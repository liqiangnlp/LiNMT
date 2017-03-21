/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/01/2016
 * Time  : 14:33
 *
 */

#include "dispatcher.h"

namespace neural_machine_translation {
  
void Dispatcher::Run(int argc, char **argv) {

  GlobalConfiguration configuration;    // initialize configuration file

  // to prevent overflow
  curr_seed__ = static_cast<unsigned int>(std::time(0));
  curr_seed__ = std::min((unsigned int)100000000, curr_seed__);

  configuration.ParseCommandLine(argc, argv);

  SystemTime system_time;
  logger<<"\n$$ Begin time\n"
        <<"   "<<system_time.GetCurrentSystemTime()<<"\n";


  if (configuration.random_seed_mode_) {
    generator__.seed(static_cast<unsigned int>(configuration.random_seed_int_));
  } else {
    generator__.seed(static_cast<unsigned int>(std::time(0)));
  }

  if (configuration.training_mode_) {
    Tuning(configuration);
  }

  if (configuration.test_mode_) {
    ForceDecoding(configuration);
  }

  if (configuration.decode_mode_) {
    Decoding(configuration);
  }

  if (configuration.decode_sentence_mode_) {
    DecodingSentence(configuration);
  }

  if (configuration.postprocess_unk_mode_) {
    PostProcessUnk(configuration);
  }

  if (configuration.calculate_bleu_mode_) {
    CalculateBleuScore(configuration);
  }

  if (configuration.average_models_mode_) {
    AverageModels(configuration);
  }

  if (configuration.vocabulary_replacement_mode_) {
    ReplaceVocabulary(configuration);
  }

  if (configuration.dump_word_embedding_mode_) {
    DumpWordEmbedding(configuration);
  }

  if (configuration.train_bpe_mode_) {
    TrainBytePairEncoding(configuration);
  }

  if (configuration.segment_bpe_mode_) {
    SegmentBytePairEncoding(configuration);
  }

  logger<<"\n$$ End time\n"
        <<"   "<<system_time.GetCurrentSystemTime()<<"\n";
}


void Dispatcher::Tuning(GlobalConfiguration &configuration) {

  std::chrono::time_point<std::chrono::system_clock> start_total, end_total;
  std::chrono::time_point<std::chrono::system_clock> begin_minibatch, end_minibatch;
  std::chrono::time_point<std::chrono::system_clock> begin_decoding, end_decoding;
  std::chrono::time_point<std::chrono::system_clock> begin_epoch;
  std::chrono::duration<double> elapsed_seconds;

  start_total = std::chrono::system_clock::now();    // start time


   // initialize neural network machine translation
  NeuralMachineTranslation<precision> neural;

  // print parameters in configuration file
  configuration.PrintParameters();

  neural.InitModel(configuration.lstm_size_, configuration.minibatch_size_, \
                   configuration.source_vocab_size_, configuration.target_vocab_size_, \
                   configuration.longest_sentence_, configuration.debug_mode_, configuration.learning_rate_, \
                   configuration.clip_gradient_mode_, configuration.norm_clip_, \
                   configuration.input_weight_file_, configuration.output_weight_file_, \
                   configuration.softmax_scaled_mode_, configuration.training_perplexity_mode_, \
                   configuration.truncated_softmax_mode_, configuration.shortlist_size_, configuration.sampled_size_, \
                   configuration.sequence_to_sequence_mode_, configuration.layers_number_, \
                   configuration.gpu_indices_, configuration.dropout_mode_, configuration.dropout_rate_, \
                   configuration.attention_configuration_, configuration);

  if (configuration.load_model_train_mode_) {
    std::string tmp_swap_weights_file = neural.input_weight_file_;
    neural.input_weight_file_ = configuration.load_model_name_;
    neural.LoadWeights();
    neural.input_weight_file_ = tmp_swap_weights_file;
  }


  // information for averaging the speed
  int current_batch_num_speed = 0;
  const int threshold_batch_num_speed = configuration.screen_print_rate_;
  int total_words_batch_speed = 0;
  double total_batch_time_speed = 0;

  // file information for the training file
  // initialize the file information
  FileHelper file_information(configuration.train_file_name_, configuration.minibatch_size_, \
                              configuration.train_num_lines_in_file_, configuration.longest_sentence_, \
                              configuration.source_vocab_size_, configuration.target_vocab_size_, \
                              configuration.train_total_words_, configuration.truncated_softmax_mode_, \
                              configuration.shortlist_size_, configuration.sampled_size_, \
                              configuration.char_cnn_config_, configuration.char_cnn_config_.char_train_file_);

  configuration.half_way_count_ = configuration.train_total_words_ / 2;

  if (configuration.google_learning_rate_) {
    logger<<"\n$$ Fixed halve learning (google learning)\n";
    logger<<"   Half way count: "<<configuration.half_way_count_<<"\n";
  }

  int current_epoch = 1;

  logger<<"\n$$ Starting model training\n"
        <<"\n>> Starting epoch 1\n\n";

  // stuff for learning rate schedule
  int total_words = 0;
  // This is only for the google learning rate
  precision tmp_learning_rate = configuration.learning_rate_;

  // used for google learning rate for halving at every 0.5 epochs
  bool learning_rate_flag = true;
  double old_perplexity = 0;
  // set the model perplexity to zero
  neural.train_perplexity_ = 0;
    
  begin_epoch = std::chrono::system_clock::now();
  while (current_epoch <= configuration.epochs_number_) {

    begin_minibatch = std::chrono::system_clock::now();
    bool success_flag = file_information.ReadMinibatch();

    // multi_source is not written

    end_minibatch = std::chrono::system_clock::now();

    elapsed_seconds = end_minibatch - begin_minibatch;

    total_batch_time_speed += elapsed_seconds.count();

    begin_minibatch = std::chrono::system_clock::now();

    neural.InitFileInformation(&file_information);

    neural.ComputeGradients(file_information.p_host_input_vocab_indices_source_, file_information.p_host_output_vocab_indices_source_, \
                            file_information.p_host_input_vocab_indices_target_, file_information.p_host_output_vocab_indices_target_, \
                            file_information.current_source_length_, file_information.current_target_length_, \
                            file_information.p_host_input_vocab_indices_source_wgrad_, file_information.p_host_input_vocab_indices_target_wgrad_, \
                            file_information.length_source_wgrad_, file_information.length_target_wgrad_, file_information.p_host_sampled_indices_, \
                            file_information.len_unique_words_trunc_softmax_, file_information.p_host_batch_information_, &file_information);

    end_minibatch = std::chrono::system_clock::now();
    elapsed_seconds = end_minibatch - begin_minibatch;

    total_batch_time_speed += elapsed_seconds.count();
    total_words_batch_speed += file_information.words_in_minibatch_;

    ++current_batch_num_speed;
    total_words += file_information.words_in_minibatch_;
    if ((current_batch_num_speed >= threshold_batch_num_speed) || !success_flag) {

      float train_percent = (float)(((float)total_words / (float)configuration.train_total_words_) * 100.0f);
      logger<<"\r>  epoch "<<current_epoch
            <<", "<<total_words<<"/"<<configuration.train_total_words_
            <<" ("<<train_percent<<"%) tuned tokens, "
            <<(total_words_batch_speed / total_batch_time_speed)<<" tokens/s";
      //logger<<">  Past "<<current_batch_num_speed<<" minibatches, epoch "<<current_epoch<<"\n";

      //if (global_grad_clip_mode__) {
      //  logger<<"   Gradient L2 norm  : "<<global_norm_clip__<<"\n";
      //}
      //logger<<"   Number of tokens  : "<<total_words_batch_speed<<"\n";
      //logger<<"   Speed             : "<<(total_words_batch_speed / total_batch_time_speed)<<" tokens/s\n";
      //logger<<"   Time              : "<<(total_batch_time_speed / 60.0)<<" minutes\n";
      //logger<<"   Tuned tokens      : "<<total_words<<" / "<<configuration.train_total_words_<<"\n\n";

      total_words_batch_speed = 0;
      total_batch_time_speed = 0;
      current_batch_num_speed = 0;
    }

    // stuff for google learning rate
    if (configuration.google_learning_rate_ && current_epoch >= configuration.epoch_to_start_halving_ \
        && total_words >= configuration.half_way_count_ && learning_rate_flag) {

      logger<<"\n>  Half way update learning rate, fixed halve\n"
            <<"   Old learning rate : "<<tmp_learning_rate<<"\n";
      tmp_learning_rate = tmp_learning_rate / 2;
      logger<<"   New learning rate : "<<tmp_learning_rate<<"\n\n";
      neural.UpdateLearningRate(tmp_learning_rate);
      learning_rate_flag = false;
    }

    // adaptive learning with dev set, stuff for perplexity based learning schedule
    if (configuration.learning_rate_schedule_ && total_words >= configuration.half_way_count_ && learning_rate_flag) {

      logger<<"\n>  Half way adaptive learning\n";

      learning_rate_flag = false;
      double new_perplexity = neural.GetPerplexity(configuration.test_file_name_, configuration.minibatch_size_, configuration.test_num_lines_in_file_, \
                                                   configuration.longest_sentence_, configuration.source_vocab_size_, configuration.target_vocab_size_, \
                                                   false, configuration.test_total_words_, configuration.output_log_mode_, false, "");

      logger<<"   Old perplexity    : "<<old_perplexity<<"\n";
      logger<<"   New perplexity    : "<<new_perplexity<<"\n";
      logger<<"   "<<total_words<<" / "<<configuration.train_total_words_<<" tokens have been trained\n\n";

      if ((new_perplexity + configuration.margin_ >= old_perplexity) && 1 != current_epoch) {
        logger<<">  Update learning rate\n"
              <<"   Decrease factor   : "<<configuration.decrease_factor_<<"\n"
              <<"   Old learning rate : "<<tmp_learning_rate<<"\n";
        tmp_learning_rate = tmp_learning_rate * configuration.decrease_factor_;
        neural.UpdateLearningRate(tmp_learning_rate);
        logger<<"   New learning rate : "<<tmp_learning_rate<<"\n\n";
      }

      // perplexity is better so output the best model file
      if (configuration.best_model_mode_ && configuration.best_model_perp_ > new_perplexity || dump_every_best__) {
        logger<<"** Dump best model\n";
        neural.DumpBestModel(configuration.best_model_file_name_, configuration.output_weight_file_);
        configuration.best_model_perp_ = new_perplexity;
        logger<<"\n";
      }

      old_perplexity = new_perplexity;
    }

    if (!success_flag) {
      current_epoch += 1;

      // stuff for google learning rate schedule
      if (configuration.google_learning_rate_ && current_epoch >= configuration.epoch_to_start_halving_) {
        logger<<"\n>  Full way update learning rate, fixed halve\n"
              <<"   Old learning rate : "<<tmp_learning_rate<<"\n";
        tmp_learning_rate = tmp_learning_rate / 2;
        logger<<"   New learning rate : "<<tmp_learning_rate<<"\n\n";
        neural.UpdateLearningRate(tmp_learning_rate);
        learning_rate_flag = true;

      }

      // stuff for stanford learning rate schedule
      if (configuration.stanford_learning_rate_ && current_epoch >= configuration.epoch_to_start_halving_full_) {
        logger<<"\n>  Full way update learning rate, fixed halve full\n"
              <<"   Old learning rate : "<<tmp_learning_rate<<"\n";
        tmp_learning_rate = tmp_learning_rate / 2;
        logger<<"   New learning rate : "<<tmp_learning_rate<<"\n\n";
        neural.UpdateLearningRate(tmp_learning_rate);
        learning_rate_flag = true;
      }

      // adaptive learning with dev set, stuff for perplexity based learning schedule
      if (configuration.learning_rate_schedule_) {

        logger<<"\n>  Full way adaptive learning\n";

        double new_perplexity = neural.GetPerplexity(configuration.test_file_name_, configuration.minibatch_size_, configuration.test_num_lines_in_file_, \
                                                     configuration.longest_sentence_, configuration.source_vocab_size_, configuration.target_vocab_size_, \
                                                     false, configuration.test_total_words_, configuration.output_log_mode_, false, "");

        logger<<"   Old perplexity    : "<<old_perplexity<<"\n";
        logger<<"   New perplexity    : "<<new_perplexity<<"\n";
        logger<<"   "<<total_words<<" / "<<configuration.train_total_words_<<" tokens have been trained\n\n";

        if ((new_perplexity + configuration.margin_ >= old_perplexity) && 1 != current_epoch) {
          logger<<">  Update learning rate\n"
                <<"   Decrease factor   : "<<configuration.decrease_factor_<<"\n"
                <<"   Old learning rate : "<<tmp_learning_rate<<"\n";
          tmp_learning_rate = tmp_learning_rate * configuration.decrease_factor_;
          neural.UpdateLearningRate(tmp_learning_rate);
          logger<<"   New learning rate : "<<tmp_learning_rate<<"\n\n";
        }

        // perplexity is better so output the best model file
        if (configuration.best_model_mode_ && configuration.best_model_perp_ > new_perplexity || dump_every_best__) {
          logger << "** Dump best model\n";
          neural.DumpBestModel(configuration.best_model_file_name_, configuration.output_weight_file_);
          configuration.best_model_perp_ = new_perplexity;
          logger << "\n";
        }

        learning_rate_flag = true;
        old_perplexity = new_perplexity;
      }

      if (configuration.training_perplexity_mode_) {
        // change to base 2 log
        neural.train_perplexity_ = neural.train_perplexity_ / std::log(2.0);
        logger<<"\n>  Epoch "<<(current_epoch-1)<<" Done\n"
              <<"   Training data     : "<<configuration.train_file_name_<<"\n"
              <<"   sum(log2(p(x_i))) : "<<neural.train_perplexity_<<"\n"
              <<"   Total target words: "<<file_information.total_target_words_<<"\n"
              <<"   Perplexity        : "<<std::pow(2, -1 * neural.train_perplexity_/file_information.total_target_words_)<<"\n";
        elapsed_seconds = std::chrono::system_clock::now() - begin_epoch;
        logger<<"   Time              : "<<(double)elapsed_seconds.count() / 60.0<<" minutes\n";

        neural.train_perplexity_ = 0;
      }

      total_words = 0;
      if (current_epoch <= configuration.epochs_number_) {
        begin_epoch = std::chrono::system_clock::now();
        logger<<"\n\n\n>> Starting epoch "<<current_epoch<<"\n\n";
      }
    }
    DeviceSyncAll();
  }

  // Now that training is done, dump the weights
  DeviceSyncAll();

  if (!configuration.load_model_train_mode_) {
    logger << "\n** Dump final model\n";
    neural.DumpWeights();
  }

  // remove the tmp directory created
  if ("NULL" != configuration.unique_dir_) {
    boost::filesystem::path tmp_path(configuration.unique_dir_);
    //boost::filesystem::remove_all(tmp_path);
  }
  
  // compute the final runtime
  // end time
  end_total = std::chrono::system_clock::now();
  // compute the final runtime
  elapsed_seconds = end_total - start_total;
  logger<<"\n$$ Total Runtime\n" 
        <<"   "<<elapsed_seconds.count()/60.0<<" minutes\n\n";  

}


void Dispatcher::Decoding(GlobalConfiguration &configuration) {

  std::chrono::time_point<std::chrono::system_clock> start_total, end_total;
  std::chrono::time_point<std::chrono::system_clock> begin_minibatch, end_minibatch;
  std::chrono::time_point<std::chrono::system_clock> begin_epoch;
  std::chrono::duration<double> elapsed_seconds;

  start_total = std::chrono::system_clock::now();    // start time
  
  // initialize neural network machine translation
  // NeuralMachineTranslation<precision> neural;

  // print parameters in configuration file
  configuration.PrintDecodingParameters();

  EnsembleFactory<precision> ensemble_decode;
  ensemble_decode.Init(configuration.model_names_, configuration.hypotheses_number_, configuration.beam_size_, \
                       configuration.min_decoding_ratio_, configuration.penalty_, configuration.longest_sentence_, \
                       configuration.print_decoding_information_mode_, configuration.print_alignments_scores_mode_, \
                       configuration.decoder_output_file_, configuration.gpu_indices_, \
                       configuration.max_decoding_ratio_, configuration.target_vocab_size_, \
                       configuration.decoding_lp_alpha_, configuration.decoding_cp_beta_, \
                       configuration.decoding_diversity_, configuration.decoding_dump_sentence_embedding_mode_, \
                       configuration.decoding_sentence_embedding_file_name_, configuration);

  logger<<"\n$$ Starting decoding\n";
  ensemble_decode.DecodeFile();

  InputFilePreprocess input_helper;
  input_helper.UnintFile(configuration.model_names_[0], configuration.decoder_output_file_, configuration.decoder_final_file_, true, true);

  // remove the tmp directory created
  if ("NULL" != configuration.unique_dir_) {
    boost::filesystem::path tmp_path(configuration.unique_dir_);
    //boost::filesystem::remove_all(tmp_path);
  }


  // compute the final runtime
  // end time
  end_total = std::chrono::system_clock::now();
  // compute the final runtime
  elapsed_seconds = end_total - start_total;
  logger<<"\n$$ Total Runtime\n" 
        <<"   "<<elapsed_seconds.count()/60.0<<" minutes\n\n";

  if (ensemble_decode.dump_sentence_embedding_mode_) {
    ensemble_decode.out_sentence_embedding_.close();
  }
}


void Dispatcher::DecodingSentence(GlobalConfiguration &configuration) {
  logger<<"\n$$ File Information\n"
        <<"   Config file              : "<<configuration.decode_sentence_config_file_<<"\n"
        <<"   Input file               : "<<configuration.decode_sentence_input_file_<<"\n"
        <<"   Output file              : "<<configuration.decode_sentence_output_file_<<"\n";

  DecoderSentence decoder_sentence;
  decoder_sentence.Init(configuration.decode_sentence_config_file_);

  std::ifstream in_file(configuration.decode_sentence_input_file_.c_str());
  if (!in_file) {
    logger<<"   Error: can not open "<<configuration.decode_sentence_input_file_<<" file!\n";
    exit(EXIT_FAILURE);
  }


  logger<<"\n$$ Translating\n";
  std::ofstream out_file(configuration.decode_sentence_output_file_.c_str());
  if (!out_file) {
    logger<<"   Error: can not write "<<configuration.decode_sentence_output_file_<<" file!\n";
    exit(EXIT_FAILURE);
  }

  int number_of_sentences = 0;
  std::string input_sentence;
  while (std::getline(in_file, input_sentence)) {
    ++number_of_sentences;
    std::string output_sentence;
    decoder_sentence.Process(input_sentence, output_sentence);
    out_file<<"["<<number_of_sentences<<"] "<<output_sentence<<"\n";
    logger<<"\r   "<<number_of_sentences<<" sentences";
  }
  logger<<"\r   "<<number_of_sentences<<" sentences\n";

  in_file.close();
  out_file.close();
  return;
}


void Dispatcher::PostProcessUnk(GlobalConfiguration &configuration) {

  logger<<"\n$$ File Information\n" \
        <<"   Config file              : "<<configuration.unk_config_file_name_<<"\n" \
        <<"   Source decoded sentences : "<<configuration.unk_source_file_name_<<"\n" \
        <<"   1best w/ unks            : "<<configuration.unk_1best_file_name_<<"\n" \
        <<"   Output w/o unks          : "<<configuration.unk_output_file_name_<<"\n";

  PostProcessUnks post_process_unks;
  post_process_unks.Init(configuration.unk_config_file_name_);

  std::ifstream nmt_translation_stream(configuration.unk_1best_file_name_.c_str());
  std::ifstream source_stream(configuration.unk_source_file_name_.c_str());
  std::ofstream output_stream(configuration.unk_output_file_name_.c_str());
  if (!nmt_translation_stream || !source_stream || !output_stream) {
    logger<<"   Error: can not open "<<configuration.unk_1best_file_name_<<", "
          <<configuration.unk_source_file_name_<<", or "
          <<configuration.unk_output_file_name_<<" file\n";
    exit(EXIT_FAILURE);
  }

  logger<<"\n$$ Replace unks\n";
  int line_num = 0;
  std::string line_nmt_translation;
  std::string line_source;
  while (std::getline(nmt_translation_stream, line_nmt_translation)) {
    std::getline(source_stream, line_source);
    ++line_num;
    if (line_num % 1000 == 0) {
      logger<<"\r   "<<line_num<<" sentences";
    }
    std::string line_output;
    post_process_unks.Process(line_source, line_nmt_translation, line_output);
    output_stream<<line_output<<"\n";
  }
  logger<<"\r   "<<line_num<<" sentences\n";

  nmt_translation_stream.close();
  source_stream.close();
  output_stream.close();
  return;
}


void Dispatcher::CalculateBleuScore(GlobalConfiguration &configuration) {
  logger<<"\n$$ File Information\n"
        <<"   1best                    : "<<configuration.bleu_one_best_file_name_<<"\n"
        <<"   References file          : "<<configuration.bleu_dev_file_name_<<"\n"
        <<"   Number of reference      : "<<configuration.bleu_number_references_<<"\n"
        <<"   Remove oov mode          : "<<configuration.bleu_remove_oov_mode_<<"\n"
        <<"   Output file              : "<<configuration.bleu_output_file_name_<<"\n";

  std::map<string, string> m_parameters;
  m_parameters["-1best"] = configuration.bleu_one_best_file_name_;
  m_parameters["-dev"] = configuration.bleu_dev_file_name_;
  m_parameters["-out"] = configuration.bleu_output_file_name_;
  m_parameters["-printinformation"] = "1";
  m_parameters["-rmoov"] = std::to_string(configuration.bleu_remove_oov_mode_);
  m_parameters["-nref"] = std::to_string(configuration.bleu_number_references_);

  IbmBleuScore ibm_bleu_score;
  ibm_bleu_score.Process(m_parameters);

  return;
}


void Dispatcher::AverageModels(GlobalConfiguration &configuration) {
  logger<<"\n$$ File Information\n"
        <<"   Input models             :";
  for (std::vector<std::string>::iterator iter = configuration.v_average_models_input_file_.begin(); \
       iter != configuration.v_average_models_input_file_.end(); ++iter) {
    logger<<" "<<*iter;
  }
  logger<<"\n";
  logger<<"   Output model             : "<<configuration.average_models_output_file_<<"\n";

  AverageNeuralModels average_neural_models;
  average_neural_models.Process(configuration.v_average_models_input_file_, configuration.average_models_output_file_);
  

  return;
}


void Dispatcher::ReplaceVocabulary(GlobalConfiguration &configuration) {
  logger<<"\n$$ File Information\n"
        <<"   Config                   : "<<configuration.vocab_replace_config_file_name_<<"\n"
        <<"   Input model              : "<<configuration.vocab_replace_model_file_name_<<"\n"
        <<"   Output model             : "<<configuration.vocab_replace_output_file_name_<<"\n";

  VocabularyReplacement vocabulary_replacement;
  vocabulary_replacement.Process(configuration.vocab_replace_config_file_name_, configuration.vocab_replace_model_file_name_, \
                                 configuration.vocab_replace_output_file_name_);

  return;
}


void Dispatcher::DumpWordEmbedding(GlobalConfiguration &configuration) {
  logger<<"\n$$ File Information\n"
        <<"   Model                    : "<<configuration.word_embedding_model_file_name_<<"\n"
        <<"   Source Vocabulary        : "<<configuration.word_embedding_source_vocab_file_name_<<"\n"
        <<"   Target Vocabulary        : "<<configuration.word_embedding_target_vocab_file_name_<<"\n";

  WordEmbedding word_embedding;
  word_embedding.Process(configuration.word_embedding_model_file_name_, configuration.word_embedding_source_vocab_file_name_, \
                         configuration.word_embedding_target_vocab_file_name_);
  return;
}


void Dispatcher::TrainBytePairEncoding(GlobalConfiguration &configuration) {
  logger<<"\n$$ File Information\n"
        <<"   Vocabulary Size          : "<<configuration.bpe_vocabulary_size_<<"\n"
        <<"   Min Frequency            : "<<configuration.bpe_min_frequency_<<"\n"
        <<"   Input File Name          : "<<configuration.bpe_input_file_name_<<"\n"
        <<"   Output File Name         : "<<configuration.bpe_output_file_name_<<"\n";

  BytePairEncoding byte_pair_encoding;
  byte_pair_encoding.Train(configuration.bpe_vocabulary_size_, configuration.bpe_min_frequency_, 
                           configuration.bpe_input_file_name_, configuration.bpe_output_file_name_);

  return;
}


void Dispatcher::SegmentBytePairEncoding(GlobalConfiguration &configuration) {
  logger<<"\n$$ File Information\n"
        <<"   Input Codes File Name    : "<<configuration.bpe_input_codes_file_name_<<"\n"
        <<"   Input File Name          : "<<configuration.bpe_input_file_name_<<"\n"
        <<"   Output File Name         : "<<configuration.bpe_output_file_name_<<"\n";

  BytePairEncoding byte_pair_encoding;
  byte_pair_encoding.Segment(configuration.bpe_input_codes_file_name_, configuration.bpe_input_file_name_, configuration.bpe_output_file_name_);
}


void Dispatcher::ForceDecoding(GlobalConfiguration &configuration) {
  std::chrono::time_point<std::chrono::system_clock> start_total, end_total;
  std::chrono::time_point<std::chrono::system_clock> begin_minibatch, end_minibatch;
  std::chrono::time_point<std::chrono::system_clock> begin_decoding, end_decoding;
  std::chrono::time_point<std::chrono::system_clock> begin_epoch;
  std::chrono::duration<double> elapsed_seconds;

  start_total = std::chrono::system_clock::now();    // start time

  // initialize neural network machine translation
  NeuralMachineTranslation<precision> neural;

  // print parameters in configuration file
  configuration.PrintParameters();

  neural.InitModel(configuration.lstm_size_, configuration.minibatch_size_, \
                   configuration.source_vocab_size_, configuration.target_vocab_size_, \
                   configuration.longest_sentence_, configuration.debug_mode_, configuration.learning_rate_, \
                   configuration.clip_gradient_mode_, configuration.norm_clip_, \
                   configuration.input_weight_file_, configuration.output_weight_file_, \
                   configuration.softmax_scaled_mode_, configuration.training_perplexity_mode_, \
                   configuration.truncated_softmax_mode_, configuration.shortlist_size_, configuration.sampled_size_, \
                   configuration.sequence_to_sequence_mode_, configuration.layers_number_, \
                   configuration.gpu_indices_, configuration.dropout_mode_, configuration.dropout_rate_, \
                   configuration.attention_configuration_, configuration);

  neural.GetPerplexity(configuration.test_file_name_, configuration.minibatch_size_, configuration.test_num_lines_in_file_, \
                       configuration.longest_sentence_, configuration.source_vocab_size_, configuration.target_vocab_size_, true, \
                       configuration.test_total_words_, configuration.output_log_mode_, true, configuration.output_force_decode_);

  if (neural.attention_configuration_.dump_alignments_mode_) {
    InputFilePreprocess input_file_preprocess;
    // is not finished
    neural.output_alignments_.close();
    std::cerr<<"\n\nhere is not finished\n\n"<<std::flush;
  }

  // remove the tmp directory created
  if ("NULL" != configuration.unique_dir_) {
    boost::filesystem::path tmp_path(configuration.unique_dir_);
    //boost::filesystem::remove_all(tmp_path);
  }


  // compute the final runtime
  // end time
  end_total = std::chrono::system_clock::now();
  // compute the final runtime
  elapsed_seconds = end_total - start_total;
  logger<<"\n$$ Total Runtime\n" 
        <<"   "<<elapsed_seconds.count()/60.0<<" minutes\n\n";  

  
}



}


