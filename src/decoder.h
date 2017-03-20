/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */
#ifndef DECODER_H_
#define DECODER_H_

#include <string>
#include <fstream>
#include <queue>
#include <vector>

#include "global_configuration.h"
#include "deep_rnn.h"
#include "file_helper.h"
#include "file_helper_decoder.h"



namespace neural_machine_translation {

#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);


// the decoder object type
template <typename T>
class DecoderGlobalObject {
public:
  T val_;
  int beam_index_;
  int vocab_index_;

  int viterbi_alignment_;
  std::vector<T> v_alignments_scores_;

public:
  // constructor
  DecoderGlobalObject(T val, int beam_index, int vocab_index, int viterbi_alignment, std::vector<T> v_alignments_scores) {
    val_ = val;
    beam_index_ = beam_index;
    vocab_index_ = vocab_index;
    viterbi_alignment_ = viterbi_alignment;
    v_alignments_scores_ = v_alignments_scores;
  }
};


template <typename T>
class DecoderObject {

public:
  T val_;
  int vocab_index_;

  int viterbi_alignment_;
  std::vector<T> v_alignments_scores_;

public:
  DecoderObject(T val, int vocab_index, int viterbi_alignment, std::vector<T> v_alignments_scores) {
    val_ = val;
    vocab_index_ = vocab_index;
    viterbi_alignment_ = viterbi_alignment;
    v_alignments_scores_ = v_alignments_scores;
  }

};


template <typename T>
class KBest {
public:
  T score_;
  T index_;

public:
  // constructor
  KBest(T score, T index) {
    score_ = score;
    index_ = index;
  }
};




class PQCompareFunctor {
public:
  template <typename T>
  bool operator() (DecoderObject<T> &a, DecoderObject<T> &b) const {
    return (a.val_ < b.val_);
  }
};





class PQGlobalCompareFunctor {

public:
  template <typename T>
  bool operator() (DecoderGlobalObject<T> &a, DecoderGlobalObject<T> &b) const { 
    return (a.val_ < b.val_); 
  }
};


class KBestCompareFunctor {
public:
  template <typename T>
  bool operator() (KBest<T> &a, KBest<T> &b) const {
    return (a.score_ < b.score_);
  }
};



template <typename T>
class EigenMatWrapper {
public:
  Eigen::Matrix<int, Eigen::Dynamic, 1> eigen_hypothesis_;
  Eigen::Matrix<int, Eigen::Dynamic, 1> eigen_viterbi_alignments_;
  Eigen::Matrix<std::vector<T>, Eigen::Dynamic, 1> eigen_alignments_scores_;

  T score_;         // log prob score along with a penalty

public:
  EigenMatWrapper(int size) {
    eigen_hypothesis_.resize(size);
    eigen_viterbi_alignments_.resize(size);
    eigen_alignments_scores_.resize(size);
  }
};




////////////////////////// CLASS
////////////////////////// Decoder
template <typename T>
class Decoder {
public:
  int beam_size_;
  int vocab_size_;                // target vocab size
  int start_symbol_;              // index of <START> is 0
  int end_symbol_;                // index of <EOF> is 1
  int max_decoding_length_;       // max size of a translation
  T min_decoding_ratio_;          // min size of a translation
  int current_index_;             // The current length of the decoded target sentence
  int num_hypotheses_;            // The number of hypotheses to be output for each translation
  T penalty_;                     // penalty term to encourage longer hypotheses, tune for bleu score
  std::string output_file_name_;  // output integerize file
  std::ofstream output_;          // ofstream for 1best-int
  bool print_decoding_information_mode_;
  bool print_alignments_scores_mode_ = false;
  //bool print_score_mode_;
  //bool print_alignment_mode_;
  //bool print_unk_alignment_mode_;


public:
//  std::priority_queue<DecoderGlobalObject<T>, std::vector<DecoderGlobalObject<T>>, PQGlobalCompareFunctor> pq_global_;
  std::priority_queue<DecoderObject<T>, std::vector<DecoderObject<T>>, PQCompareFunctor> pq_;
  std::priority_queue<DecoderGlobalObject<T>, std::vector<DecoderGlobalObject<T>>, PQGlobalCompareFunctor> pq_global_;


public:
  std::vector<EigenMatWrapper<T>> v_hypotheses_;  // Stores all hypotheses


#ifdef EIGEN
public:
  Eigen::Matrix<int, Eigen::Dynamic, 1> eigen_current_indices_;                         // beam_size_ x 1

  // size (beam size) * (max decoder length)
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> eigen_top_sentences_;              // beam_size_ x max_decoding_length_
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> eigen_top_sentences_tmp_;          // beam_size_ x max_decoding_length_, 
                                                                                        // tmp to copy old ones into this

  // size (beam size) * (max decoder length)
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> eigen_top_sentences_viterbi_;      // beam_size_ x max_decoding_length_
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> eigen_top_sentences_viterbi_tmp_;  // beam_size_ x max_decoding_length_
                                                                                        // tmp to copy old ones into this

  Eigen::Matrix<std::vector<T>, Eigen::Dynamic, Eigen::Dynamic> eigen_top_sentences_alignments_scores_;
  Eigen::Matrix<std::vector<T>, Eigen::Dynamic, Eigen::Dynamic> eigen_top_sentences_alignments_scores_tmp_;

  // size (beam size) * 1, score are stored as log probabilities
  Eigen::Matrix<T, Eigen::Dynamic, 1> eigen_top_sentences_scores_;                      // beam_size_ x 1
  Eigen::Matrix<T, Eigen::Dynamic, 1> eigen_top_sentences_scores_tmp_;                  // beam_size_ x 1

  Eigen::Matrix<int, Eigen::Dynamic, 1> eigen_new_indices_changes_;                     // beam_size_ x 1 
                                                                                        // used to swap around hidden and cell states based on new beam results
#endif

public:
  int *p_host_current_indices_;         // beam_size_ x 1
  int *p_device_current_indices_;

public:
  Decoder() {}
  ~Decoder();

public:
  void Init(int beam_size, int vocab_size, int start_symbol, int end_symbol, int max_decoding_length, \
            T min_decoding_ratio, T penalty, std::string output_file_name, int num_hypotheses, \
            bool print_decoding_information_mode, bool print_alignments_scores_mode);

  void InitDecoderSentence(int beam_size, int vocab_size, int start_symbol, int end_symbol, int max_decoding_length, \
                           T min_decoding_ratio, T penalty, std::string output_file_name, int num_hypotheses, \
                           bool print_decoding_information_mode, bool print_alignments_scores_mode);


public:
  void InitDecoder();

public:
  template <typename Derived>
  void ExpandHypothesis(const Eigen::MatrixBase<Derived> &eigen_outputdist, int index, \
                        std::vector<int> &viterbi_alignments, std::vector<T> &all_alignments_scores, \
                        T diversity);

public:
  void EmptyQueueGlobal();
  void EmptyQueuePQ();


public:
  template <typename Derived>
  void FinishCurrentHypotheses(const Eigen::MatrixBase<Derived> &eigen_outputdist, \
                               std::vector<int> &viterbi_alignments, std::vector<T> &v_alignments_scores);

public:
  void OutputKBestHypotheses(int source_length, T lp_alpha, T cp_beta);
  void ObtainOneBestHypothesis(int source_length, T lp_alpha, T cp_beta, std::string &output_sentence);


};


template <typename T>
void Decoder<T>::Init(int beam_size, int vocab_size, int start_symbol, int end_symbol, int max_decoding_length, \
                      T min_decoding_ratio, T penalty, std::string output_file_name, int num_hypotheses, \
                      bool print_decoding_information_mode, bool print_alignments_scores_mode) {

  beam_size_ = beam_size;
  vocab_size_ = vocab_size;
  start_symbol_ = start_symbol;
  end_symbol_ = end_symbol;
  max_decoding_length_ = max_decoding_length;
  min_decoding_ratio_ = min_decoding_ratio;
  penalty_ = penalty;
  output_file_name_ = output_file_name;
  num_hypotheses_ = num_hypotheses;

  print_decoding_information_mode_ = print_decoding_information_mode;
  print_alignments_scores_mode_ = print_alignments_scores_mode;

  output_.open(output_file_name.c_str());
  output_<<std::fixed<<std::setprecision(3);

  eigen_current_indices_.resize(beam_size);
    
  eigen_top_sentences_.resize(beam_size, max_decoding_length);
  eigen_top_sentences_tmp_.resize(beam_size, max_decoding_length);

  eigen_top_sentences_viterbi_.resize(beam_size, max_decoding_length);
  eigen_top_sentences_viterbi_tmp_.resize(beam_size, max_decoding_length);

  eigen_top_sentences_alignments_scores_.resize(beam_size, max_decoding_length);
  eigen_top_sentences_alignments_scores_tmp_.resize(beam_size, max_decoding_length);

  eigen_top_sentences_scores_.resize(beam_size);
  eigen_top_sentences_scores_tmp_.resize(beam_size);

  eigen_new_indices_changes_.resize(beam_size);

  p_host_current_indices_ = (int *)malloc(beam_size * 1 * sizeof(int));
}


template <typename T>
void Decoder<T>::InitDecoderSentence(int beam_size, int vocab_size, int start_symbol, int end_symbol, int max_decoding_length, \
                                     T min_decoding_ratio, T penalty, std::string output_file_name, int num_hypotheses, \
                                     bool print_decoding_information_mode, bool print_alignments_scores_mode) {

  beam_size_ = beam_size;
  vocab_size_ = vocab_size;
  start_symbol_ = start_symbol;
  end_symbol_ = end_symbol;
  max_decoding_length_ = max_decoding_length;
  min_decoding_ratio_ = min_decoding_ratio;
  penalty_ = penalty;
  num_hypotheses_ = num_hypotheses;

  print_decoding_information_mode_ = print_decoding_information_mode;
  print_alignments_scores_mode_ = print_alignments_scores_mode;

  eigen_current_indices_.resize(beam_size);
    
  eigen_top_sentences_.resize(beam_size, max_decoding_length);
  eigen_top_sentences_tmp_.resize(beam_size, max_decoding_length);

  eigen_top_sentences_viterbi_.resize(beam_size, max_decoding_length);
  eigen_top_sentences_viterbi_tmp_.resize(beam_size, max_decoding_length);

  eigen_top_sentences_alignments_scores_.resize(beam_size, max_decoding_length);
  eigen_top_sentences_alignments_scores_tmp_.resize(beam_size, max_decoding_length);

  eigen_top_sentences_scores_.resize(beam_size);
  eigen_top_sentences_scores_tmp_.resize(beam_size);

  eigen_new_indices_changes_.resize(beam_size);

  p_host_current_indices_ = (int *)malloc(beam_size * 1 * sizeof(int));
}


template <typename T>
Decoder<T>::~Decoder() {
  output_.close();
}



template <typename T>
void Decoder<T>::InitDecoder() {
  current_index_ = 0;
  eigen_top_sentences_scores_.setZero();
  v_hypotheses_.clear();
  
  for (int i = 0; i < beam_size_; ++i) {
    eigen_current_indices_(i) = start_symbol_;
    p_host_current_indices_[i] = start_symbol_;
  }

  for (int i = 0; i < beam_size_; ++i) {
    for (int j = 0; j < max_decoding_length_; ++j) {
      eigen_top_sentences_(i, j) = start_symbol_;
      eigen_top_sentences_viterbi_(i, j) = -20;
    }
  }
}


template <typename T>
template <typename Derived>
void Decoder<T>::ExpandHypothesis(const Eigen::MatrixBase<Derived> &eigen_outputdist, int index, \
                                  std::vector<int> &viterbi_alignments, std::vector<T> &all_alignments_scores, \
                                  T diversity) {

  if (all_alignments_scores.size() != 0 && all_alignments_scores.size() != beam_size_ * max_decoding_length_) {
    logger<<"Error: all_alignments_scores error!\n";
    exit(EXIT_FAILURE);
  }

  if (all_alignments_scores.size() == 0) {
    for (int i = 0; i < beam_size_; ++i) {
      for (int j = 0; j < max_decoding_length_; ++j) {
        all_alignments_scores.push_back(0);
      }
    }
  }
    
  std::vector<std::vector<T> > v_v_alignments_scores(beam_size_);   // beam_size x longest_sentence
  for (int i = 0; i < beam_size_; ++i) {
    for (int j = 0; j < max_decoding_length_; ++j) {
      v_v_alignments_scores.at(i).push_back(all_alignments_scores[IDX2C(j, i, max_decoding_length_)]);
    }
  }
  
  if (viterbi_alignments.size() != 0 && viterbi_alignments.size() != beam_size_) {
    logger<<"Error: viterbi alignment error\n";
    exit(EXIT_FAILURE);
  }

  if (viterbi_alignments.size() == 0) {
    for (int i = 0; i < beam_size_; ++i) {
      viterbi_alignments.push_back(-1);
    }
  }

  int cols = eigen_outputdist.cols();

  if (index == 0) {
    cols = 1;
  }

  EmptyQueueGlobal();

  for (int i = 0; i < cols; ++i) {
    EmptyQueuePQ();

    for (int j = 0; j < eigen_outputdist.rows(); ++j) {
      if (pq_.size() < beam_size_ + 1) {
        pq_.push(DecoderObject<T>(-eigen_outputdist(j, i), j, viterbi_alignments[i], v_v_alignments_scores.at(i)));
      } else {
        if (-eigen_outputdist(j, i) < pq_.top().val_) {
          pq_.pop();
          pq_.push(DecoderObject<T>(-eigen_outputdist(j, i), j, viterbi_alignments[i], v_v_alignments_scores.at(i)));
        }
      }
    }

    // Now have the top elements
    while (!pq_.empty()) {
      int rank = pq_.size();

      // diversity
      T rank_penalty = (T)rank * diversity;
      
      DecoderObject<T> tmp = pq_.top();
      pq_.pop();

      T current_pq_global_score = std::log(-tmp.val_) + eigen_top_sentences_scores_(i) - rank_penalty;

      pq_global_.push(DecoderGlobalObject<T>(current_pq_global_score, i, tmp.vocab_index_, tmp.viterbi_alignment_, tmp.v_alignments_scores_));
      //pq_global_.push(DecoderGlobalObject<T>(std::log(-tmp.val_) + eigen_top_sentences_scores_(i), i, tmp.vocab_index_, tmp.viterbi_alignment_, tmp.v_alignments_scores_));
    }
  }

  // Now have global heap with (beam_size * beam_size) elements
  // Go through until (beam size) new hypotheses
  int i = 0;
  while (i < beam_size_) {
    DecoderGlobalObject<T> tmp = pq_global_.top();
    pq_global_.pop();

    if (tmp.vocab_index_ != start_symbol_) {
      if (tmp.vocab_index_ == end_symbol_) {
        v_hypotheses_.push_back(EigenMatWrapper<T>(current_index_ + 2));

        v_hypotheses_.back().eigen_hypothesis_ = eigen_top_sentences_.block(tmp.beam_index_, 0, 1, current_index_ + 2).transpose(); 
        v_hypotheses_.back().eigen_hypothesis_(current_index_ + 1) = end_symbol_;

        v_hypotheses_.back().eigen_viterbi_alignments_ = eigen_top_sentences_viterbi_.block(tmp.beam_index_, 0, 1, current_index_ + 2).transpose();
        v_hypotheses_.back().eigen_viterbi_alignments_(current_index_ + 1) = tmp.viterbi_alignment_;

        v_hypotheses_.back().eigen_alignments_scores_ = eigen_top_sentences_alignments_scores_.block(tmp.beam_index_, 0, 1, current_index_ + 2).transpose();
        v_hypotheses_.back().eigen_alignments_scores_(current_index_ + 1) = tmp.v_alignments_scores_;

        v_hypotheses_.back().score_ = tmp.val_ + penalty_;
      } else {

        eigen_top_sentences_tmp_.row(i) = eigen_top_sentences_.row(tmp.beam_index_);
        eigen_top_sentences_tmp_(i, current_index_ + 1) = tmp.vocab_index_;

        eigen_top_sentences_viterbi_tmp_.row(i) = eigen_top_sentences_viterbi_.row(tmp.beam_index_);
        eigen_top_sentences_viterbi_tmp_(i, current_index_ + 1) = tmp.viterbi_alignment_;

        eigen_top_sentences_alignments_scores_tmp_.row(i) = eigen_top_sentences_alignments_scores_.row(tmp.beam_index_);
        eigen_top_sentences_alignments_scores_tmp_(i, current_index_ + 1) = tmp.v_alignments_scores_;

        eigen_current_indices_(i) = tmp.vocab_index_;
        eigen_new_indices_changes_(i) = tmp.beam_index_;

        eigen_top_sentences_scores_tmp_(i) = tmp.val_ + penalty_;

        ++i;
      }
    }
  }

  eigen_top_sentences_ = eigen_top_sentences_tmp_;
  eigen_top_sentences_viterbi_ = eigen_top_sentences_viterbi_tmp_;
  eigen_top_sentences_alignments_scores_ = eigen_top_sentences_alignments_scores_tmp_;
  eigen_top_sentences_scores_ = eigen_top_sentences_scores_tmp_;
  current_index_ += 1;

  for (int i = 0; i < beam_size_; ++i) {
    p_host_current_indices_[i] = eigen_current_indices_(i);
  }
}



template <typename T>
void Decoder<T>::EmptyQueueGlobal() {
  while (!pq_global_.empty()) {
    pq_global_.pop();
  }
}


template <typename T>
void Decoder<T>::EmptyQueuePQ() {
  while (!pq_.empty()) {
    pq_.pop();
  }
}


template <typename T>
template <typename Derived>
void Decoder<T>::FinishCurrentHypotheses(const Eigen::MatrixBase<Derived> &eigen_outputdist, \
                                         std::vector<int> &viterbi_alignments, std::vector<T> &v_alignments_scores) {
  for (int i = 0; i < beam_size_; ++i) {
    eigen_top_sentences_(i, current_index_ + 1) = end_symbol_;
    eigen_top_sentences_scores_(i) += std::log(eigen_outputdist(1, i)) + penalty_;
    v_hypotheses_.push_back(EigenMatWrapper<T>(current_index_ + 2));
    v_hypotheses_.back().eigen_hypothesis_ = eigen_top_sentences_.block(i, 0, 1, current_index_ + 2).transpose();
    
    v_hypotheses_.back().eigen_viterbi_alignments_ = eigen_top_sentences_viterbi_.block(i, 0, 1, current_index_ + 2).transpose();
    v_hypotheses_.back().eigen_viterbi_alignments_(current_index_ + 1) = viterbi_alignments[i];

    std::vector<std::vector<T> > v_v_alignments_scores(beam_size_);   // beam_size x longest_sentence
    for (int k = 0; k < beam_size_; ++k) {
      for (int j = 0; j < max_decoding_length_; ++j) {
        v_v_alignments_scores.at(k).push_back(v_alignments_scores[IDX2C(j, k, max_decoding_length_)]);
      }
    }
    v_hypotheses_.back().eigen_alignments_scores_ = eigen_top_sentences_alignments_scores_.block(i, 0, 1, current_index_ + 2).transpose();
    v_hypotheses_.back().eigen_alignments_scores_(current_index_ + 1) = v_v_alignments_scores.at(i);

    v_hypotheses_.back().score_ = eigen_top_sentences_scores_(i);
  }
  current_index_ += 1;
}


template <typename T>
void Decoder<T>::OutputKBestHypotheses(int source_length, T lp_alpha, T cp_beta) {
  std::priority_queue<KBest<T>, std::vector<KBest<T>>, KBestCompareFunctor> pq_best_hypotheses;

  // length normalation, lp(Y) = ((5 + |Y|)^lp_alpha) / ((5 + 1)^lp_alpha)
  if (0 != lp_alpha) {
    for (int i = 0; i < v_hypotheses_.size(); ++i) {
      T lp = powf((T)(5 + v_hypotheses_[i].eigen_hypothesis_.size()), lp_alpha) / powf((T)(5 + 1), lp_alpha);
      v_hypotheses_[i].score_ /= lp;
    }
  }

  // penalty, cp(X;Y) = cp_beta * sum_{i=1}^{|X|}log(min(sum_{j=1}^{|Y|}p_{i,j}, 1.0))
  if (0 != cp_beta) {
    for (int i = 0; i < v_hypotheses_.size(); ++i) {
      // normalization, sum to 1.0
      for (int j = 1; j < v_hypotheses_[i].eigen_hypothesis_.size() - 1; ++j) {
        T sum = 0;
        for (int k = 0; k < source_length; ++k) {
          //v_hypotheses_[i].eigen_alignments_scores_(j + 1).at(k) += 0.01;
          //sum += v_hypotheses_[i].eigen_alignments_scores_(j + 1).at(k);
          v_hypotheses_[i].eigen_alignments_scores_(j).at(k) += 0.01;
          sum += v_hypotheses_[i].eigen_alignments_scores_(j).at(k);
        }
        for (int k = 0; k < source_length; ++k) {
          //v_hypotheses_[i].eigen_alignments_scores_(j + 1).at(k) /= sum;
          v_hypotheses_[i].eigen_alignments_scores_(j).at(k) /= sum;
        }
      }  

      //
      T cp_x_y = 0;
      for (int k = 0; k < source_length; ++k) {
        T sum = 0;
        for (int j = 1; j < v_hypotheses_[i].eigen_hypothesis_.size() - 1; ++j) {
          //sum += v_hypotheses_[i].eigen_alignments_scores_(j + 1).at(k);
          sum += v_hypotheses_[i].eigen_alignments_scores_(j).at(k);
        }

        cp_x_y += std::log(std::min(sum, (T)1.0));
      }
      cp_x_y *= cp_beta;
      if (cp_x_y < -20) {
        cp_x_y = -20;
      }
      v_hypotheses_[i].score_ += cp_x_y;
    }
  }


  T len_ratio;
  for (int i = 0; i < v_hypotheses_.size(); ++i) {
    len_ratio = ((T)v_hypotheses_[i].eigen_hypothesis_.size()) / source_length;
    if (len_ratio > min_decoding_ratio_) {
      if (pq_best_hypotheses.size() < num_hypotheses_) {
        pq_best_hypotheses.push(KBest<T>(-v_hypotheses_[i].score_, i));
      } else {
        if (-1 * pq_best_hypotheses.top().score_ < v_hypotheses_[i].score_) {
          pq_best_hypotheses.pop();
          pq_best_hypotheses.push(KBest<T>(-v_hypotheses_[i].score_, i));
        }
      }
    }
  }

  // for making k-best list descending
  std::priority_queue<KBest<T>, std::vector<KBest<T>>, KBestCompareFunctor> pq_best_hypotheses_tmp;
  while (!pq_best_hypotheses.empty()) {
    pq_best_hypotheses_tmp.push(KBest<T>(-1 * pq_best_hypotheses.top().score_, pq_best_hypotheses.top().index_));
    pq_best_hypotheses.pop();
  }

  while (!pq_best_hypotheses_tmp.empty()) {

    bool first_word_flag = true;
    for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
      if (first_word_flag) {
        output_<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_(j);
        first_word_flag = false;
      } else {
        output_<<" "<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_(j);
      }
    }

    if (print_decoding_information_mode_) {
      output_<<" |||| "<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].score_;

      output_<<" ||||";
      int alignment_number = 0;
      for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
        ++alignment_number;
        //output_<<" "<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_viterbi_alignments_(j + 1);
        output_<<" "<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_viterbi_alignments_(j);
      }
      if (0 == alignment_number) {
        output_<<" ";
      }

      output_<<" ||||";
      int unk_number = 0;
      for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
        if (2 == v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_(j)) {
          ++unk_number;
          //output_<<" "<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_viterbi_alignments_(j + 1);
          output_<<" "<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_viterbi_alignments_(j);
        }
      }
      if (0 == unk_number) {
        output_<<" -1";
      }

      if (print_alignments_scores_mode_) {
        output_<<" ||||";
        int alignments_scores_number = 0;
        for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
          ++alignments_scores_number;
          output_<<" ";
          for (int k = 0; k < source_length; ++k) {
            if (0 != k) {
              output_<<"/";
            }
            //output_<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_alignments_scores_(j + 1).at(k);
            output_<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_alignments_scores_(j).at(k);
          }
        }
        if (0 == alignments_scores_number) {
          output_<<" ";
        }
      }
    }
    output_<<"\n";


    /*
    if (unk_replacement_mode__) {
      for (int j = 0; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size(); ++j) {
        if (v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_(j) == 2) {
          unk_rep_file_stream__<<v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_viterbi_alignments_(j + 1)<<" ";
        }
      }
      unk_rep_file_stream__<<"\n";
      unk_rep_file_stream__.flush();
    }
    */

    pq_best_hypotheses_tmp.pop();
  }

  if (1 != num_hypotheses_) {
    output_<<"================\n";
  }

  output_.flush();
}


template <typename T>
void Decoder<T>::ObtainOneBestHypothesis(int source_length, T lp_alpha, T cp_beta, std::string &output_sentence) {

  std::priority_queue<KBest<T>, std::vector<KBest<T>>, KBestCompareFunctor> pq_best_hypotheses;

  // length normalation, lp(Y) = ((5 + |Y|)^lp_alpha) / ((5 + 1)^lp_alpha)
  if (0 != lp_alpha) {
    for (int i = 0; i < v_hypotheses_.size(); ++i) {
      T lp = powf((T)(5 + v_hypotheses_[i].eigen_hypothesis_.size()), lp_alpha) / powf((T)(5 + 1), lp_alpha);
      v_hypotheses_[i].score_ /= lp;
    }
  }

  // penalty, cp(X;Y) = cp_beta * sum_{i=1}^{|X|}log(min(sum_{j=1}^{|Y|}p_{i,j}, 1.0))
  if (0 != cp_beta) {
    for (int i = 0; i < v_hypotheses_.size(); ++i) {
      // normalization, sum to 1.0
      for (int j = 1; j < v_hypotheses_[i].eigen_hypothesis_.size() - 1; ++j) {
        T sum = 0;
        for (int k = 0; k < source_length; ++k) {
          v_hypotheses_[i].eigen_alignments_scores_(j).at(k) += 0.01;
          sum += v_hypotheses_[i].eigen_alignments_scores_(j).at(k);
        }
        for (int k = 0; k < source_length; ++k) {
          v_hypotheses_[i].eigen_alignments_scores_(j).at(k) /= sum;
        }
      }  

      //
      T cp_x_y = 0;
      for (int k = 0; k < source_length; ++k) {
        T sum = 0;
        for (int j = 1; j < v_hypotheses_[i].eigen_hypothesis_.size() - 1; ++j) {
          sum += v_hypotheses_[i].eigen_alignments_scores_(j).at(k);
        }

        cp_x_y += std::log(std::min(sum, (T)1.0));
      }
      cp_x_y *= cp_beta;
      if (cp_x_y < -20) {
        cp_x_y = -20;
      }
      v_hypotheses_[i].score_ += cp_x_y;
    }
  }


  T len_ratio;
  for (int i = 0; i < v_hypotheses_.size(); ++i) {
    len_ratio = ((T)v_hypotheses_[i].eigen_hypothesis_.size()) / source_length;
    if (len_ratio > min_decoding_ratio_) {
      if (pq_best_hypotheses.size() < num_hypotheses_) {
        pq_best_hypotheses.push(KBest<T>(-v_hypotheses_[i].score_, i));
      } else {
        if (-1 * pq_best_hypotheses.top().score_ < v_hypotheses_[i].score_) {
          pq_best_hypotheses.pop();
          pq_best_hypotheses.push(KBest<T>(-v_hypotheses_[i].score_, i));
        }
      }
    }
  }

  // for making k-best list descending
  std::priority_queue<KBest<T>, std::vector<KBest<T>>, KBestCompareFunctor> pq_best_hypotheses_tmp;
  while (!pq_best_hypotheses.empty()) {
    pq_best_hypotheses_tmp.push(KBest<T>(-1 * pq_best_hypotheses.top().score_, pq_best_hypotheses.top().index_));
    pq_best_hypotheses.pop();
  }


  BasicMethod basic_method;

  while (!pq_best_hypotheses_tmp.empty()) {

    bool first_word_flag = true;
    for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
      if (first_word_flag) {
        output_sentence += basic_method.IntToString(v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_(j));
        first_word_flag = false;
      } else {
        output_sentence += " " + basic_method.IntToString(v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_(j));
      }
    }

    if (print_decoding_information_mode_) {
      output_sentence += " |||| " + basic_method.FloatToString(v_hypotheses_[pq_best_hypotheses_tmp.top().index_].score_);

      output_sentence += " ||||";

      int alignment_number = 0;
      for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
        ++alignment_number;
        output_sentence += " " + basic_method.IntToString(v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_viterbi_alignments_(j));
      }
      if (0 == alignment_number) {
        output_sentence += " ";
      }

      output_sentence += " ||||";
      int unk_number = 0;
      for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
        if (2 == v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_(j)) {
          ++unk_number;
          output_sentence += " " + basic_method.IntToString(v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_viterbi_alignments_(j));
        }
      }
      if (0 == unk_number) {
        output_sentence += " -1";
      }

      if (print_alignments_scores_mode_) {
        output_sentence += " ||||";
        int alignments_scores_number = 0;
        for (int j = 1; j < v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_hypothesis_.size() - 1; ++j) {
          ++alignments_scores_number;
          output_sentence += " ";
          for (int k = 0; k < source_length; ++k) {
            if (0 != k) {
              output_sentence += "/";
            }
            output_sentence += basic_method.FloatToString(v_hypotheses_[pq_best_hypotheses_tmp.top().index_].eigen_alignments_scores_(j).at(k));
          }
        }
        if (0 == alignments_scores_number) {
          output_sentence += " ";
        }
      }
    }
    pq_best_hypotheses_tmp.pop();
  }

  return;
}




////////////////////////// CLASS
////////////////////////// DecoderModelWrapper
template <typename T>
class NeuralMachineTranslation;


// The entire model will lie on one GPU, but different models in the ensemble can lie on different GPUs.
template <typename T>
class DecoderModelWrapper {

public: 
  int gpu_number_;                               // ID of current gpu
  int *p_device_ones_;                           // beamsize x 1
  T *p_host_outputdist_;                         // target_vocab_size_ x beam_size_
  T *p_device_tmp_swap_values_;                  // lstm_size_ x beam_size_
  int *p_device_input_vocab_indices_source_;     // longest_sentence
  int *p_device_current_indices_;                // beam_size_

public:
  NeuralMachineTranslation<T> *p_neuralmt_;

public:
  FileHelperDecoder *p_file_helper_;            // for file input, so each file can get read in seperately
  FileHelperDecoder *p_file_helper_multi_src_;  // reads in additional multi-source file

public:
  int source_length_;  // current length of the source sentence being decoded
  int beam_size_;
  int source_vocab_size_;
  int target_vocab_size_;
  int num_layers_;
  int lstm_size_;
  bool attention_model_mode_;
  bool feed_input_mode_;
  bool combine_lstm_mode_;
  int num_lines_in_file_ = -1;
  int longest_sentence_;

  bool multi_source_mode_ = false;
  int source_length_bi_;             // current length of the source sentence being decoded
  int *p_device_input_vocab_indices_source_bi_;

  bool char_cnn_mode_ = false;
  int *p_device_char_vocab_indices_source_;
  int longest_word_;
  std::unordered_map<int, std::vector<int>> uno_word_to_char_map_;  // for word index, what is the character sequence, this is read from a file
  int *p_host_new_char_indices_;
  int *p_device_new_char_indices_;

  std::string main_weight_file_;                    // model file name
  std::string multi_source_weight_file_;            // model file name for multi-source
  std::string main_integerized_file_;               // integerized decoded file
  std::string multi_source_integerized_file_;       // integerized decoded file for multi-source


public:
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_outputdist_;      // target_vocab_size_ x beam_size_

public:
  std::vector<int> v_viterbi_alignments_individual_;   // individual viterbi alignments before voting
  std::vector<T> v_viterbi_alignments_scores_;         // individual viterbi scores


public:
  DecoderModelWrapper() {}

public:
  void Init(int gpu_num, int beam_size, std::string main_weight_file, std::string multi_src_weight_file, std::string main_integerized_file, \
            std::string multi_source_integerized_file, int longest_sentence, GlobalConfiguration &config);

  void InitDecoderSentence(int gpu_num, int beam_size, std::string main_weight_file, int longest_sentence, GlobalConfiguration &config);


public:
  void ExtractModelInformation(std::string weights_file_name);          // get how many layers, hiddenstate size, vocab sizes, etc

public:
  void MemcpyVocabIndices();
  void MemcpyVocabIndicesSentence(const std::vector<int> &v_input_sentence_int);

public:
  void ForwardPropSource();
  void ForwardPropTarget(int curr_index, int *p_host_current_indices);

public:
  void DumpSentenceEmbedding(std::ofstream &out_sentence_embedding);

public:
  template <typename Derived>
  void CopyDistToEigen(T *p_host_outputdist_, const Eigen::MatrixBase<Derived> &eigen_outputdist_const);

public:
  template <typename Derived>
  void SwapDecodingStates(const Eigen::MatrixBase<Derived> &eigen_indices, int index);

public:
  void TargetCopyPrevStates();
};


template <typename T>
void DecoderModelWrapper<T>::Init(int gpu_num, int beam_size, std::string main_weight_file, std::string multi_src_weight_file, \
                                  std::string main_integerized_file, std::string multi_source_integerized_file, \
                                  int longest_sentence, GlobalConfiguration &config) {

  gpu_number_ = gpu_num;
  beam_size_ = beam_size;
  longest_sentence_ = config.longest_sentence_;

  main_weight_file_ = main_weight_file;
  multi_source_weight_file_ = multi_src_weight_file;
  main_integerized_file_ = main_integerized_file;
  multi_source_integerized_file_ = multi_source_integerized_file;

  // now switch to the current GPU
  cudaSetDevice(gpu_num);

  // get model parameters from the model file
  ExtractModelInformation(main_weight_file);

  // allocate p_device_ones_
  CudaErrorWrapper(cudaMalloc((void**)&p_device_ones_, beam_size * 1 * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_ones_ failed\n");
  // set p_device_ones_ to all 1
  OnesMatKernel<<<1, 256>>>(p_device_ones_, beam_size);
  cudaDeviceSynchronize();

  // allocate the output distribution on the CPU
  p_host_outputdist_ = (T *)malloc(target_vocab_size_ * beam_size * sizeof(T));
  eigen_outputdist_.resize(target_vocab_size_, beam_size);

  // allocate the swap values
  CudaErrorWrapper(cudaMalloc((void **)&p_device_tmp_swap_values_, lstm_size_ * beam_size * sizeof(T)), "DecoderModelWrapper::DecoderModelWrapper p_device_tmp_swap_values_ failed\n");

  // allocate the input vocab indices
  CudaErrorWrapper(cudaMalloc((void **)&p_device_input_vocab_indices_source_, longest_sentence * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_input_vocab_indices_source_ failed\n");

  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
    ;
  }

  // now initialize the file input
  p_file_helper_ = new FileHelperDecoder();
  p_file_helper_->Init(main_integerized_file, num_lines_in_file_, config.longest_sentence_);

  if (multi_source_integerized_file != "NULL") {
    p_file_helper_multi_src_ = new FileHelperDecoder();
    p_file_helper_multi_src_->Init(multi_source_integerized_file, num_lines_in_file_, config.longest_sentence_);
  }

  if (multi_source_mode_) {
    CudaErrorWrapper(cudaMalloc((void **)&p_device_input_vocab_indices_source_bi_, longest_sentence * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_input_vocab_indices_source_bi_ failed\n");
  }

  // allocate the current indices
  CudaErrorWrapper(cudaMalloc((void **)&p_device_current_indices_, beam_size * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_current_indices_ failed\n");

  p_neuralmt_ = new NeuralMachineTranslation<T>();
  p_neuralmt_->InitModelDecoding(lstm_size_, beam_size, source_vocab_size_, target_vocab_size_, \
                                  num_layers_, main_weight_file, gpu_num, config, attention_model_mode_, \
                                  feed_input_mode_, multi_source_mode_, combine_lstm_mode_, char_cnn_mode_);

  // initialize additional stuff for model
  p_neuralmt_->InitPreviousStates(num_layers_, lstm_size_, beam_size, gpu_num, multi_source_mode_);

  // load in weights for the model
  p_neuralmt_->LoadWeights();

}


template <typename T>
void DecoderModelWrapper<T>::InitDecoderSentence(int gpu_num, int beam_size, std::string main_weight_file, int longest_sentence, GlobalConfiguration &config) {

  gpu_number_ = gpu_num;
  beam_size_ = beam_size;
  longest_sentence_ = config.longest_sentence_;

  main_weight_file_ = main_weight_file;
  //multi_source_weight_file_ = multi_src_weight_file;
  //main_integerized_file_ = main_integerized_file;
  //multi_source_integerized_file_ = multi_source_integerized_file;

  // now switch to the current GPU
  cudaSetDevice(gpu_num);

  // get model parameters from the model file
  ExtractModelInformation(main_weight_file);

  // allocate p_device_ones_
  CudaErrorWrapper(cudaMalloc((void**)&p_device_ones_, beam_size * 1 * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_ones_ failed\n");
  // set p_device_ones_ to all 1
  OnesMatKernel<<<1, 256>>>(p_device_ones_, beam_size);
  cudaDeviceSynchronize();

  // allocate the output distribution on the CPU
  p_host_outputdist_ = (T *)malloc(target_vocab_size_ * beam_size * sizeof(T));
  eigen_outputdist_.resize(target_vocab_size_, beam_size);

  // allocate the swap values
  CudaErrorWrapper(cudaMalloc((void **)&p_device_tmp_swap_values_, lstm_size_ * beam_size * sizeof(T)), "DecoderModelWrapper::DecoderModelWrapper p_device_tmp_swap_values_ failed\n");

  // allocate the input vocab indices
  CudaErrorWrapper(cudaMalloc((void **)&p_device_input_vocab_indices_source_, longest_sentence * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_input_vocab_indices_source_ failed\n");

  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
    ;
  }

  // now initialize the file input
  p_file_helper_ = new FileHelperDecoder();
  p_file_helper_->InitDecoderSentence(config.longest_sentence_);

  /*
  if (multi_source_integerized_file != "NULL") {
    p_file_helper_multi_src_ = new FileHelperDecoder();
    p_file_helper_multi_src_->Init(multi_source_integerized_file, num_lines_in_file_, config.longest_sentence_);
  }
  */

  if (multi_source_mode_) {
    CudaErrorWrapper(cudaMalloc((void **)&p_device_input_vocab_indices_source_bi_, longest_sentence * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_input_vocab_indices_source_bi_ failed\n");
  }

  // allocate the current indices
  CudaErrorWrapper(cudaMalloc((void **)&p_device_current_indices_, beam_size * sizeof(int)), "DecoderModelWrapper::DecoderModelWrapper p_device_current_indices_ failed\n");

  p_neuralmt_ = new NeuralMachineTranslation<T>();
  p_neuralmt_->InitModelDecoding(lstm_size_, beam_size, source_vocab_size_, target_vocab_size_, \
                                  num_layers_, main_weight_file, gpu_num, config, attention_model_mode_, \
                                  feed_input_mode_, multi_source_mode_, combine_lstm_mode_, char_cnn_mode_);

  // initialize additional stuff for model
  p_neuralmt_->InitPreviousStates(num_layers_, lstm_size_, beam_size, gpu_num, multi_source_mode_);

  // load in weights for the model
  p_neuralmt_->LoadWeights();

}


template <typename T>
void DecoderModelWrapper<T>::ExtractModelInformation(std::string weights_file_name) {
  logger<<"\n$$ Model Parameters\n";

  std::ifstream weight_stream;
  weight_stream.open(weights_file_name.c_str());

  std::vector<std::string> model_params;
  std::string tmp_line;
  std::string tmp_word;

  std::getline(weight_stream, tmp_line);
  std::istringstream my_ss(tmp_line, std::istringstream::in);
  while (my_ss >> tmp_word) {
    model_params.push_back(tmp_word);
  }

  if (model_params.size() != 9) {
    logger<<"Error: model format is not correct for decoding with file: "<<weights_file_name<<"\n";
  }

  num_layers_ = std::stoi(model_params[0]);
  lstm_size_ = std::stoi(model_params[1]);
  target_vocab_size_ = std::stoi(model_params[2]);
  source_vocab_size_ = std::stoi(model_params[3]);
  attention_model_mode_ = std::stoi(model_params[4]);
  feed_input_mode_ = std::stoi(model_params[5]);
  multi_source_mode_ = std::stoi(model_params[6]);
  combine_lstm_mode_ = std::stoi(model_params[7]);
  char_cnn_mode_ = std::stoi(model_params[8]);

  logger<<"   Input nmt model          : "<<weights_file_name<<"\n"
        <<"   Layers number            : "<<num_layers_<<"\n"
        <<"   Lstm size                : "<<lstm_size_<<"\n"
        <<"   Target vocabulary size   : "<<target_vocab_size_<<"\n"
        <<"   Source vocabulary size   : "<<source_vocab_size_<<"\n";

  if (attention_model_mode_) {
    logger<<"   Attention mode           : TRUE\n";
  } else {
    logger<<"   Attention mode           : FALSE\n";
  }

  if (feed_input_mode_) {
    logger<<"   Feed input mode          : TRUE\n";
  } else {
    logger<<"   Feed input mode          : FALSE\n";
  }

  if (multi_source_mode_) {
    logger<<"   Multi source mode        : TRUE\n";
  } else {
    logger<<"   Multi source mode        : FALSE\n";
  }

  if (combine_lstm_mode_) {
    logger<<"   Tree combine lstm mode   : TRUE\n";
  } else {
    logger<<"   Tree combine lstm mode   : FALSE\n";
  }

  if (char_cnn_mode_) {
    logger<<"   Char RNN mode            : TRUE\n";
  } else {
    logger<<"   Char RNN mode            : FALSE\n";
  }

  weight_stream.close();
}


template <typename T>
void DecoderModelWrapper<T>::MemcpyVocabIndices() {
  p_file_helper_->ReadSentence();

  source_length_ = p_file_helper_->sentence_length_;

  if (multi_source_mode_) {
    // multi_source_mode_ is not written
  } else {
    source_length_bi_ = 0;
  }

  cudaSetDevice(gpu_number_);

  CudaErrorWrapper(cudaMemcpy(p_device_input_vocab_indices_source_, p_file_helper_->p_host_input_vocab_indices_source_, source_length_ * sizeof(int), cudaMemcpyHostToDevice), "decoder memcpy_vocab_indices 1\n");

  if (multi_source_mode_) {
    // multi_source_mode_ is not written
  }

  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
  }

  if (attention_model_mode_) {
    for (int i = 0; i < beam_size_; ++i) {
      CudaErrorWrapper(cudaMemcpy(p_neuralmt_->decoder_attention_layer_.p_device_batch_information_ + i, \
                       p_file_helper_->p_host_batch_information_, 1 * sizeof(int), cudaMemcpyHostToDevice), "decoder memcpy_vocab_indices 2\n");
      CudaErrorWrapper(cudaMemcpy(p_neuralmt_->decoder_attention_layer_.p_device_batch_information_ + beam_size_ + i, \
                       p_file_helper_->p_host_batch_information_ + 1, 1 * sizeof(int), cudaMemcpyHostToDevice), "decoder memcpy_vocab_indices 2\n");

      if (multi_source_mode_) {
        // multi_source_mode_ is not written
      }
    }
  }
}


template <typename T>
void DecoderModelWrapper<T>::MemcpyVocabIndicesSentence(const std::vector<int> &v_input_sentence_int) {
  p_file_helper_->UseSentence(v_input_sentence_int);

  source_length_ = p_file_helper_->sentence_length_;

  if (multi_source_mode_) {
    // multi_source_mode_ is not written
  } else {
    source_length_bi_ = 0;
  }

  cudaSetDevice(gpu_number_);

  CudaErrorWrapper(cudaMemcpy(p_device_input_vocab_indices_source_, p_file_helper_->p_host_input_vocab_indices_source_, source_length_ * sizeof(int), cudaMemcpyHostToDevice), "decoder memcpy_vocab_indices 1\n");

  if (multi_source_mode_) {
    // multi_source_mode_ is not written
  }

  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
  }

  if (attention_model_mode_) {
    for (int i = 0; i < beam_size_; ++i) {
      CudaErrorWrapper(cudaMemcpy(p_neuralmt_->decoder_attention_layer_.p_device_batch_information_ + i, \
                       p_file_helper_->p_host_batch_information_, 1 * sizeof(int), cudaMemcpyHostToDevice), "decoder memcpy_vocab_indices 2\n");
      CudaErrorWrapper(cudaMemcpy(p_neuralmt_->decoder_attention_layer_.p_device_batch_information_ + beam_size_ + i, \
                       p_file_helper_->p_host_batch_information_ + 1, 1 * sizeof(int), cudaMemcpyHostToDevice), "decoder memcpy_vocab_indices 2\n");

      if (multi_source_mode_) {
        // multi_source_mode_ is not written
      }
    }
  }
}


template <typename T>
void DecoderModelWrapper<T>::ForwardPropSource() {
  p_neuralmt_->ForwardPropSource(p_device_input_vocab_indices_source_, p_device_input_vocab_indices_source_bi_, \
                                 p_device_ones_, source_length_, source_length_bi_, lstm_size_, p_device_char_vocab_indices_source_);

  p_neuralmt_->source_length_ = source_length_;
}


template <typename T>
void DecoderModelWrapper<T>::DumpSentenceEmbedding(std::ofstream &out_sentence_embedding) {
  p_neuralmt_->DumpSentenceEmbedding(lstm_size_, out_sentence_embedding);
  return;
}



template <typename T>
void DecoderModelWrapper<T>::ForwardPropTarget(int curr_index, int *p_host_current_indices) {

  cudaSetDevice(gpu_number_);
  CudaErrorWrapper(cudaMemcpy(p_device_current_indices_, p_host_current_indices, beam_size_ * sizeof(int), cudaMemcpyHostToDevice), "forward prop target decoder 1\n");

  if (char_cnn_mode_) {
    // char_cnn_mode_ is not written
  }

  p_neuralmt_->ForwardPropTarget(curr_index, p_device_current_indices_, p_device_ones_, lstm_size_, beam_size_, p_device_new_char_indices_);

  cudaSetDevice(gpu_number_);

  // copy the outputdist to CPU
  cudaDeviceSynchronize();
  CudaGetLastError("Error Above!!");

  CudaErrorWrapper(cudaMemcpy(p_host_outputdist_, p_neuralmt_->p_softmax_layer_->GetDistPtr(), target_vocab_size_ * beam_size_ * sizeof(T), cudaMemcpyDeviceToHost), "forward prop target decoder 2\n");

  // copy the outputdist to eigen from CPU
  CopyDistToEigen(p_host_outputdist_, eigen_outputdist_);

  if (unk_replacement_mode__) {
    v_viterbi_alignments_individual_ = viterbi_alignments__;
    v_viterbi_alignments_scores_ = alignment_scores__;
  }

}



template <typename T>
template <typename Derived>
void DecoderModelWrapper<T>::CopyDistToEigen(T *p_host_outputdist_, const Eigen::MatrixBase<Derived> &eigen_outputdist_const) {
  UNCONST(Derived, eigen_outputdist_const, eigen_outputdist_);
  for (int i = 0; i < eigen_outputdist_.rows(); ++i) {
    for (int j = 0; j < eigen_outputdist_.cols(); ++j) {
      eigen_outputdist_(i, j) = p_host_outputdist_[IDX2C(i, j, eigen_outputdist_.rows())];  
    }
  }
}


template <typename T>
template <typename Derived>
void DecoderModelWrapper<T>::SwapDecodingStates(const Eigen::MatrixBase<Derived> &eigen_indices, int index) {
  p_neuralmt_->SwapDecodingStates(eigen_indices, index, p_device_tmp_swap_values_);
}

template <typename T>
void DecoderModelWrapper<T>::TargetCopyPrevStates() {
  p_neuralmt_->TargetCopyPrevStates(lstm_size_, beam_size_);
}

////////////////////////// CLASS
////////////////////////// EnsembleFactory
template <typename T>
class EnsembleFactory {

public:
  std::vector<DecoderModelWrapper<T> > v_models_;       // nmt models loaded for decoding

public:
  Decoder<T> *p_decoder_;                               // pass the output dists to this

public:
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_outputdist_;       // target_vocab_size_ x beam_size_
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_normalization_;                     // 1 x beam_size_

public:
  int num_lines_in_file_;                               // how many lines in --decoded-files
  int target_vocab_size_;                               // target vocabulary size, must agree on all models
  int longest_sentence_;                                // set a max to the longest sentence that could be decoded by the decoder
  T max_decoding_ratio_;

public:
  T lp_alpha_;
  T cp_beta_;
  T diversity_;

public:
  bool dump_sentence_embedding_mode_ = false;           // dump sentence embedding mode
  std::ofstream out_sentence_embedding_;                // out stream for sentence embedding

  const int start_symbol_ = 0;                          // index of <START> is 0
  const int end_symbol_ = 1;                            // index of <EOF> is 1

public:
  EnsembleFactory() {}

public:
  void Init(std::vector<std::string> weight_file_names, int num_hypotheses, int beam_size, T min_decoding_ratio, \
            T penalty, int longest_sentence, bool print_decoding_information_mode, \
            bool print_alignments_scores_mode, \
            std::string decoder_output_file, std::vector<int> v_gpu_nums, T max_decoding_ratio, \
            int target_vocab_size, T lp_alpha, T cp_beta, \
            T diversity, bool dump_sentence_embedding_mode, \
            std::string &sentence_embedding_file_name, GlobalConfiguration &config);

  void InitDecoderSentence(std::vector<std::string> weight_file_names, int num_hypotheses, int beam_size, T min_decoding_ratio, \
                           T penalty, int longest_sentence, bool print_decoding_information_mode, \
                           bool print_alignments_scores_mode, \
                           std::string decoder_output_file, std::vector<int> v_gpu_nums, \
                           T max_decoding_ratio, int target_vocab_size, T lp_alpha, T cp_beta, \
                           T diversity, bool dump_sentence_embedding_mode, \
                           std::string &sentence_embedding_file_name, GlobalConfiguration &config);

public:
  void DecodeFile();
  void DecodeSentence(const std::vector<int> &v_input_sentence_int, std::string &output_sentence);

public:
  void EnsemblesModels();
};


template <typename T>
void EnsembleFactory<T>::InitDecoderSentence(std::vector<std::string> weight_file_names, int num_hypotheses, int beam_size, T min_decoding_ratio, \
                                             T penalty, int longest_sentence, bool print_decoding_information_mode, \
                                             bool print_alignments_scores_mode, \
                                             std::string decoder_output_file, std::vector<int> v_gpu_nums, \
                                             T max_decoding_ratio, int target_vocab_size, T lp_alpha, T cp_beta, \
                                             T diversity, bool dump_sentence_embedding_mode, \
                                             std::string &sentence_embedding_file_name, GlobalConfiguration &config) {
  // Get the target vocab from the first file
  target_vocab_size_ = target_vocab_size;
  max_decoding_ratio_ = max_decoding_ratio;
  longest_sentence_ = longest_sentence;

  lp_alpha_ = lp_alpha;
  cp_beta_ = cp_beta;
  diversity_ = diversity;

  dump_sentence_embedding_mode_ = dump_sentence_embedding_mode;
  if (dump_sentence_embedding_mode_) {
    out_sentence_embedding_.open(sentence_embedding_file_name);
    out_sentence_embedding_<<std::fixed<<std::setprecision(9);
  }

  // to make sure beam search does halt
  if (beam_size > (int)std::sqrt(target_vocab_size)) {
    beam_size = (int)std::sqrt(target_vocab_size);
  }

  p_decoder_ = new Decoder<T>();
  p_decoder_->InitDecoderSentence(beam_size, target_vocab_size, start_symbol_, end_symbol_, \
                                  longest_sentence, min_decoding_ratio, penalty, decoder_output_file, \
                                  num_hypotheses, print_decoding_information_mode, print_alignments_scores_mode);

  // initialize all of the models
  DecoderModelWrapper<T> decoder_model_wrapper;
  decoder_model_wrapper.InitDecoderSentence(v_gpu_nums[0], beam_size, config.model_names_[0], longest_sentence, config);
  v_models_.push_back(decoder_model_wrapper);

  // check to be sure all modes have the same target vocab size and vocab indices and get the target vocab size
  target_vocab_size_ = v_models_[0].target_vocab_size_;
  for (int i = 0; i < v_models_.size(); ++i) {
    if (v_models_[0].target_vocab_size_ != target_vocab_size) {
      logger<<"Error: The target vocabulary sizes are not same in ensemble\n";
      exit(EXIT_FAILURE);
    }
  }

  // resize the outputdist that gets sent to the decoder
  eigen_outputdist_.resize(target_vocab_size, beam_size);
  eigen_normalization_.resize(1, beam_size);
}



template <typename T>
void EnsembleFactory<T>::Init(std::vector<std::string> weight_file_names, int num_hypotheses, int beam_size, T min_decoding_ratio, \
                              T penalty, int longest_sentence, bool print_decoding_information_mode, \
                              bool print_alignments_scores_mode, \
                              std::string decoder_output_file, std::vector<int> v_gpu_nums, \
                              T max_decoding_ratio, int target_vocab_size, T lp_alpha, T cp_beta, \
                              T diversity, bool dump_sentence_embedding_mode, \
                              std::string &sentence_embedding_file_name, GlobalConfiguration &config) {


  // Get the target vocab from the first file
  target_vocab_size_ = target_vocab_size;
  max_decoding_ratio_ = max_decoding_ratio;
  longest_sentence_ = longest_sentence;

  lp_alpha_ = lp_alpha;
  cp_beta_ = cp_beta;
  diversity_ = diversity;

  dump_sentence_embedding_mode_ = dump_sentence_embedding_mode;
  if (dump_sentence_embedding_mode_) {
    out_sentence_embedding_.open(sentence_embedding_file_name);
    out_sentence_embedding_<<std::fixed<<std::setprecision(9);
  }

  // to make sure beam search does halt
  if (beam_size > (int)std::sqrt(target_vocab_size)) {
    beam_size = (int)std::sqrt(target_vocab_size);
  }

  std::ifstream tmp_input;
  // first integerize file to be decoded
  tmp_input.open(config.decode_tmp_files_[0]);
  // get number of sentences
  cc_util::GetFileStatsSource(num_lines_in_file_, tmp_input);
  tmp_input.close();

  p_decoder_ = new Decoder<T>();
  p_decoder_->Init(beam_size, target_vocab_size, start_symbol_, end_symbol_, \
                  longest_sentence, min_decoding_ratio, penalty, decoder_output_file, \
                  num_hypotheses, print_decoding_information_mode, print_alignments_scores_mode);

  // initialize all of the models
  for (int i = 0; i < weight_file_names.size(); ++i) {
    DecoderModelWrapper<T> decoder_model_wrapper;
    decoder_model_wrapper.Init(v_gpu_nums[i], beam_size, config.model_names_[i], config.model_names_multi_source_[i], \
                              config.decode_tmp_files_[i], config.decode_tmp_files_additional_[i], longest_sentence, config);
    v_models_.push_back(decoder_model_wrapper);
  }

  // check to be sure all modes have the same target vocab size and vocab indices and get the target vocab size
  target_vocab_size_ = v_models_[0].target_vocab_size_;
  for (int i = 0; i < v_models_.size(); ++i) {
    if (v_models_[0].target_vocab_size_ != target_vocab_size) {
      logger<<"Error: The target vocabulary sizes are not same in ensemble\n";
      exit(EXIT_FAILURE);
    }
  }

  // resize the outputdist that gets sent to the decoder
  eigen_outputdist_.resize(target_vocab_size, beam_size);
  eigen_normalization_.resize(1, beam_size);
}


template <typename T>
void EnsembleFactory<T>::DecodeFile() {

  std::chrono::time_point<std::chrono::system_clock> begin_decoding, end_decoding;
  begin_decoding = std::chrono::system_clock::now();

  for (int i = 0; i < num_lines_in_file_; ++i) {

    for (int j = 0; j < v_models_.size(); ++j) {
      v_models_[j].MemcpyVocabIndices();
    }

    DeviceSyncAll();

    // init decoder
    p_decoder_->InitDecoder();

    // run forward prop on the source
    for (int j = 0; j < v_models_.size(); ++j) {
      v_models_[j].ForwardPropSource();
      if (dump_sentence_embedding_mode_) {
        v_models_[j].DumpSentenceEmbedding(out_sentence_embedding_);
      }
    }
    int last_index = 0;

    // for dumping hidden states we can just return
    if (tsne_dump_mode__) {
      continue;
    }

    int source_length = std::max(v_models_[0].source_length_, v_models_[0].source_length_bi_);
    for (int curr_index = 0; curr_index < std::min((int)(max_decoding_ratio_ * source_length), longest_sentence_ - 2); curr_index++) {

      for (int j = 0; j < v_models_.size(); ++j) {

        v_models_[j].ForwardPropTarget(curr_index, p_decoder_->p_host_current_indices_);   
        // now take the viterbi alignments

      }

      // now ensemble the models together
      // this also does voting for unk-replacement
      EnsemblesModels();

      // run decoder for this iteration
      p_decoder_->ExpandHypothesis(eigen_outputdist_, curr_index, viterbi_alignments__, alignment_scores__, diversity_);

      // swap the decoding states
      for (int j = 0; j < v_models_.size(); ++j) {
        v_models_[j].SwapDecodingStates(p_decoder_->eigen_new_indices_changes_, curr_index);
        v_models_[j].TargetCopyPrevStates();
      }

      // for the scores of the last hypothesis
      last_index = curr_index;
    }

    // now run one last iteration
    for (int j = 0; j < v_models_.size(); ++j) {
      v_models_[j].ForwardPropTarget(last_index + 1, p_decoder_->p_host_current_indices_);
    }

    // output the final results of the decoder
    EnsemblesModels();
    p_decoder_->FinishCurrentHypotheses(eigen_outputdist_, viterbi_alignments__, alignment_scores__);
    p_decoder_->OutputKBestHypotheses(source_length, lp_alpha_, cp_beta_);

    end_decoding = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_decoding = end_decoding - begin_decoding;
    logger<<"\r   "<<i + 1<<" out of "<<num_lines_in_file_<<" sentences, "
          <<(float)(i+1)/elapsed_seconds_decoding.count()<<" sentences/s";    
  }
  logger<<"\n";
}


template <typename T>
void EnsembleFactory<T>::DecodeSentence(const std::vector<int> &v_input_sentence_int, std::string &output_sentence) {

  for (int j = 0; j < v_models_.size(); ++j) {
    v_models_[j].MemcpyVocabIndicesSentence(v_input_sentence_int);
  }

  DeviceSyncAll();

  // init decoder
  p_decoder_->InitDecoder();

  // run forward prop on the source
  for (int j = 0; j < v_models_.size(); ++j) {
    v_models_[j].ForwardPropSource();
    if (dump_sentence_embedding_mode_) {
      v_models_[j].DumpSentenceEmbedding(out_sentence_embedding_);
    }
  }
  int last_index = 0;

  // for dumping hidden states we can just return
  if (tsne_dump_mode__) {
    return;
  }

  int source_length = std::max(v_models_[0].source_length_, v_models_[0].source_length_bi_);
  for (int curr_index = 0; curr_index < std::min((int)(max_decoding_ratio_ * source_length), longest_sentence_ - 2); curr_index++) {
    for (int j = 0; j < v_models_.size(); ++j) {
      v_models_[j].ForwardPropTarget(curr_index, p_decoder_->p_host_current_indices_);   
      // now take the viterbi alignments
    }

    // now ensemble the models together
    // this also does voting for unk-replacement
    EnsemblesModels();

    // run decoder for this iteration
    p_decoder_->ExpandHypothesis(eigen_outputdist_, curr_index, viterbi_alignments__, alignment_scores__, diversity_);

    // swap the decoding states
    for (int j = 0; j < v_models_.size(); ++j) {
      v_models_[j].SwapDecodingStates(p_decoder_->eigen_new_indices_changes_, curr_index);
      v_models_[j].TargetCopyPrevStates();
    }

    // for the scores of the last hypothesis
    last_index = curr_index;
  }

  // now run one last iteration
  for (int j = 0; j < v_models_.size(); ++j) {
    v_models_[j].ForwardPropTarget(last_index + 1, p_decoder_->p_host_current_indices_);
  }

  // output the final results of the decoder
  EnsemblesModels();
  p_decoder_->FinishCurrentHypotheses(eigen_outputdist_, viterbi_alignments__, alignment_scores__);
  p_decoder_->ObtainOneBestHypothesis(source_length, lp_alpha_, cp_beta_, output_sentence);

  return;
}



template <typename T>
void EnsembleFactory<T>::EnsemblesModels() {
  int num_models = v_models_.size();

  for (int i = 0; i < eigen_outputdist_.rows(); ++i) {
    for (int j = 0; j < eigen_outputdist_.cols(); ++j) {
      double tmp_sum = 0;
      for (int k = 0; k < v_models_.size(); ++k) {
        tmp_sum += v_models_[k].eigen_outputdist_(i, j);
      }
      eigen_outputdist_(i, j) = tmp_sum / num_models;
    }
  }

  eigen_normalization_.setZero();

  for (int i = 0; i < eigen_outputdist_.rows(); ++i) {
    eigen_normalization_ += eigen_outputdist_.row(i);
  }

  for (int i = 0; i < eigen_outputdist_.rows(); ++i) {
    eigen_outputdist_.row(i) = (eigen_outputdist_.row(i).array() / eigen_normalization_.array()).matrix();
  }

  // now averaging alignment scores for unk replacement
  if (unk_replacement_mode__) {
    // average the scores
    for (int i = 0; i < v_models_[0].longest_sentence_; ++i) {
      for (int j = 0; j < v_models_[0].beam_size_; ++j) {
        T tmp_sum = 0;
        
        for (int k = 0; k < v_models_.size(); ++k) {
          tmp_sum += v_models_[k].v_viterbi_alignments_scores_[IDX2C(i, j, v_models_[0].longest_sentence_)];
        }

        alignment_scores__[IDX2C(i, j, v_models_[0].longest_sentence_)] = tmp_sum;
      }
    }

    // choose the max and fill in viterbi_alignments__
    for (int i = 0; i < v_models_[0].beam_size_; ++i) {
      T max_val = 0;
      int max_index = -1;
      
      for (int j = 0; j < v_models_[0].longest_sentence_; ++j) {
        T tmp_val = alignment_scores__[IDX2C(j, i, v_models_[0].longest_sentence_)];

        if (tmp_val > max_val) {
          max_val = tmp_val;
          max_index = j;
        }
      }

      viterbi_alignments__[i] = max_index;
    }
  }
}


}

#endif


