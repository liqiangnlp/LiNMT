/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/19/2016
 * Time  : 16:39
 *
 */

#ifndef IBM_BLEU_SCORE_H_
#define IBM_BLEU_SCORE_H_

#include <string>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include "debug.h"
#include "utility_cc.h"
#include "utility_cu.h"

using namespace std;


namespace neural_machine_translation {

class MatchInformation {
 public:
  string            sentence_id_             ;
  int               current_reference_length_;
  vector< size_t >  match_count_             ;
  vector< size_t >  reference_count_         ;
  vector< size_t >  translation_count_       ;
  vector< float >   reference_information_   ;
  vector< float >   translation_information_ ;
  vector< float >   ibm_bleu_score_;


 public:
  MatchInformation() : current_reference_length_( 0 ) {};
  ~MatchInformation(){};

 public:
  bool Initialize( int &max_ngram, string &sentence_id );
};


class IbmBleuScore : public BasicMethod {
 public:
  IbmBleuScore() : total_words_number_( 0 ), max_ngram_( 9 ), sentences_number_( 0 ), print_information_(true) {};
  ~IbmBleuScore(){};

 public:
  int    references_number_            ;
  int    sentences_number_             ;
  string translation_results_file_name_;
  string src_and_ref_file_name_        ;   // file name for source sentences and references
  string output_file_name_             ;

 public:
  bool   print_information_;
  bool   remove_oov_mode_ = false;

 public:
  int    total_words_number_;

 public:
  int    max_ngram_;

 public:
  map< string, float > ngrams_info_;

// public:
//  vector< float > ibm_bleu_score_;

 private:
  vector< vector< string > >  references_data_;             // references_data_[ reference_id ][ sentence_id ]
  vector< string >            source_data_;
  vector< string >            translation_results_data_;

 public:
  float Process(map<string, string> &parameters);

 private:
  bool Initialize( map< string, string > &parameters );

 private:
  bool PrintConfiguration();

 private:
  bool CheckFiles( map< string, string > &parameters );

 private:
  bool CheckFile( map< string, string > &parameters, string &file_key );

 private:
  bool CheckEachParameter( map< string, string > &parameters, string &parameter_key, string &default_value );

 private:
  bool LoadingData();

 private:
  bool Tokenization( string &input_string, string &output_string );

 private:
  bool ConvertSgmlTags( string &input_string, string &output_string );

 private:
  bool TokenizePunctuation( string &input_string, string &output_string );
  bool TokenizePeriodAndComma( string &input_string, string &output_string );
  bool TokenizeDash( string &input_string, string &output_string );

 private:
  bool ComputeNgramInfomation();
  bool Words2Ngrams( vector< string > &words, map< string, int > &ngrams_count );

 private:
  float ScoreSystem();
  bool ScoreSegment( size_t &sentence_id, MatchInformation &match_information );
  float CalculateBleuScoreSmoothing(MatchInformation &match_information, ofstream &output_file);

 private:
  int BrevityPenaltyClosest( int &current_length, int &reference_length, int &candidate_length );

 public:
  static bool PrintIbmBleuScoreLogo();

};

}


#endif
