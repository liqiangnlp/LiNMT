/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/19/2016
 * Time  : 16:39
 *
 */

#include "ibm_bleu_score.h"

namespace neural_machine_translation {

/*
 * $Name: Initialize
 * $Function:
 * $Date: 20140411
 */
bool MatchInformation::Initialize( int &max_ngram, string &sentence_id ) {
  sentence_id_ = sentence_id;
  reference_count_.resize( max_ngram + 1, 0 );
  translation_count_.resize( max_ngram + 1, 0 );
  reference_information_.resize( max_ngram + 1, 0 );
  translation_information_.resize( max_ngram + 1, 0 );
  match_count_.resize( max_ngram + 1, 0 );
  return true;
}


/*
 * $Name: Process
 * $Function:
 * $Date: 20140401, in Nanjing
 * $Last Modify:
 *  20151102, 20:43, Los Angeles, USA
 */
float IbmBleuScore::Process(map<string, string> &parameters) {
  Initialize(parameters);
  ComputeNgramInfomation();
  return ScoreSystem();
}


/*
 * $Name: Initialize
 * $Function:
 * $Date: 20140411
 */
bool IbmBleuScore::Initialize(map<string, string> &parameters) {
  CheckFiles(parameters);
  translation_results_file_name_ = parameters[ "-1best" ];
  src_and_ref_file_name_ = parameters[ "-dev" ];
  output_file_name_ = parameters["-out"];
  print_information_ = parameters["-printinformation"] == "0" ? false : true;
  remove_oov_mode_ = stoi(parameters["-rmoov"].c_str());

  ofstream output_file(output_file_name_.c_str());
  if (!output_file) {
    logger<<"   ERROR: Please check the path of \"-out\":"<<" "<<output_file_name_<<"\n";
    exit(EXIT_FAILURE);
  }
  output_file.close();

  string parameter_key = "-nref";
  string default_value = "1";
  CheckEachParameter( parameters, parameter_key, default_value );
  references_number_ = atoi( parameters[ "-nref" ].c_str() );
  //if (print_information_) {
    //PrintIbmBleuScoreLogo();
    //PrintConfiguration();
  //}
  LoadingData();
  return true;
}



bool IbmBleuScore::PrintConfiguration() {
  logger<<setfill( ' ' );
  logger<<"  Configuration:"<<"\n"
        <<"      -nref            :"<<setw( 9 )<<references_number_<<"\n"
        <<"  Translation Results:"<<"\n"
        <<"      "<<translation_results_file_name_<<"\n"
        <<"  Source Sentences and References:"<<"\n"
        <<"      "<<src_and_ref_file_name_<<"\n";
  return true;
}


/*
 * $Name: CheckFiles
 * $Function: If the files used in calculating ibm bleu score do not exist, exit!
 * $Date: 2014-04-01, in Nanjing
 */
bool IbmBleuScore::CheckFiles( map< string, string > &parameters ) {
  string file_key = "-1best";
  CheckFile( parameters, file_key );

  file_key = "-dev";
  CheckFile( parameters, file_key );

  if( parameters.find( "-out" ) == parameters.end() ) {
    logger<<"   ERROR: Please add parameter \"-out\" in your command line!\n";
    exit(EXIT_FAILURE);
  }
  return true;
}


/*
 * $Name: CheckFile
 * $Function: If the file to be checked does not exist, exit!
 * $Date: 2014-04-01
 */
bool IbmBleuScore::CheckFile( map< string, string > &parameters, string &file_key ) {
  if ( parameters.find( file_key ) != parameters.end() ) {
    ifstream in_file( parameters[ file_key ].c_str() );
    if ( !in_file ) {
      logger<<"ERROR: Please check the path of \""<<file_key<<"\".\n";
      exit(EXIT_FAILURE);
    }
    in_file.clear();
    in_file.close();
  } else {
    logger<<"ERROR: Please add parameter \""<<file_key<<"\" in your command line!\n";
    exit(EXIT_FAILURE);
  }
  return true;
}


/*
 * $Name: CheckEachParameter 
 * $Function:
 * $Date: 20140401, in Nanjing
 */
bool IbmBleuScore::CheckEachParameter( map< string, string > &parameters, string &parameter_key, string &default_value ) {
  if( parameters.find( parameter_key ) == parameters.end() ) {
    parameters[ parameter_key ] = default_value;
  }
  return true;
}


/*
* $Name: LoadingData 
* $Function: Loading source sentences, references, and translation results.
* $Date: 20140411
*/
bool IbmBleuScore::LoadingData() {
  if (print_information_) {
    logger<<"\n$$ Loading source sentences and references\n";
  }
  clock_t start, finish;
  start = clock();

  ifstream src_and_ref (src_and_ref_file_name_.c_str());
  if (!src_and_ref) {
      logger<<"   ERROR: Please check the path of \""<<src_and_ref_file_name_<<"\".\n";
      exit (EXIT_FAILURE);
  }

  references_data_.resize(references_number_);
  for (size_t i = 0; i < references_number_; ++ i) {
    references_data_.at(i).reserve(5000);
  }

  string line_of_src_and_ref;
  size_t line_no = 0;
  while (getline(src_and_ref, line_of_src_and_ref))
  {
    if ((line_no % (references_number_ + 2)) == 0) {
      source_data_.push_back(line_of_src_and_ref);
    } else if ((line_no % (references_number_ + 2)) != 1) {
      string token_line;
      Tokenization(line_of_src_and_ref, token_line);
      references_data_.at((line_no % (references_number_ + 2)) - 2).push_back(token_line);
    }
    ++ line_no;
    if (print_information_) {
      if (line_no % 10000 == 0) {
        logger<<"\r   "<<line_no<<" lines";
      }
    }
  }
  sentences_number_ = (int)line_no/(references_number_ + 2);
  finish = clock();
  if (print_information_) {
    logger<<"\r   "<<line_no<<" lines, "<<sentences_number_
          <<" sentences, "<<(double)( finish - start )/CLOCKS_PER_SEC<<" seconds, "
          <<line_no/((double)( finish - start )/CLOCKS_PER_SEC)<<" sent/s\n";
  }
  src_and_ref.close();

  if (print_information_) {
    logger<<"\n$$ Loading translation results\n";
  }
  start = clock();

  ifstream translation_results( translation_results_file_name_.c_str() );
  if (!translation_results) {
      logger<<"   ERROR: Please check the path of \""<<translation_results_file_name_<<"\".\n";
      exit (EXIT_FAILURE);
  }

  string line_of_tgt;
  translation_results_data_.reserve( 5000 );
  line_no = 0;
  while( getline( translation_results, line_of_tgt ) ){
    std::vector<std::string> v_line_of_tgt;
    SplitWithString(line_of_tgt, " |||| ", v_line_of_tgt);
    line_of_tgt = v_line_of_tgt[0];

    if (remove_oov_mode_) {
      std::vector<std::string> v_words_without_oov;
      std::string tmp_line_of_tgt = line_of_tgt;
      line_of_tgt = "";
      Split(tmp_line_of_tgt, ' ', v_words_without_oov);
      for (std::vector<std::string>::iterator iter = v_words_without_oov.begin();
           iter != v_words_without_oov.end(); ++iter) {
        if (iter->front() == '<' && iter->back() == '>') {
          continue;
        } else {
          if (iter != v_words_without_oov.begin()) {
            line_of_tgt += " ";
          }
          line_of_tgt += *iter;
        }
      }
    }
   
    string token_line;
    Tokenization( line_of_tgt, token_line );
    translation_results_data_.push_back( token_line );
    ++ line_no;
    if (print_information_) {
      if( line_no % 10000 == 0 ) {
        logger<<"\r   "<<line_no<<"  sentences";
      }
    }
  } 
  if( sentences_number_ != line_no ){
    logger<<"   Error: The number of references is not the same with translations!\n";
    exit(EXIT_FAILURE);
  }
  finish = clock();
  if (print_information_) {
    logger<<"\r   "<<line_no<<" sentences, "
          <<(double)( finish - start )/CLOCKS_PER_SEC<<" seconds, "
          <<line_no/((double)( finish - start )/CLOCKS_PER_SEC)<<" sent/s\n";
  }
  src_and_ref.close();
  return true;
}


/*
* $Name: Tokenization 
* $Function:
* $Date: 20140414
*/
bool IbmBleuScore::Tokenization( string &input_string, string &output_string ) {
  string tmp_output_string;
  tmp_output_string.reserve( 5000 );
  string tmp_string;
  tmp_string.reserve( 5000 );
  output_string.reserve( 5000 );

  ClearIllegalChar( input_string );
  ConvertSgmlTags( input_string, tmp_output_string );

  // language-dependent part (assuming Western languages)
  tmp_output_string = " " + tmp_output_string + " ";

  // lowercase the uppercase letters 
  ToLower( tmp_output_string );

  tmp_string = tmp_output_string;
  tmp_output_string = "";
  TokenizePunctuation( tmp_string, tmp_output_string );
  tmp_string = tmp_output_string;
  tmp_output_string = "";
  TokenizePeriodAndComma( tmp_string, tmp_output_string );
  tmp_string = tmp_output_string;
  tmp_output_string = "";
  TokenizeDash( tmp_string, tmp_output_string );
  tmp_string = tmp_output_string;
  tmp_output_string = "";
  RemoveExtraSpace( tmp_string, tmp_output_string );
  RmStartSpace( tmp_output_string );
  RmEndSpace( tmp_output_string );
  output_string = tmp_output_string;

  return true;
}


/*
 * $Name: ConvertSgmlTags 
 * $Function:
 * $Date: 20140414
 */
bool IbmBleuScore::ConvertSgmlTags( string &input_string, string &output_string ) {
  // language-independent part
  output_string = input_string;
  size_t current_position = 0;
  while( ( current_position = output_string.find( "<skipped>", current_position ) ) != string::npos ) {
    output_string.replace( current_position, 9, "" );
  }

  /*
  current_position = 0;
  while( ( current_position = output_string.find( "-\n", current_position ) ) != string::npos ) {
    output_string.replace( current_position, 2, "" );
  }
  */

  /*
  current_position = 0;
  while( ( current_position = output_string.find( "\n", current_position ) ) != string::npos ) {
      output_string.replace( current_position, 1, "" );
  }
  */

  // convert SGML tag for quote to "
  current_position = 0;
  while( ( current_position = output_string.find( "&quot;", current_position ) ) != string::npos ) {
    output_string.replace( current_position, 6, "\"" );
  }

  // convert SGML tag for ampersand to &
  current_position = 0;
  while( ( current_position = output_string.find( "&amp;", current_position ) ) != string::npos ) {
    output_string.replace( current_position, 5, "&" );
  }

  // convert SGML tag for less-than to <
  current_position = 0;
  while( ( current_position = output_string.find( "&lt;", current_position ) ) != string::npos ) {
    output_string.replace( current_position, 4, "<" );
  }

  // convert SGML tag for greater-than to >
  current_position = 0;
  while( ( current_position = output_string.find( "&gt;", current_position ) ) != string::npos ) {
    output_string.replace( current_position, 4, ">" );
  }
  return true;
}


/*
 * $Name: TokenizePunctuation 
 * $Function:
 * $Date: 20140414
 */
bool IbmBleuScore::TokenizePunctuation( string &input_string, string &output_string ) {
  // convert "{", "|", "}", "~" to " { ", " | ", " } ", " ~ " respectively 
  // convert "[", "\", "]", "^", "_", "`" to " [ ", " \ ", " ] ", " ^ ", " _ ", " ` " respectively
  // convert "!", "\"", "#", "$", "%", "&" to " ! ", " \" ", " # ", " $ ", " % ", " & " respectively
  // convert "/" to " / "
  for( string::iterator iter = input_string.begin(); iter != input_string.end(); ++ iter ) {
    if( ( *iter >= '!' && *iter <= '&' ) || ( *iter >= '{' && *iter <= '~' ) || ( *iter >= '[' &&  *iter <= '`' ) ||
        ( *iter >= '(' && *iter <= '+' ) || ( *iter >= ':' && *iter <= '@' ) || *iter == '/' ) {
      output_string.push_back( ' ' );
      output_string.push_back( *iter );
      output_string.push_back( ' ' );
    } else {
      output_string.push_back( *iter );
    }
  }
  return true;
}


/*
* $Name: TokenizePeriodAndComma 
* $Function:
* $Date: 20140414
*/
bool IbmBleuScore::TokenizePeriodAndComma( string &input_string, string &output_string ) {
  // tokenize period and comma unless preceded by a digit
  char preceded_char = ' ';
  for( string::iterator iter = input_string.begin(); iter != input_string.end(); ++ iter ) {
    if( ( *iter == '.' || *iter == ',' ) && ( preceded_char < '0' || preceded_char > '9' ) ) {
      output_string.push_back( ' ' );
      output_string.push_back( *iter );
      output_string.push_back( ' ' );
      preceded_char = *iter;
    } else {
      output_string.push_back( *iter );
      preceded_char = *iter;
    }
  }
    
  // tokenize period and comma unless followed by a digit
  string tmp_output_string;
  char followed_char = ' ';
  for( string::reverse_iterator riter = output_string.rbegin(); riter != output_string.rend(); ++ riter ) {
    if( ( *riter == '.' || *riter == ',' ) && ( followed_char < '0' || followed_char > '9' ) ) {
      tmp_output_string.push_back( ' ' );
      tmp_output_string.push_back( *riter );
      tmp_output_string.push_back( ' ' );
      followed_char = *riter;
    } else {
      tmp_output_string.push_back( *riter );
      followed_char = *riter;
    }
  }
  output_string = "";
  for( string::reverse_iterator riter = tmp_output_string.rbegin(); riter != tmp_output_string.rend(); ++ riter ) {
      output_string.push_back( *riter );
  }
  return true;
}


/*
* $Name: TokenizeDash 
* $Function:
* $Date: 20140414
*/
bool IbmBleuScore::TokenizeDash( string &input_string, string &output_string ) {
  // tokenize dash when preceded by a digit
  string tmp_output_string = "";
  char preceded_char = ' ';
  for( string::iterator iter = input_string.begin(); iter != input_string.end(); ++ iter ) {
    if( *iter == '-' && preceded_char >= '0' && preceded_char <= '9' ) {
      tmp_output_string.push_back( ' ' );
      tmp_output_string.push_back( *iter );
      tmp_output_string.push_back( ' ' );
      preceded_char = *iter;
    } else {
      tmp_output_string.push_back( *iter );
      preceded_char = *iter;
    }
  }
  output_string = tmp_output_string;
  return true;
}


/*
* $Name: ComputeNgramInfomation 
* $Function:
* $Date: 20140411
*/
bool IbmBleuScore::ComputeNgramInfomation() {
  if (print_information_) {
    logger<<"\n$$ Convert words to n-grams\n";
  }
  clock_t start, finish;
  start = clock();

  map<string, int> ngrams_count;

  size_t line_no = 0;
  for( size_t i = 0; i < references_number_; ++ i ) {
    int number = 0;
    for( vector< string >::iterator iter = references_data_.at( i ).begin(); iter != references_data_.at( i ).end(); ++iter ) {
      vector< string > tmp_words_vec;
      Split( *iter, ' ', tmp_words_vec );
      total_words_number_ += ( int )tmp_words_vec.size();
      number += ( int )tmp_words_vec.size();
      Words2Ngrams( tmp_words_vec, ngrams_count );
      ++line_no;
      if (print_information_) {
        if( line_no % 10000 == 0 ) {
          logger<<"\r   "<<line_no<<" lines";
        }
      }
    }
  }

  finish = clock();
  if (print_information_) {
    logger<<"\r   "<<line_no<<" lines, "
          <<(double)( finish - start )/CLOCKS_PER_SEC<<" seconds, "
          <<line_no/((double)( finish - start )/CLOCKS_PER_SEC)<<" sent/s\n";
  }

  if (print_information_) {
    logger<<"\n$$ Count n-gram informations\n";
  }
  line_no = 0;
  start = clock();
  for( map< string, int >::iterator iter = ngrams_count.begin(); iter != ngrams_count.end(); ++ iter ) {
    vector< string > tmp_words_vec;
    Split( iter->first, ' ', tmp_words_vec );
    tmp_words_vec.pop_back();
    string mgram;
    bool first_flag = true;
    for( vector< string >::iterator iter_vec = tmp_words_vec.begin(); iter_vec != tmp_words_vec.end(); ++ iter_vec ) {
      if( first_flag ) {
        mgram = *iter_vec;
        first_flag = false;
      } else {
        mgram += " " + *iter_vec;
      }
    }

    ngrams_info_[ iter->first ] = - log ( mgram != "" ? (float)ngrams_count[ iter->first ]/(float)ngrams_count[ mgram ] : (float)ngrams_count[ iter->first ]/(float)total_words_number_ ) / log( 2.0f );

    ++line_no;
    if (print_information_) {
      if (line_no % 100000 == 0) {
        logger<<"\r   "<<line_no<<" sentences";
      }
    }
  }
  finish = clock();
  if (print_information_) {
    logger<<"\r   "<<line_no<<" sentences, "
          <<(double)( finish - start )/CLOCKS_PER_SEC<<" seconds, "
          <<line_no/((double)( finish - start )/CLOCKS_PER_SEC)<<" sent/s\n";
  }
  return true;
}


/*
 * $Name: Words2Ngrams 
 * $Function: Convert a string of words to an Ngram count hash
 * $Date: 20140411
 */
bool IbmBleuScore::Words2Ngrams( vector< string > &words, map< string, int > &ngrams_count ) {

  for( size_t i = 0; i < words.size(); ++ i ) {
    string ngram;
    bool first_flag = true;
    for( size_t j = i; ( j < words.size() ) && ( j - i < max_ngram_ ); ++ j ) {
      if( first_flag ) {
        ngram = words.at( j );
        ++ ngrams_count[ ngram ];
        first_flag = false;
      } else {
        ngram += " " + words.at( j );
        ++ ngrams_count[ ngram ];
      }
    }
  }

  return true;
}


/*
* $Name: ScoreSystem 
* $Function: 
* $Date: 20140416
*/
float IbmBleuScore::ScoreSystem() {
  ofstream output_file(output_file_name_.c_str());
  if (!output_file) {
    logger<<"   ERROR: Please check the path of \"-out\":"<<" "<<output_file_name_<<"\n";
    exit(EXIT_FAILURE);
  }

  MatchInformation all_match_information;
  string all_sentence_id( "all_sentence" );
  all_match_information.Initialize( max_ngram_, all_sentence_id );


  for( size_t sentence_id = 0; sentence_id < sentences_number_; ++ sentence_id ) {
    MatchInformation match_information;
    string sentence_id_str( size_tToString( sentence_id ) );
    sentence_id_str += "_sentence";
    match_information.Initialize( max_ngram_, sentence_id_str );

    ScoreSegment( sentence_id, match_information );

    CalculateBleuScoreSmoothing( match_information, output_file );
    all_match_information.current_reference_length_ += match_information.current_reference_length_;
    for( size_t i = 1; i <= max_ngram_; ++ i ) {
        all_match_information.match_count_.at( i )             += match_information.match_count_.at( i );
        all_match_information.translation_count_.at( i )       += match_information.translation_count_.at( i );
        all_match_information.reference_count_.at( i )         += match_information.reference_count_.at( i );
        all_match_information.translation_information_.at( i ) += match_information.translation_information_.at( i );
        all_match_information.reference_information_.at( i )   += match_information.reference_information_.at( i );
    }
  }
  return CalculateBleuScoreSmoothing(all_match_information, output_file);
}


/*
 * $Name: ScoreSegment 
 * $Function: 
 * $Date: 20140416
 */
bool IbmBleuScore::ScoreSegment( size_t &sentence_id, MatchInformation &match_information ) {
  map< string, int > tst_ngrams_count;
  map< string, int > ref_ngrams_max;

  vector< string > tst_words_vec;
  Split( translation_results_data_.at( sentence_id ), ' ', tst_words_vec );
  Words2Ngrams( tst_words_vec, tst_ngrams_count );

  for( size_t i = 1; i <= max_ngram_; ++ i ) {
    match_information.translation_count_.at( i ) = ( ( i <= tst_words_vec.size() ) ? tst_words_vec.size() - i + 1 : 0 ); 
  }
  
  for( size_t reference_id = 0; reference_id < references_number_; ++ reference_id ) {
    vector< string > ref_words_vec;
    map< string, int > ref_ngrams_count;

    Split( references_data_.at( reference_id ).at( sentence_id ), ' ', ref_words_vec );
    Words2Ngrams( ref_words_vec, ref_ngrams_count );
    for( map< string, int >::iterator iter = ref_ngrams_count.begin(); iter != ref_ngrams_count.end(); ++ iter ) {
      vector< string > tmp_words_vec;
      Split( iter->first, ' ', tmp_words_vec );
      match_information.reference_information_.at( tmp_words_vec.size() ) += ngrams_info_[ iter->first ];
      if ( ref_ngrams_max.find( iter->first ) != ref_ngrams_max.end() ) {
        if( ref_ngrams_max[ iter->first ] < ref_ngrams_count[ iter->first ] ) {
          ref_ngrams_max[ iter->first ] = ref_ngrams_count[ iter->first ];
        }
      } else {
        ref_ngrams_max[ iter->first ] = ref_ngrams_count[ iter->first ];
      }
    }

    for( int j = 1; j <= max_ngram_; ++ j ) {
      match_information.reference_count_.at( j ) += ( ( j <= ref_words_vec.size() ) ? ( ref_words_vec.size() - j + 1 ) : 0 );
    }

    if( match_information.current_reference_length_ == 0 ) {
      match_information.current_reference_length_ = (int)ref_words_vec.size();
    } else {
      int reference_length = (int)ref_words_vec.size();
      int candidate_length = (int)tst_words_vec.size();
      match_information.current_reference_length_ = BrevityPenaltyClosest( match_information.current_reference_length_, reference_length, candidate_length );
    }
  }

  for( map< string, int >::iterator iter = tst_ngrams_count.begin(); iter != tst_ngrams_count.end(); ++ iter ) {
    if( ref_ngrams_max.find( iter->first ) == ref_ngrams_max.end() ) {
      continue;
    } else {
      vector< string > tmp_words_vec;
      Split( iter->first, ' ', tmp_words_vec );
      match_information.translation_information_.at( tmp_words_vec.size() ) += ngrams_info_[ iter->first ] * min( tst_ngrams_count[ iter->first ], ref_ngrams_max[ iter->first ] );

      if( tst_ngrams_count[ iter->first ] <= ref_ngrams_max[ iter->first ] ) {
        match_information.match_count_.at( tmp_words_vec.size() ) += tst_ngrams_count[ iter->first ];
      } else {
        match_information.match_count_.at( tmp_words_vec.size() ) += ref_ngrams_max[ iter->first ];
      }
    }
  }
  return true;
}



/*
 * $Name: CalculateBleuScore
 * $Function: Default method used to compute the BLEU score, using smoothing.
 *            The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each precision score whose matching n-gram count is null
 *            k is 1 for the first 'n' value for which the n-gram match count is null
 *            For example, if the text contains:
 *               - one 2-gram match
 *               - and (consequently) two 1-gram matches
 *            the n-gram count for each individual precision score would be:
 *               - n=1  =>  prec_count = 2     (two unigrams)
 *               - n=2  =>  prec_count = 1     (one bigram)
 *               - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
 *               - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
* $Date: 20140416
*/
float IbmBleuScore::CalculateBleuScoreSmoothing(MatchInformation &match_information, ofstream &output_file) {
  float exp_length_score = 0.0f;
  float cur_score = 0.0f;
  float cur_iscore = 0.0f;
  float smooth = 1.0f;
  match_information.ibm_bleu_score_.resize(max_ngram_ + 1, 0.0f);

  if (match_information.translation_count_.at(1) > 0) {
    float tmp_score = 1.0f - (float)match_information.current_reference_length_ / (float)match_information.translation_count_.at(1);
    if (tmp_score < 0) {
      exp_length_score = exp(tmp_score);
    } else {
      exp_length_score = exp(0.0f);
    }
  }

  for (size_t i = 1; i <= max_ngram_; ++i) {
    if (match_information.translation_count_.at(i) == 0) {
      cur_iscore = 0.0f;
    } else if (match_information.match_count_.at(i) == 0) {
      smooth *= 2;
      cur_iscore = log(1.0f / (smooth * (float)match_information.translation_count_.at(i)));
    } else {
      cur_iscore = log((float)match_information.match_count_.at(i) / match_information.translation_count_.at(i));
    }
    cur_score += cur_iscore;
    match_information.ibm_bleu_score_.at(i) = exp(cur_score / (float)i) * exp_length_score;
  }
  output_file<<"sentence_id="<<match_information.sentence_id_<<"\tBLEU_SCORE="<<match_information.ibm_bleu_score_.at( 4 )<<"\n";
  return match_information.ibm_bleu_score_.at(4);
}

/*
 * $Name: CalculateBleuScore
 * $Function: 
 * $Date: 20140416
 */
int IbmBleuScore::BrevityPenaltyClosest( int &current_length, int &reference_length, int &candidate_length ) {
  int result = current_length;

  if( abs( candidate_length - reference_length ) <= abs( candidate_length - current_length ) ) {
    if( abs( candidate_length - reference_length ) == abs( candidate_length - current_length ) ) {
      if( current_length > reference_length ) {
        result = reference_length;
      }
    } else {
      result = reference_length;
    }
  }

  return result;
}


/*
 * $Name: PrintIbmBleuScoreLogo
 * $Function:
 * $Date: 20140401, in Nanjing
 */
bool IbmBleuScore::PrintIbmBleuScoreLogo() {
  logger<<"####### SMT ####### SMT ####### SMT ####### SMT ####### SMT #######\n"
        <<"# Calculate IBM BLEU Score                                        #\n"
        <<"#                                            Version 0.0.1        #\n"
        <<"#                                            liqiangneu@gmail.com #\n"
        <<"####### SMT ####### SMT ####### SMT ####### SMT ####### SMT #######\n";
  return true;
}



}



