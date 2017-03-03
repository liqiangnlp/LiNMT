/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 12/16/2016
 * Time  : 16:58
 *
 */

#ifndef REPLACE_VOCABULARY_H_
#define REPLACE_VOCABULARY_H_

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include "utility_cu.h"
#include "utility_cc.h"


namespace neural_machine_translation {

class VocabularyReplacement {

public:
  BasicMethod basic_method_;

public:
  void Process(std::string &replaced_words_file_name, std::string &input_file_name, std::string &output_file_name);

};

} // end of neural_machine_translation namespace



#endif


