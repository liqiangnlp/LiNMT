/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/20/2016
 * Time  : 15:00
 *
 */

#ifndef AVERAGE_MODELS_H_
#define AVERAGE_MODELS_H_

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include "utility_cu.h"
#include "utility_cc.h"


namespace neural_machine_translation {

class AverageNeuralModels {

public:
  BasicMethod basic_method_;

public:
  void Process(std::vector<std::string> &v_input_file_names, std::string &output_file_name);

};

} // end of neural_machine_translation namespace



#endif


