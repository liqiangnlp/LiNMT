/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/01/2016
 * Time  : 14:32
 *
 */

#ifndef DISPATCHER_H_
#define DISPATCHER_H_


#include <iostream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>

#include "debug.h"

#include "deep_rnn.h"
#include "global_configuration.h"
#include "file_helper.h"
#include "decoder.h"
#include "decoder_sentence.h"
#include "postprocess_unks.h"
#include "ibm_bleu_score.h"
#include "average_models.h"
#include "replace_vocabulary.h"
#include "word_embedding.h"
#include "byte_pair_encoding.h"


namespace neural_machine_translation {

class Dispatcher {

public:
  void Run(int argc, char **argv);

public:
  void Tuning(GlobalConfiguration &configuration);

public:
  void Decoding(GlobalConfiguration &configuration);
  void DecodingSentence(GlobalConfiguration &configuration);
  void DecodingSentenceTwoEncoders(GlobalConfiguration &configuration);

public:
  void PostProcessUnk(GlobalConfiguration &configuration);

public:
  void CalculateBleuScore(GlobalConfiguration &configuration);

public:
  void AverageModels(GlobalConfiguration &configuration);

public:
  void ReplaceVocabulary(GlobalConfiguration &configuration);

public:
  void DumpWordEmbedding(GlobalConfiguration &configuration);

public:
  void TrainBytePairEncoding(GlobalConfiguration &configuration);
  void SegmentBytePairEncoding(GlobalConfiguration &configuration);

public:
  void TrainPhraseBytePairEncoding(GlobalConfiguration &configuration);
  void SegmentPhraseBytePairEncoding(GlobalConfiguration &configuration);

public:
  void ForceDecoding(GlobalConfiguration &configuration);

};

}


#endif


