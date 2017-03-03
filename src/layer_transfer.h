/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef TRANSFER_LAYER_H_
#define TRANSFER_LAYER_H_

#include "layer_loss.h"
#include "layer_hidden_to_hidden.h"
#include "layer_input_to_hidden.h"


namespace neural_machine_translation {

template <typename T>
class UpperTransferLayer {

public:
  bool upper_softmax_mode_ = true;     // This is true if the layer above is a softmax, false is hidden layer
  bool copy_h_t_mode_ = false;         // Ture if upper layer lies on different GPUs, false if not
  bool source_side_mode_ = true;

public:
  BaseLossLayer<T> *p_softmax_;

public:
  HiddenToHiddenLayer<T> *p_hidden_layer_;  

public:
  UpperTransferLayer() {};

public:
  // CHECK: OK //
  void InitUpperTransferLayer(bool upper_softmax_mode, bool copy_h_t_mode, bool source_side_mode, BaseLossLayer<T> *p_softmax, HiddenToHiddenLayer<T> *p_hidden_layer) {
    upper_softmax_mode_ = upper_softmax_mode;
    copy_h_t_mode_ = copy_h_t_mode;
    p_softmax_ = p_softmax;
    source_side_mode_ = source_side_mode;
    p_hidden_layer_ = p_hidden_layer;
  }
};


template <typename T>
class LowerTransferLayer {

public:
  bool lower_input_mode_ = true;        // This is true if the layer below is an input layer, false if hidden layer
  bool copy_d_err_ht_mode_ = false;     // True if the lower layer lies on different GPUs, false if not
  
public:
  InputToHiddenLayer<T> *p_input_layer_;
  HiddenToHiddenLayer<T> *p_hidden_layer_;

public:
  LowerTransferLayer() {};

public:
  // CHECK: OK //
  void InitLowerTransferLayer(bool lower_input, bool copy_d_err_ht, InputToHiddenLayer<T> *p_input_layer, HiddenToHiddenLayer<T> *p_hidden_layer) {
    lower_input_mode_ = lower_input;
    copy_d_err_ht_mode_ = copy_d_err_ht;
    p_input_layer_ = p_input_layer;
    p_hidden_layer_ = p_hidden_layer;
  }
};


}

#endif




