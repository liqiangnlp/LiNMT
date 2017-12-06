/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/21/2017
 * Time  : 18:59
 *
 */

#ifndef ANOTHER_ENCODER_H_
#define ANOTHER_ENCODER_H_

#include "tree_lstm.h"


namespace neural_machine_translation {

template <typename T>
class NeuralMachineTranslation;


template <typename T>
class AnotherEncoder {
public:
  int num_layers_;
  int lstm_size_;
  int minibatch_size_;
  T norm_clip_;
  T learning_rate_;
  int longest_sentence_minibatch_s1_ = -1;
  int longest_sentence_minibatch_s2_ = -1;

  bool decode_mode_ = false;

  std::vector<int> v_gpu_indices_;

  NeuralMachineTranslation<T> *p_model_;

public:
  bool lstm_combine_mode_ = false;
  std::vector<TreeLstm<T>*> v_p_lstm_combiner_layers_;

public:
  std::vector<T*> v_p_device_hs_final_target_;
  std::vector<T*> v_p_device_horiz_param_s1_;
  std::vector<T*> v_p_device_horiz_param_s2_;
  std::vector<T*> v_p_device_horiz_bias_;
  std::vector<T*> v_p_device_ct_final_target_;
  std::vector<T*> v_p_device_horiz_param_s1_ct_;
  std::vector<T*> v_p_device_horiz_param_s2_ct_;

public:
  // for norm clipping
  std::vector<T*> v_p_device_tmp_result_;
  std::vector<T*> v_p_device_result_;



public:
  void InitLayerDecoder(NeuralMachineTranslation<T> *p_model, int gpu_num, bool lstm_combine_mode, int lstm_size, int num_layers);
  void LoadWeights(std::ifstream &input_stream);

public:
  void ForwardProp();
};


template<typename T>
void AnotherEncoder<T>::InitLayerDecoder(NeuralMachineTranslation<T> *p_model, int gpu_num, bool lstm_combine_mode, int lstm_size, int num_layers) {
  num_layers_ = num_layers;
  lstm_size_ = lstm_size;
  minibatch_size_ = 1;
  p_model_ = p_model;
  lstm_combine_mode_ = lstm_combine_mode;
  decode_mode_ = true;

  for (int i = 0; i < num_layers; ++i) {
    v_gpu_indices_.push_back(gpu_num);
  }

  T *p_host_tmp;
  for (int i = 0; i < num_layers; ++i) {
    v_p_device_hs_final_target_.push_back(NULL);
    v_p_device_horiz_param_s1_.push_back(NULL);
    v_p_device_horiz_param_s2_.push_back(NULL);
    v_p_device_horiz_bias_.push_back(NULL);
    v_p_device_ct_final_target_.push_back(NULL);
    v_p_device_horiz_param_s1_ct_.push_back(NULL);
    v_p_device_horiz_param_s2_ct_.push_back(NULL);
    v_p_device_tmp_result_.push_back(NULL);
    v_p_device_result_.push_back(NULL);
    v_p_lstm_combiner_layers_.push_back(NULL);
  }

  for (int i = 0; i < num_layers; ++i) {
    cudaSetDevice(v_gpu_indices_[i]);
    FullMatrixSetup(&p_host_tmp, &v_p_device_hs_final_target_[i], lstm_size, minibatch_size_);
    FullMatrixSetup(&p_host_tmp, &v_p_device_horiz_param_s1_[i], lstm_size, lstm_size);
    FullMatrixSetup(&p_host_tmp, &v_p_device_horiz_param_s2_[i], lstm_size, lstm_size);
    FullMatrixSetup(&p_host_tmp, &v_p_device_horiz_bias_[i], lstm_size, 1);
    FullMatrixSetup(&p_host_tmp, &v_p_device_ct_final_target_[i], lstm_size, minibatch_size_);
    FullMatrixSetup(&p_host_tmp, &v_p_device_horiz_param_s1_ct_[i], lstm_size, lstm_size);
    FullMatrixSetup(&p_host_tmp, &v_p_device_horiz_param_s2_ct_[i], lstm_size, lstm_size);

    if (lstm_combine_mode) {
      v_p_lstm_combiner_layers_[i] = new TreeLstm<T>(lstm_size, gpu_num, this);
    }
  }
}


template<typename T>
void AnotherEncoder<T>::LoadWeights(std::ifstream &input_stream) {
  for (int i = 0; i < num_layers_; ++i) {
    cudaSetDevice(v_gpu_indices_[i]);

    if (lstm_combine_mode_) {
      v_p_lstm_combiner_layers_[i]->LoadWeights(input_stream);
    } else {
      ReadMatrixGpu(v_p_device_horiz_param_s1_[i], lstm_size_, lstm_size_, input_stream);
      ReadMatrixGpu(v_p_device_horiz_param_s2_[i], lstm_size_, lstm_size_, input_stream);
      ReadMatrixGpu(v_p_device_horiz_bias_[i], lstm_size_, 1, input_stream);
    }
  }

}

template<typename T>
void AnotherEncoder<T>::ForwardProp() {
  for (int i = 0; i < num_layers_; ++i) {
    T alpha = 1;
    T beta = 0;
    cudaSetDevice(v_gpu_indices_[i]);
    cublasHandle_t tmp_handle;

    T *p_device_h_t_1;
    T *p_device_h_t_2;
    T *p_device_c_t_1;
    T *p_device_c_t_2;

    if (decode_mode_) {
      if (0 == i) {
        tmp_handle = p_model_->input_layer_source_.input_hidden_layer_information_.handle_;
        p_device_h_t_1 = p_model_->input_layer_source_.v_nodes_[0].p_device_h_t_;
        p_device_h_t_2 = p_model_->input_layer_source_bi_.v_nodes_[0].p_device_h_t_;
        p_device_c_t_1 = p_model_->input_layer_source_.v_nodes_[0].p_device_c_t_;
        p_device_c_t_2 = p_model_->input_layer_source_bi_.v_nodes_[0].p_device_c_t_;
      } else {
        tmp_handle = p_model_->v_hidden_layers_source_[i - 1].hidden_hidden_layer_information_.handle_;
        p_device_h_t_1 = p_model_->v_hidden_layers_source_[i - 1].v_nodes_[0].p_device_h_t_;
        p_device_h_t_2 = p_model_->v_hidden_layers_source_bi_[i - 1].v_nodes_[0].p_device_h_t_;
        p_device_c_t_1 = p_model_->v_hidden_layers_source_[i - 1].v_nodes_[0].p_device_c_t_;
        p_device_c_t_2 = p_model_->v_hidden_layers_source_bi_[i - 1].v_nodes_[0].p_device_c_t_;
      }
    } else {
      // training mode
    }

    if (lstm_combine_mode_) {
      cudaMemcpy(v_p_lstm_combiner_layers_[i]->p_device_child_ht_1_, p_device_h_t_1, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
      cudaMemcpy(v_p_lstm_combiner_layers_[i]->p_device_child_ht_2_, p_device_h_t_2, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
      cudaMemcpy(v_p_lstm_combiner_layers_[i]->p_device_child_ct_1_, p_device_c_t_1, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
      cudaMemcpy(v_p_lstm_combiner_layers_[i]->p_device_child_ct_2_, p_device_c_t_2, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
      DeviceSyncAll();
      v_p_lstm_combiner_layers_[i]->Forward();
      DeviceSyncAll();
      cudaMemcpy(v_p_device_hs_final_target_[i], v_p_lstm_combiner_layers_[i]->p_device_h_t_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
      cudaMemcpy(v_p_device_ct_final_target_[i], v_p_lstm_combiner_layers_[i]->p_device_c_t_, lstm_size_ * minibatch_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
      
    }

    DeviceSyncAll();
  }
}




} // end of neural_machine_translation namespace



#endif


