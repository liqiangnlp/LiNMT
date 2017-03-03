/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#ifndef GPU_INFORMATION_H_
#define GPU_INFORMATION_H_

#include "utility_cu.h"
#include "debug.h"

/* cuda */
#include <cublas_v2.h>
#include <cublas_api.h>
#include <driver_types.h>
#include <cuda_runtime.h>


namespace neural_machine_translation {

class LayerGpuInformation {

public:
  int device_number_ = 0;      // Input layer always gets device 0

public:
  cublasHandle_t handle_;

public:
  // streams are shared for forward and back prop
  cudaStream_t s00_, s01_, s02_, s03_, s04_, \
               s05_, s06_, s07_, s08_, s09_, \
               s10_, s11_, s12_, s13_, s14_, \
               s15_, s16_, s17_, s18_, s19_, \
               s20_, s21_, s22_, s23_, s24_, \
               s25_, s26_, s27_;


public:
  // forward prop events
  cudaEvent_t sparse_forward_start_;
  cudaEvent_t i_t_part1_, i_t_full_;                // cudaEvent_t for calculating LstmInputHiddenNode::p_device_i_t_
  cudaEvent_t f_t_part1_, f_t_full_;
  cudaEvent_t c_prime_t_tanh_part1_, \
              c_prime_t_tanh_full_;
  cudaEvent_t o_t_part1_, o_t_full_;


public:
  // backprop events
  cudaEvent_t backprop_init_;
  cudaEvent_t error_ot_done_;
  cudaEvent_t error_ft_done_;
  cudaEvent_t error_tanhcpt_done_;
  cudaEvent_t error_it_done_;

  cudaEvent_t htm1_p1_done_;
  cudaEvent_t htm1_p2_done_;
  cudaEvent_t htm1_p3_done_;
  cudaEvent_t htm1_p4_done_;

  cudaEvent_t w_grad_p1_done_;
  cudaEvent_t w_grad_p2_done_;
  cudaEvent_t w_grad_p3_done_;
  cudaEvent_t w_grad_p4_done_;

  cudaEvent_t attention_forward_;    // this is gotten from the attention layer if feed input is true
  cudaEvent_t error_htild_below_;    // this is created here and shared with the attention layer

  // These are for synchronization for the backprop
  cudaEvent_t htm1_done_;
  cudaEvent_t htm1_done_tmp_;
  cudaEvent_t ctm1_done_;

  cudaEvent_t w_grad_full_done_;
  cudaEvent_t w_hi_grad_done_;
  cudaEvent_t w_hf_grad_done_;
  cudaEvent_t w_ho_grad_done_;
  cudaEvent_t w_hc_grad_done_;

  cudaEvent_t m_i_grad_done_;
  cudaEvent_t m_f_grad_done_;
  cudaEvent_t m_o_grad_done_;
  cudaEvent_t m_c_grad_done_;
  
  cudaEvent_t b_i_grad_done_;
  cudaEvent_t b_f_grad_done_;
  cudaEvent_t b_o_grad_done_;
  cudaEvent_t b_c_grad_done_;

  cudaEvent_t h_t_below_transfer_;     // SendHTAbove, transfer h_t to upper layer
  cudaEvent_t dropout_done_;

  cudaEvent_t d_error_ht_done_;
  

public:
  void Init(int device_number);
  
};


class SoftmaxLayerGpuInformation {

public:
  int device_number_ = 0;    // this is for single GPU at the moment

public:
  cublasHandle_t handle_;

public:
  cudaStream_t s00_, s01_, s02_, s03_;

public:
  cudaEvent_t output_dist_done_;
  cudaEvent_t d_error_ht_done_;
  cudaEvent_t d_b_d_grad_done_;
  cudaEvent_t d_d_grad_done_;

public:
  void Init(int device_number);

};


class AttentionLayerGpuInformation {

public:
  int device_number_ = 0;

public:
  cudaStream_t s00_;

public:
  cudaEvent_t start_forward_;
  cudaEvent_t start_backward_;

  cudaEvent_t forward_prop_done_;
  cudaEvent_t backward_prop_done_;

  cudaEvent_t error_htild_below_;          // this is created here and shared with the attention layer

public:
  void Init(int device_number, int d);
};





}

#endif