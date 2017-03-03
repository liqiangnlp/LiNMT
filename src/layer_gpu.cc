/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#include "layer_gpu.h"

namespace neural_machine_translation {

void LayerGpuInformation::Init(int device_number) {
  device_number_ = device_number;
  cudaSetDevice(device_number_);
  CublasErrorWrapper(cublasCreate(&handle_), "LayerGpuInformation::Init CUBLAS handler initialization failed\n");

  cudaStreamCreate(&s00_);
  cudaStreamCreate(&s01_);
  cudaStreamCreate(&s02_);
  cudaStreamCreate(&s03_);
  cudaStreamCreate(&s04_);
  cudaStreamCreate(&s05_);
  cudaStreamCreate(&s06_);
  cudaStreamCreate(&s07_);
  cudaStreamCreate(&s08_);
  cudaStreamCreate(&s09_);
  cudaStreamCreate(&s10_);
  cudaStreamCreate(&s11_);
  cudaStreamCreate(&s12_);
  cudaStreamCreate(&s13_);
  cudaStreamCreate(&s14_);
  cudaStreamCreate(&s15_);
  cudaStreamCreate(&s16_);
  cudaStreamCreate(&s17_);
  cudaStreamCreate(&s18_);
  cudaStreamCreate(&s19_);
  cudaStreamCreate(&s20_);
  cudaStreamCreate(&s21_);
  cudaStreamCreate(&s22_);
  cudaStreamCreate(&s23_);
  cudaStreamCreate(&s24_);
  cudaStreamCreate(&s25_);
  cudaStreamCreate(&s26_);
  cudaStreamCreate(&s27_);

  cudaEventCreate(&sparse_forward_start_);
  cudaEventCreate(&i_t_part1_);
  cudaEventCreate(&i_t_full_);
  cudaEventCreate(&f_t_part1_);
  cudaEventCreate(&f_t_full_);
  cudaEventCreate(&c_prime_t_tanh_part1_);
  cudaEventCreate(&c_prime_t_tanh_full_);
  cudaEventCreate(&o_t_part1_);
  cudaEventCreate(&o_t_full_);
  cudaEventCreate(&w_grad_full_done_);
  cudaEventCreate(&error_htild_below_);

  cudaEventCreate(&backprop_init_);
  cudaEventCreate(&error_ot_done_);
  cudaEventCreate(&error_ft_done_);
  cudaEventCreate(&error_tanhcpt_done_);
  cudaEventCreate(&error_it_done_);
  cudaEventCreate(&htm1_p1_done_);
  cudaEventCreate(&htm1_p2_done_);
  cudaEventCreate(&htm1_p3_done_);
  cudaEventCreate(&htm1_p4_done_);

  cudaEventCreate(&w_grad_p1_done_);
  cudaEventCreate(&w_grad_p2_done_);
  cudaEventCreate(&w_grad_p3_done_);
  cudaEventCreate(&w_grad_p4_done_);

  cudaEventCreate(&htm1_done_);
  cudaEventCreate(&htm1_done_tmp_);
  cudaEventCreate(&ctm1_done_);

  cudaEventCreate(&w_hi_grad_done_);
  cudaEventCreate(&w_hf_grad_done_);
  cudaEventCreate(&w_ho_grad_done_);
  cudaEventCreate(&w_hc_grad_done_);

  cudaEventCreate(&m_i_grad_done_);
  cudaEventCreate(&m_f_grad_done_);
  cudaEventCreate(&m_o_grad_done_);
  cudaEventCreate(&m_c_grad_done_);

  cudaEventCreate(&b_i_grad_done_);
  cudaEventCreate(&b_f_grad_done_);
  cudaEventCreate(&b_o_grad_done_);
  cudaEventCreate(&b_c_grad_done_);

  cudaEventCreate(&h_t_below_transfer_);
  cudaEventCreate(&b_c_grad_done_);
  cudaEventCreate(&dropout_done_);
  cudaEventCreate(&d_error_ht_done_);
  cudaEventCreate(&attention_forward_);
  cudaSetDevice(0);
}


// CHECK: OK //
void SoftmaxLayerGpuInformation::Init(int device_number) {
  device_number_ = device_number;
  cudaSetDevice(device_number_);
  CublasErrorWrapper(cublasCreate(&handle_), "CUBLAS handler initialization failed\n");

  cudaStreamCreate(&s00_);
  cudaStreamCreate(&s01_);
  cudaStreamCreate(&s02_);
  cudaStreamCreate(&s03_);

  cudaEventCreate(&output_dist_done_);
  cudaEventCreate(&d_error_ht_done_);
  cudaEventCreate(&d_d_grad_done_);
  cudaEventCreate(&d_b_d_grad_done_);

  cudaSetDevice(0);
}


// CHECK: OK //
void AttentionLayerGpuInformation::Init(int device_number, int d) {

#ifdef DEBUG_CHECKPOINT_3
    std::cerr<<"\n************CP3 In *AttentionLayerGpuInformation* *Init*\n"
             <<"   device_number: "<<device_number<<"\n"
             <<"               d: "<<d<<"\n"
             <<std::flush;
#endif

  device_number_ = device_number;
  cudaSetDevice(device_number);
  cudaStreamCreate(&s00_);

  cudaEventCreate(&start_forward_);
  cudaEventCreate(&start_backward_);
  cudaEventCreate(&forward_prop_done_);
  cudaEventCreate(&backward_prop_done_);
}


}




