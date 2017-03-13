# *                                * #
# * Author: Qiang Li               * #
# * Email : liqiangneu@gmail.com   * #
# * Date  : 12/16/2016             * #
# * Time  : 17:46                  * #
# *                                * #

SOURCE_CODE_FOLDER = src/
PROGRAM_FOLDER = bin/

CUDA_INCLUDE = /usr/local/cuda-8.0/include/
BOOST_INCLUDE = /usr/local/include/boost/
CUDA_LIB_64 = /usr/local/cuda-8.0/lib64/
BOOST_LIB = /usr/local/lib/
EIGEN = ./eigen/

NVCCFLAGS = -DCUDNN_STATIC -O3 -std=c++11 \
            -g -Xcompiler -fopenmp \
            -I $(CUDA_INCLUDE) \
            -I $(BOOST_INCLUDE) \
            -I $(EIGEN) \
            $(BOOST_LIB)libboost_system.a \
            $(BOOST_LIB)libboost_filesystem.a \
            $(BOOST_LIB)libboost_program_options.a \
            $(CUDA_LIB_64)libcublas_static.a \
            $(CUDA_LIB_64)libcudadevrt.a \
            $(CUDA_LIB_64)libcudart_static.a \
            $(CUDA_LIB_64)libculibos.a \
            $(CUDA_LIB_64)libcurand_static.a \
            $(CUDA_LIB_64)libcusolver_static.a \
            $(CUDA_LIB_64)libcusparse_static.a \
            $(CUDA_LIB_64)libnppc_static.a \
            $(CUDA_LIB_64)libnppi_static.a \
            $(CUDA_LIB_64)libnpps_static.a

SOURCE_CODE = $(SOURCE_CODE_FOLDER)main.cu \
              $(SOURCE_CODE_FOLDER)deep_rnn_kernel.cu \
              $(SOURCE_CODE_FOLDER)dispatcher.cu \
              $(SOURCE_CODE_FOLDER)global_configuration.cc \
              $(SOURCE_CODE_FOLDER)layer_gpu.cc \
              $(SOURCE_CODE_FOLDER)utility_cu.cu \
              $(SOURCE_CODE_FOLDER)utility_cc.cc \
              $(SOURCE_CODE_FOLDER)file_helper.cc \
              $(SOURCE_CODE_FOLDER)input_file_preprocess.cc \
              $(SOURCE_CODE_FOLDER)file_helper_decoder.cc \
              $(SOURCE_CODE_FOLDER)postprocess_unks.cc \
              $(SOURCE_CODE_FOLDER)ibm_bleu_score.cc \
              $(SOURCE_CODE_FOLDER)average_models.cc \
              $(SOURCE_CODE_FOLDER)replace_vocabulary.cc \
              $(SOURCE_CODE_FOLDER)word_embedding.cc \
              $(SOURCE_CODE_FOLDER)byte_pair_encoding.cc 



NiuTrans.NMT:
	nvcc $(NVCCFLAGS) $(SOURCE_CODE) -o $(PROGRAM_FOLDER)$@

clean:
	-rm $(PROGRAM_FOLDER)NiuTrans.NMT
