# NMT
NMT developed by Qiang


\# *                              
\# * Author: Qiang Li             
\# * Email : liqiangneu@gmail.com 
\# * Date  : 10/30/2016           
\# * Time  : 13:01                
\# *                              


1. Required
   a) gcc 4.9.3
   b) boost 1_55_0
   c) CUDA_8.0 or CUDA_7.5


2. Install gcc-4.9.3

   first you should download gcc-4.9.3.tar.bz2, then

   \# yum install libmpc-devel mpfr-devel gmp-devel
   $ tar xvfj gcc-4.9.3.tar.bz2
   $ cd gcc-4.9.3
   $ ./configure --disable-multilib --enable-languages=c,c++
   $ make -j 4
   \# make install
   $ ln -s /usr/local/lib64/libstdc++.so.6.0.20 path/NiuTrans.NMT // where you want to run NiuTrans.NMT, you should link libstdc++.6.0.20


3. Install boost_1_55_0

   first you should download boost_1_55_0.tar.gz, then

   $ rpm -qa boost      // if show something, then use the second command, otherwise use the third command 
   \# yum remove boost   // uninstall boost, then use 'rpm -qa boost', you will see nothing
   $ tar xzvf boost_1_55_0.tar.gz 
   $ cd boost_1_55_0
   $ ./bootstrap.sh
   $ ./b2
   $ ./b2 install


4. Install CUDA_8.0 or CUDA_7.5

   $ ./cuda_8.0.44_linux.run --tmpdir=.              // all option is 'yes'
   $ sudo vi /etc/profile.d/cuda.sh                  // write enviroment variable 
   \# export PATH=$PATH:/usr/local/cuda/bin 
   \# export LD_LIBRARY_PATH=$PATH:/usr/local/cuda/lib64 
   $ source /etc/profile.d/cuda.sh
   $ nvidia-smi // check


5. Compile the NiuTrans.NMT
   $ cd NiuTrans.NMT/
   $ tar xzvf eigen.tar.gz
   $ make
   $ vim ~/.bashrc                                   // add 'export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH'
   $ source ~/.bashrc


6. Training a NMT model
   $ cd NiuTrans.NMT/scripts/
   $ perl NiuTrans.NMT.pl -config ../config/NiuTrans.NMT-nmt-training.config


7. Decoding a file with source sentences
   $ cd NiuTrans.NMT/scripts/
   $ perl NiuTrans.NMT.pl -config ../config/NiuTrans.NMT-nmt-decoding.config


   
