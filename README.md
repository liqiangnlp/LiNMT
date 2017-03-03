# NMT
NMT developed by Qiang

Email : liqiangneu@gmail.com <br>
Date  : 10/30/2016           <br>
Time  : 13:01                <br>
                     


1. Required

   a) gcc 4.9.3              <br>
   b) boost 1_55_0           <br>
   c) CUDA_8.0 or CUDA_7.5   <br>


2. Install gcc-4.9.3

   first you should download gcc-4.9.3.tar.bz2, then  <br>

   \# yum install libmpc-devel mpfr-devel gmp-devel   <br>
   $ tar xvfj gcc-4.9.3.tar.bz2                       <br>
   $ cd gcc-4.9.3                                     <br>
   $ ./configure --disable-multilib --enable-languages=c,c++       <br>
   $ make -j 4                                        <br>
   \# make install                                    <br>
   $ ln -s /usr/local/lib64/libstdc++.so.6.0.20 path/NiuTrans.NMT // where you want to run NiuTrans.NMT, you should link libstdc++.6.0.20 <br>


3. Install boost_1_55_0

   first you should download boost_1_55_0.tar.gz, then  <br>

   $ rpm -qa boost      // if show something, then use the second command, otherwise use the third command  <br> 
   \# yum remove boost   // uninstall boost, then use 'rpm -qa boost', you will see nothing                 <br>
   $ tar xzvf boost_1_55_0.tar.gz                                                                           <br>
   $ cd boost_1_55_0                                                                                        <br>
   $ ./bootstrap.sh                                                                                         <br>
   $ ./b2                                                                                                   <br>
   $ ./b2 install                                                                                           <br>


4. Install CUDA_8.0 or CUDA_7.5

   $ ./cuda_8.0.44_linux.run --tmpdir=.              // all option is 'yes'          <br>
   $ sudo vi /etc/profile.d/cuda.sh                  // write enviroment variable    <br>
   \# export PATH=$PATH:/usr/local/cuda/bin                                          <br>
   \# export LD_LIBRARY_PATH=$PATH:/usr/local/cuda/lib64                             <br>
   $ source /etc/profile.d/cuda.sh                                                   <br>
   $ nvidia-smi // check                                                             <br>


5. Compile the NiuTrans.NMT

   $ cd NiuTrans.NMT/                                                                <br>
   $ tar xzvf eigen.tar.gz                                                           <br>
   $ make                                                                            <br>
   $ vim ~/.bashrc                                   // add 'export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH' <br>
   $ source ~/.bashrc                                                                <br>


6. Training a NMT model

   $ cd NiuTrans.NMT/scripts/                                                        <br>
   $ perl NiuTrans.NMT.pl -config ../config/NiuTrans.NMT-nmt-training.config         <br>


7. Decoding a file with source sentences

   $ cd NiuTrans.NMT/scripts/                                                        <br>
   $ perl NiuTrans.NMT.pl -config ../config/NiuTrans.NMT-nmt-decoding.config         <br>


   
