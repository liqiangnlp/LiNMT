# NiuTrans.NMT Linux部署

by 王强

2016/11/01

---

[TOC]

---

# 1.准备

* OS: CentOS 7.2
* GCC: 4.9.3
* Eigen: 3.2.92 (在`\Eigen\src\Core\util\`中查看)
* Boost:1.55
* Cuda: 8.0
# 2.安装

## 2.1 FTP服务器

* FTP客户端无法连接，**防火墙问题**

  ```shell
  （下列均需root）
  firewall-cmd --permanent --zone=public --add-service=ftp
  firewall-cmd --reload
  service vsftpd restart
  ```

* FTP能连接，但无法上传文件，**selinux问题**

  ```shell
  setsebool -P allow_ftpd_full_access 1
  service vsftpd restart
  ```

---

## 2.2 安装依赖项

使用yum安装依赖项（此步不做会影响安装gcc），需root

```shell
yum install libmpc-devel mpfr-devel gmp-devel
```

提示错误：`Could not resolve host: mirrorlist.centos.org;`

**DNS设置问题**

```shell
vim /etc/resolv.conf
写入:
search 38forsbm
nameserver 8.8.8.8
nameserver 218.85.152.99
```

---

## 2.3 安装gcc

在`package/`目录下，执行

```shell
tar xvfj gcc-4.9.3.tar.bz2
cd gcc-4.9.3
./configure --disable-multilib --enable-languages=c,c++
make -j8 //需要等一会
make install //需root
```

---

## 2.4 安装boost

在`package/`目录下，执行

```shell
rpm -qa boost //检查是否已安装boost
yum remove boost //如果已安装boost，执行此步骤，卸载boost，需root
tar xzvf boost_1_55_0.tar.gz
cd boost_1_55_0
./bootstrap.sh
./b2 //需要等一会
./b2 install //需root
```

---

## 2.5 安装eigen

在`package/`目录下，执行

```shell
tar zxvf eigen.tar.gz
```

---

## 2.6 安装cuda

在`package/`目录下，校验安装文件是否损坏

```shell
md5sum cuda_8.0.44_linux.run
```

正确的md5值为`6dca912f9b7e2b7569b0074a41713640`

用本地runfile文件安装cuda

```shell
./cuda_8.0.44_linux.run --tmpdir=.  //--tmpdir不确定是否可以不加
```

> Do you accept the previously read EULA?
> accept/decline/quit:            `accept`
>
> Do you want to install the OpenGL libraries?
> (y)es/(n)o/(q)uit [ default is yes ]: `回车`，默认是yes
>
> Do you want to run nvidia-xconfig?
> `回车`，默认是no
>
> Install the CUDA 8.0 Toolkit?
> (y)es/(n)o/(q)uit: `y`
>
> Enter Toolkit Location
> \[ default is /usr/local/cuda-8.0 ]: `回车`
>
> Do you want to install a symbolic link at /usr/local/cuda?
> (y)es/(n)o/(q)uit: `y`
>
> Install the CUDA 8.0 Samples?
> (y)es/(n)o/(q)uit: `y`
>
> Enter CUDA Samples Location
>
> \[ default is /root ]:`回车` 

所有的选项能`默认`就默认，不能默认就选择`yes`

## 2.7 配置环境变量

只配置当前用户的环境变量，编辑`~/.bashrc`，在结尾追加

```shell
# 加入cuda相关的可执行程序，如nvcc
export PATH=$PATH:/usr/local/cuda/bin 
# 加入cuda相关的动态库; libstdc++.so的动态库
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib64
```

刷新环境变量

```shell
source ~/.bashrc
```

---

# 3. 编译

如果全是默认的操作，不需要编辑Makefile，否则编辑Makefile

```shell
CUDA_INCLUDE = /usr/local/cuda-8.0/include/
BOOST_INCLUDE = /usr/local/include/boost/
CUDA_LIB_64 = /usr/local/cuda-8.0/lib64/
BOOST_LIB = /usr/local/lib/
EIGEN = ./package/eigen/
```

build出可执行程序，在`/NiuTrans.NMT/`目录下，执行

```shell
make
```

在`/bin/`目录下生成`NiuTrans.NMT`

---

# 4. 快速验证

@加入一个快速测试是否安装正确的模块@

---

# 5. 训练

## 1. 参数设置

编辑`/config/`目录下的配置文件`NiuTrans.NMT-nmt-training.config`

### 1. 主要路径设置  

|            参数名             |                 描述                 |     类型      |             默认值              |
| :------------------------: | :--------------------------------: | :---------: | :--------------------------: |
|         --training         |      训练数据、输出模型路径(网络结构的参数，词表)       |   string    |              -               |
|        --best-model        |            保存的模型前缀（带路径）            |   string    |  ../work/nmt/nmt-model-best  |
|     --save-all-models      |      是否保存所有在validation上有提高的模型      |  int {0,1}  |              1               |
|           --log            |               log路径                |   string    | ../work/nmt/nmt-training.log |
|     --tmp-dir-location     |         **临时目录（不知道干什么用）**          |   string    |         ../work/nmt/         |
|    --adaptive-halve-lr     |          **校验集路径(参数名。。)**          |   string    |              -               |
| --adaptive-decrease-factor | 学习率衰减系数（校验集cost增加，就衰减学习率，不想衰减就设置1） | float (0,1] |              1               |
|      --fixed-halve-lr      |           前几个epoch不更新学习率           |     int     |              6               |

> `--training`指定的输出模型中，源语词典没有`<EOS>`，目标语词典有这个标记
>
> 更新学习率的策略这里有两种：
>
> 1. adaptive：提供校验集，根据校验集上的cost，自动更新（衰减）学习率
>
> 2. fixed: 不需要校验集，设置前n个epoch不调整学习率，从n+1开始，每半个epoch学习率就减半
>
>    实际使用时，是两者结合，提供校验集（只是看看校验集上的cost变化），但是不让根据校验集自动衰减学习率，所以设置`--adaptive-decrease-factor`为1，学习率的更新还是使用`fixed`模式

### 2. 网络结构设置

|         参数名         |                  描述                  |    类型    | 默认值  |
| :-----------------: | :----------------------------------: | :------: | :--: |
| --target-vocab-size |                目标语词汇数                |   int    | 30K  |
| --source-vocab-size |                源语词汇数                 |   int    | 30K  |
|    --hidden-size    |             **递归隐藏层大小?**             |   int    | 1000 |
|   --layers-number   |                递归网络层数                |   int    |  4   |
|  --attention-mode   | 是否使用attention（使用就只能是local attention） | int{0,1} |  1   |
|  --attention-width  |         local attention的窗口大小         |   int    |  10  |
|    --feed-input     |           是否使用feed-input方法           | int{0,1} |  1   |
|      --dropout      |             keep率（垂直方向）              |  float   | 0.8  |

> 1. embedding的大小是固定的，同`--hidden-szie`一样，不能改变

### 3. 训练过程设置

|          参数名           |                描述                |    类型    |    默认值     |
| :--------------------: | :------------------------------: | :------: | :--------: |
|    --minibatch-size    |           mini-batch大小           |   int    |     64     |
|    --learning-rate     |              学习率初值               |  float   |    0.7     |
|   --parameter-range    |           Uniform分布的范围           |  float   | -0.08 0.08 |
| --whole-clip-gradients |       gradient clipping阈值        |  float   |     5      |
|       --shuffle        |     是否打乱数据(每一个epoch都shuffle)     | int{0,1} |     1      |
|     --longest-sent     |           训练中允许的最大句子长度           |   int    |    100     |
|    --number-epochs     |            最大的epoch数             |   int    |     10     |
|      --multi-gpu       | 分配gpu(rnn层数+1层softmax，gpu从0开始编号) |   int    | 0 0 0 0 0  |
|  --screen-print-rate   |               打屏频率               |   int    |     50     |

> 本程序只能运行在GPU环境下，支持`单机单卡` 、 `单机多卡`
>
> `--multi-gpu`需要设置为一组整数，需要的整数数量是`--layer_num`+1（输出层的softmax），每一个整数值表示分配的gpu编号，gpu从0开始编号
>
> 假设`--layer_num`=4
>
> `单机单卡`时，`--multi-gpu`设置为：`0 0 0 0 0`，表示一共给5个层分配gpu，每个层都使用的是编号为0的gpu
>
> `单机多卡`时，假设有4个gpu，`--multi-gpu`可以设置为：`0 1 2 3 3`，表示RNN的第一层到第四层分别分配给gpu0、gpu1、gpu2、gpu3，输出层分配给gpu3，这个设置是随意的

## 2. 启动训练

在`/script/`目录下，执行

```shell
perl NiuTrans.NMT.pl -config ../config/NiuTrans.NMT-nmt-training.config
```

@一些开始训练前的检测没有做，比如文件是否存在，需要更友好的提示信息@

## 3. Finetuning



---

# 6. 解码

## 1. 参数设置

编辑`/config/`目录下的配置文件`NiuTrans.NMT-nmt-decoding.config`

|           参数名            |            描述             |     类型     |             默认值              |
| :----------------------: | :-----------------------: | :--------: | :--------------------------: |
|        --decoding        |    设置nbest、模型路径、翻译结果文件    | int+string |           nbest=1            |
| --decode-main-data-files |          待翻译的文件           |   string   |              -               |
|          --log           |           日志文件            |   string   | ../work/nmt/nmt-decoding.log |
|       --unk-decode       | 记录每一个译文的unk所对应的源语位置（从0计数） |   string   |  ../work/nmt/1best-unks.txt  |
|       --beam-size        |          beam大小           |    int     |              12              |
|        --penalty         |           ??不知道           |            |              0               |
|     --decoding-ratio     |      限制目标语长度需要满足的范围       |   float    |           1.0 2.0            |
|      --longest-sent      |          最大产生的词数          |    int     |             200              |
|    --tmp-dir-location    |        临时文件目录(作用)         |   string   |         ../work/nmt          |
|      --print-score       |       是否输出分数，译文的概率？       |  int{0,1}  |              0               |

## 2. 启动解码

在`/script/`目录下，执行

```shell
perl NiuTrans.NMT.pl -config ../config/NiuTrans.NMT-nmt-decoding.config
```

@一些开始翻译前的检测没有做，比如文件是否存在，需要更友好的提示信息@

---

# 7. 强制解码

---