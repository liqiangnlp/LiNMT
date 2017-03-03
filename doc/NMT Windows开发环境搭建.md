# NMT Windows开发环境搭建

by 王强

2016/11/1

---

[TOC]

---

# 1. 准备

* OS: Win7 64位
* IDE: VS2013 (不能使用vs2010，由于不完全支持c++11；不能使用vs2015，cuda支持)
* GPU: GeForce 730
* Eigen: 3.2.92 (在`\Eigen\src\Core\util\`中查看)
* Boost: 1.6.0
* CUDA: 8.0
# 2. 安装

## 2.1 Visual Studio 2013

使用镜像``，在u盘里了

## 2.2 Boost

使用安装文件`boost_1_60_0-msvc-12.0-64.exe`（我双击.exe没有反应，是在CMD下执行的.exe才开始安装的）

其中`-12.0`表示`vs2013`用的；`-64`表示`64位`

## 2.3 Eigen

直接解压`eigen.tar.gz`就行

## 2.4 CUDA

使用安装文件``

# 3. 配置Visual Studio 2013

* 新建CUDA项目
  > `新建` -> `项目` -> `NVIDIA` -> `CUDA 8.0`
  > 删除自动生成的文件 `kernel.cu`
* 导入源码
  > `项目属性` -> `添加` -> `现有项`
  >
* 配置
  > `配置管理器` -> `x64`
  > `项目属性` -> `c/c++` -> `常规` -> `附加包含目录` 添加 **D:\boost\boost_1_60_0**(boost安装位置);**$(ProjectDir)\..\package\eigen**(eigen的安装位置)
  >
  > `项目属性` -> `CUDA c/c++` -> `common` -> `Additional Include Directories` 添加 **D:\boost\boost_1_60_0**
  >
  > `项目属性` -> `连接器` -> `常规` -> `附加库目录` 添加 **D:\boost\boost_1_60_0\lib64-msvc-12.0**; **%(CUDA_PATH)\lib\x64**
  >
  > `项目属性` -> `链接器` -> `输入` -> `附加依赖项` 添加 **curand.lib**; **cublas.lib**


##  

