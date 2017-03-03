# NMT Server  Linux部署 

by 王强

2016.11.30

---

[TOC]

## 单机单节点

### 1. 配置环境变量

* 在`PN/scripts/`下执行

```shell
./Install.sh
```

* 在`PD/scirpts/`下执行

```shell
./Install.sh
```

*注意`LD_LIBRARY_PATH`覆盖问题*

将`LD_LIBRARY_PATH`写入到了`~/.bashrc`

* 刷新环境变量

执行

```shell
source ~/.bashrc
```

* 检查计算节点

```shell
ldd PN/lib/libDecoder.so
```

如果没有提示`***Not Found`或者`GLIBCXX*** is required`，则表示动态库正常

* 检查调度服务器

```shell
ldd PD/lib/libWordSegmentation.so
```

如果没有提示`***Not Found`或者`GLIBCXX*** is required`，则表示动态库正常

ldd `PN/lib/libDecoder`

### 2. 启动调度服务器

执行

```shell
StartNiuTransServerPD
```

### 3. 启动计算节点

执行

```shell
StartNiuTransServerPN
```

---

## 单机多节点

下面以两节点为例

### 1. 创建多用户

* 新建用户1

```shell
su #需ROOT
adduser nmt_1 #添加节点1
passwd nmt_1 
```

* 新建用户2

步骤同上，假设新建用户名为`nmt_2`

* 放置程序

假设`nmt_1`放置`PD`和`PN`

`nmt_2`放置`PN`

### 2. 配置nmt_1用户

登录`nmt_1`用户

#### 2.1 配置环境变量

同`单机单节点`

#### 2.2 配置调度服务器

编辑`PD/config/ScheduleNode.Conf`

修改

> param="IPaddressNumber"                 value="1"
> param="IpaddressList0"                        value="127.0.0.1:7777"

为

>param="IPaddressNumber"                 value="2"
>param="IpaddressList0"                        value="127.0.0.1:7777"
>param="IpaddressList1"                        value="127.0.0.1:7775"

这里设置第一个计算节点的端口为`7777`（`PN`默认的端口号是`7777`），第二个计算节点的端口为`7775`

如果有更多的节点，仿照着写，保证每一个节点有不同的端口

#### 2.3 配置计算节点

这里不用特殊配置，使用`PN`的默认设置即可。

`PN`默认的端口号是`7777`，解码器参数`--multi-gpu`默认是`0`

### 3. 配置nmt_2用户

登录`nmt_2`用户

#### 3.1 配置计算节点

* 编辑`PN/scripts/Install.sh`

修改第`8`行

>proxyID=$(netstat -tln | grep 7777 | awk '{print $1}')

为

> proxyID=$(netstat -tln | grep 7775 | awk '{print $1}')

因为端口`7777`已经被`nmt_1`占用，`nmt_2`使用端口号`7775`

- 刷新环境变量

执行

```shell
source ~/.bashrc
```

* 编辑`PN/bin/Startup.sh`

修改第`6`行

> NodeID=$(netstat -tln | grep 7777 | awk '{print $1}')

为

> NodeID=$(netstat -tln | grep 7775 | awk '{print $1}')

理由同上

* 编辑`PN/config/ComputeNode.Conf`

修改

> param="ListenPort"              value="7777"

为

> param="ListenPort"              value="7775"

让计算节点监听`7775`端口

#### 3.2 启动计算节点

执行

```shell
StartNiuTransServerPN
```

---

## 其他

### 1. 配置环境变量

执行`PD`和`PN`的`script/Install.sh`

编辑`~/.bashrc` ，追加

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib64
export PATH=$PATH:/usr/local/cuda/bin
```

更新

```shell
source ~/.bashrc
```

### 2. 动态库加载失败

使用`ldd`查看失败原因

#### 2.1 GLIBCXX

* 原因：libstdc++.so的版本不对
* 方法：`ldd`查看libstdc++.so实际使用的是哪个文件；`locate`找到正确版本的文件；删除掉实际使用的那个软连接，将正确版本的文件软连接过去

#### 2.2 Not Found

* 原因：提示'not found'是环境变量配置不对
* 方法：修改`~/.bashrc`
* `export LD_LIBRARY_PATH`=`$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib`

### 3. PD已经启动成功，但是浏览器打不开页面

* 原因：防火墙
* 方法：关闭防火墙

`sudo` `systemctl stop firewalld.service`

