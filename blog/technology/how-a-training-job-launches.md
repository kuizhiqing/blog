# 大模型任务是怎样启动的？

## 背景

本文的讨论范围在以 GPU 加速卡为硬件基础，以 Allreduce 为代表的集合通信方式为主要数据交换方式的大规模集群分布式训练场景。

问题的起点，我们有多台能够互联互通的机器需要能够共同计算一个任务，这些机器上当前主流机型配备了 8 个 GPU 卡（图示以 4 卡为例）。

![image](/assets/gpu-nodes.svg)

首先，机器会通过如 kubernetes 等方式组成集群，机器的设备包括 CPU、内存、GPU 和网络等都会被抽象成资源进行管理，集群管理的资源可以统一进行分配和使用。补充一点，这里的资源不仅包括物理资源，例如端口号甚至自定义的概念也是可以被当作资源的。

其次，机器间的网络设备主流有以太网络和 RDMA 网络，且网络拓扑复杂，程序层使用 NCCL 作为通信库实现 GPU 间的数据交换。

## WHAT ?

当用户需要执行任务即跑程序时，会根据需要分配对应的资源并使用容器的方式绑定资源并使用。容器化的概念这里不再展开，简单的说，已知操作系统是管理物理资源的媒介，容器化就是在操作系统内进行资源隔离和管理使用的逻辑媒介。

Kubernetes 中资源分配和使用的单元叫做 Pod, 一个 Pod 可以包含多个容器，简化理解，也是主流使用场景一个 Pod 对应一个容器即可，所以本文后容器无特殊说明等价于 Pod。以 GPU 资源为例，一台机器上可以启动一个容器分配 8 个 GPU，或者也可以启动 8 个容器每个分配 1 个 GPU 资源。实际情况看，一台机器使用一个容器能获得更好的性能。所以大规模的任务需要分配多机的资源，典型的方式是在每台机器上启动 1 个容器可以使用本机所有 GPU 卡，多个容器组合启动分布式任务。

区别于 PS 模式，当前大模型训练主要以集合通信为主，这种架构以 single program multi run 的方式让各个参与的节点几乎是“对等”的方式参与到计算和通信当中。大模型训练的 3D 并行或更复杂的策略让情况变得更复杂一些，但基本的概念不变，需要考虑局部和整体的关系。

然而，今天讨论的 GPU 训练场景，参与训练的最小单位并不是机器，而是 GPU，当前主流的框架包括 PyTorch、Tensorflow、PaddlePaddle 在启动分布式任务的时候都会启动和 GPU 相同数量的进程与 GPU 设备一对一绑定，后文把这个进程称为 GPU 进程。

![image](/assets/gpu-process.svg)

处在不同网络拓扑节点的两个 GPU 之间进行数据交换的效率是不一样的，最简略的例子比如机内 GPU 的通信效率会比机间高很多，所以在设计并行策略的时候会使用机内 GPU 组成模型并行而机间 GPU 组成数据并行。实现这些策略所依赖的是 GPU 也即对应的进程被分配序号 rank，包括总体的序号 global rank 和机内序号 local rank. 基于 rank，GPU 会被分成不同的分组 group 以进行不同阶段的数据交换。不同的 group 会建立不同的通信域，同一个 GPU 会被分进多个不同的 group，所以一个 GPU 进程会持有多个通信域。

每一个 GPU 进程根据分配的 rank 调用设备的 CUDA 相关接口进行计算操作，并使用 NCCL 在通信域中传递数据。所以，GPU 进程启动时需要知道所有 GPU 进程的信息其中包含了自己的相对位置。

## HOW ?

下面来看这些和 GPU 设备一一对应的进程是如何启动的呢？

主流的有两种启动模式：这里总结为远程命令分发模式和主节点汇合模式。

远程命令分发模式是前容器时代的产物，理论上不需要将机器组成 Kubernetes 集群运行。它依赖节点上都启动 sshd 服务，且已通过 ssh 免密互联互通，主节点按照节点信息配置（比如 mpi 的hostfile）通过 ssh 远程启动进程的方式启动 GPU 进程。主节点的在远程启动进程的时候将所有进程的信息包括 rank 信息通过环境变量传递给了 GPU 进程。

![image](/assets/mpirun.svg)

以 GPU 节点的视角，主进程是 sshd 服务进程，GPU 进程是它的子进程。

远程命令分发模式在大模型时代以 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 为代表，在主节点上执行启动命令，依赖 mpirun/pdsh/pssh 等工具，远程在所有节点上启动与 GPU 卡数相同的进程。

> 主节点不一定是参与计算的工作节点，如果主节点同时也是工作节点，启动原理同样是通过 ssh “远程”登录的方式，只不过客户端和服务端在同一个环境而已。

在 Kubernetes 集群中使用远程命令分发模式时，在任务容器启动时，需要在容器中启动 sshd 服务，且在容器之间配置 ssh 免密互信，并且需要收集所有容器的 IP 或域名信息在命令下发时使用，实现或使用方式可以参考 [mpi-operator](https://github.com/kubeflow/mpi-operator).

在 Kubernetes 集群中启动分布式任务，分配资源然后在多个节点上启动容器时可以配置具体的启动命令，所以以 PyTorch 原生 [torchrun](https://pytorch.org/docs/stable/elastic/run.html) 为代表的主节点汇合模式是更加云原生的方式。这种方式在启动时，每个节点的启动进程叫做 agent，主节点的 agent 启动额外服务（如 TCPStore）供所有节点汇聚信息，所有节点的 agent 启动后通过和主节点建立连接上报自己的信息并获取所有节点信息，然后根据当前节点的相对位置和 GPU 数量启动对应的 GPU 进程并配置不同的环境变量。从而每个 GPU 进程可以获取到所有 GPU 进程的信息和自己的 rank 的信息。

![image](/assets/master-gather.svg)

以 GPU 节点的视角，主进程是 agent 服务进程，GPU 进程是它的子进程。

主节点汇合模式在大模型时代以 [Megatron](https://github.com/NVIDIA/Megatron-LM) 为代表，如 [gpt3 example](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/gpt3/train_gpt3_175b_distributed.sh) 默认使用原生  torchrun 方式启动。

## WHY ?

 从抽象意义上说，命令分模式是信息的 [broadcast](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/)，而主节点汇合模式是信息的 [gather](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/)。在分布式场景下这两种模式可以说无处不在。

![image](/assets/broadcast-process.svg)

![image](/assets/gather-process.svg)

 > 不同于 scatter，这里 broadcast 的信息一般是“全局”信息；信息聚合的阶段算 gather，加上后面完整的信息同步流程可以看作是 allgather 的过程。

在机器组建成 Kuberntes 集群时，装有 kubelet 的机器通过 join 的方式向 api-server 注册，使用的是向主节点汇合的 gather 模式。在任务提交后，各个节点从 api-server 获取到任务执行的信息，使用的是 broadcast 信息分发的模式，稍有不同的是这里的 broadcast 实现方式是 pull 模式而非 push 模式。

在 NCCL 初始化的过程中，首先 rank 为 0 的进程生成 nccl unique id （`ncclUniqueId` 的本质是 `socketAddress`）是通过 broadcast 的模式将该 unique id 分发到各个节点。各个节点通过该 id 注册自己的信息，使用的本质上是 gather 信息汇合模式，通过汇聚所有节点信息才能建立 full mesh 连接和建立通信域。

> NCCL 同时也提供主节点汇合模式进行初次的信息聚合，但在复杂拓扑网络中使用受限，更多信息参考 NCCL_COMM_ID 和 [issue](https://github.com/NVIDIA/nccl/issues/730)。

所以在不同的抽象层，这两种信息交换模式会被交替使用，当然每一层传递的信息是可能是不一样的。注意到这二者还有相互的依赖关系，类似于先有鸡还是先有蛋的问题：

* 要使用 broadcast 模式分发信息首先需要知道全局的通信信息；
* 要知道全局的通信信息和通过 gather 的模式让各个节点汇聚自己的信息；
* 要让各个节点上报自己的信息可以使用 broadcast 模式向节点分发信息。

而破解这个依赖的关键就是使用上一个抽象层传递的信息，例如 NCCL 初始化过程中 broadcast  ncclUniqueId 往往使用上层的通信链路如 mpi 或者 gloo 等。

本文就介绍到这里，更多信息欢迎留言讨论。

下一篇讲一讲超大规模的模型稳定运行需要突破的一些技术难点，敬请期待。

# References
* https://github.com/microsoft/DeepSpeed
* https://github.com/kubeflow/mpi-operator
* https://github.com/NVIDIA/Megatron-LM
* https://github.com/NVIDIA/nccl
* https://pytorch.org/docs/stable/elastic/run.html
* https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
