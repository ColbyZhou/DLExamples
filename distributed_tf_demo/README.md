# 训练方式对比

## 1. 分布式架构模式：

- 1.1 In-Graph是Worker(相当于slave) join等待ParameterServer(相当于master)下发的计算任务，类似单机多卡情况下，将任务分发给多个GPU计算，最终由ParameterServer收集并更新参数。
优点是配置简单，worker只要join等待计算任务就好，使用起来像是本机的多个GPU，通过tf.device("/job:worker/task:n")即可定位使用。
缺点是数据分发都在一个节点上，压力过大，影响训练速度。

- 1.2 Between-Graph是ParameterServer join等待接受来自Worker的参数更新。此时，每一个worker是一个独立的计算节点，有各自的Graph，根据自己的数据计算完梯度之后，将其发送给ParameterServer。
优点是不用分发数据，worker有自己的数据和Graph，训练速度快。
缺点是配置复杂。


## 2. 参数更新方式：

- 2.2 Synchronous(同步)是指更新参数时，等待所有的worker都计算完了一步(step)之后更新参数，然后同步发送新参数给所有worker。
优点是loss稳定，缺点是同步需要等待，训练慢，适合数据不大、各个计算节点性能均衡的情况。

- 2.3 Asynchronous(异步)是指更新参数时，各个worker计算完之后直接发送给ParameterServer直接更新，不用等待其他的worker完成当前这一步(step)
优点是训练速度快，缺点是loss不稳定，适合数据大、各个计算节点性能差异大的情况。

