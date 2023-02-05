# alchemy

希望通过模块化设计将项目编写简化为编写模块+修改配置文件。

[Wiki](https://github.com/XiPotatonium/alchemy/wiki)

## Related repos

* [alchemy-web](https://github.com/XiPotatonium/alchemy-web) 是alchemy前端框架项目
* [alchemy-ner](https://github.com/XiPotatonium/alchemy-ner) 是alchemy的一个样例项目，实现了命名实体识别task并且实现了若干baseline

## TODO

- [ ] （低优先级）断点续训练。alchemy的断点续训练其实可以有一些假设，例如从某个record dir resume
- [ ] web server.能不能提供链接的功能，可以在网页上填实验结果的表格，然后将结果链接到某个日志文件夹。是不是其实需要一个.alchemy的文件夹来提供一个全局的信息，包括配置文件，以及实验表格之类的东西，也就是说需要有一个数据库。可以有一个mailbox，跑的实验如果完成了可以有一个提示，如果出错了，可以不可以捕获到异常并提示？（捕获异常似乎是一个比较侵入式的改动，要谨慎考虑，或者修改Runner return的逻辑，默认是捕获异常的，并且将异常塞到Result里面去，这也是具有合理性的，毕竟持续跑实验的时候报错信息很难看到，也并不希望报错干扰其他部分）。

## Known Issues

- [ ] (**重要**) `pytorch 1.11.0 py3.9_cuda11.3_cudnn8.2.0_0`可复现性存在问题。但是`py3.8_cuda11.1_cudnn8.0.5_0`就不存在这个问题。我发现InitEval没有问题，我是把数据集的shuffle关掉的，这说明模型的初始化以及forward是没有问题的。tagging模型是可复现的，说明框架上是没有问题的。甚至`use_determininstic_algorithms`检查是可以通过的，所以就很迷惑了。可能是`TrfEncoder`存在一些问题，如果设置`use_lstm=true`就不可复现了(似乎单层LSTM也不会出错)，在`BiPyramid`和`PnRNet`上都是这个样子，`BertBPETagging`因为是没有`TrfEncoder`所以是可复现的
- [x] `pytorch1.8`的`AdamW`为什么`betas`会有问题？只有PnR会有问题，别的模型可以用`AdamW` [github issue](https://github.com/pytorch/pytorch/issues/53354)

## 依赖

conda:

* rich
* loguru
* typer
* tomlkit
* pytorch
* numpy
* tensorboard (optional)

pip:

* fastapi (optional)
* uvicorn (optional)
* pynvml (如果用户自定义entry且不使用官方的alloc_cuda，那么不是必须的)
