# Streaming Dataloader API
- [Streaming Dataloader API](#streaming-dataloader-api)
  - [设计动机和理念](#设计动机和理念)
  - [函数接口](#函数接口)
  - [创建配置文件](#创建配置文件)
  - [支持的数据集类型](#支持的数据集类型)
  - [数据输出格式](#数据输出格式)
  - [状态保存和恢复](#状态保存和恢复)
  - [随机性的处理](#随机性的处理)
  - [变数据并行规模加载](#变数据并行规模加载)
  - [数据集长度统计和建议](#数据集长度统计和建议)

## 设计动机和理念
对于多模态异构的数据读取，存在很多痛点：
1. 对于图像等非定长数据，通常采取连续存储，流式读取的方法，例如[webdataset](https://github.com/webdataset/webdataset)；同时对于文本、或者SFT阶段需要经常可视化和编辑的小数据，通常使用随机读取（`__getitem__`）的方法。
2. 不同数据集来源复杂，需要配置文件方便管理和**设定比例**。
3. 流式、随机读取均需要引入随机性；且需要恢复进度。
4. 需要对不同数据集灵活设置处理函数，并且在配置文件中可以更改，部分处理参数，例如序列长度、图像大小等，在运行时传入才知道，不能在配置文件中写死。

Streaming API采取yaml格式来管理配置不同数据集，使用稍加修改的Omegaconf读取和初始化对象。提供`MergedDataset`类来融合不同类型的数据，并产生统一的读取、加载、切分等接口。

## 函数接口
```python
from cogdata.streaming import instantiate_from_yaml, to_state, mixed_collate
```
本模块提供了几个关键函数，用于处理配置文件和数据处理：

* `instantiate_from_yaml(config_path: str, variables: dict = {}) -> Any:`
此函数用于从YAML配置文件加载并实例化对象。它接受一个配置文件路径和一个可选的变量字典，这些变量将在配置文件中被解析。返回的是根据YAML文件中定义的指令创建的Python对象。详见[创建配置文件](#创建配置文件)章节。

* `to_state(obj: Any) -> dict:`
将给定的Python对象转换为状态字典，便于保存当前迭代状态。详见[状态保存和恢复](#状态保存和恢复)章节。

* `mixed_collate(batch: list) -> Any:`
用于Dataloader混合成batch数据，得到[数据输出格式](#数据输出格式)中的返回格式。

## 创建配置文件

配置文件使用yaml来管理，通过`instantiate_from_yaml`来读取，需要注意的是：
1. 如果某个yaml的字典对象包含`target`和`params`两个键，将会在最终被转化为一个`target`对应的类的对象，即`target(**params)`. 其他的键值对将变成这个对象的属性。`target`可以为任何能import到的类路径，例如`cogdata.streaming.MetaDistributedWebDataset`.
2. 如果某个值为`${variables:vname}`，则它会在最终被使用时解析为`instantiate_from_yaml(config_path, variables)`函数调用的`variables`字典中对应的值。这个功能是为了处理读数据时依赖训练超参的情况，例如图像大小和序列长度。
3. 如果某个yaml的字典对象包含`include`键，则值为一个yaml路径。最终会将该yaml instantiate的结果作为此处的对象，并添加/覆盖其他的键值对。这个一般用在混合多个数据集的时候。

## 支持的数据集类型

1. 任何Pytorch的Dataset的子类，支持`__len__()`和`__getitem__()`.
2. `cogdata.streaming.MetaDistributedWebDataset`. webdataset加强版，每个tar带有一个同名的jsonl文件记录附属信息的格式。
3. `cogdata.streaming.JsonlIterableDataset`. 类似webdataset，但是每次读一行jsonl。
4. `cogdata.streaming.MergedDataset`. 融合多个数据集的`IterableDataset`，支持嵌套和按权重采样，也是本库唯一支持的顶层数据集（即使只使用一个数据集，也要包裹一层MergedDataset）。

一个混合多种不同类型数据的[配置样例](/tests/streaming/merge_testcase.yaml)。


## 数据输出格式

`MergedDataset`返回的数据，期望的每个batch的输出格式一个字典列表，每个字典是一个样本。
例如：
```python
[
    # for iterable webdataset
    {
        'img': tensor1, 
        'caption': str1, 
        ..., 
        '__datasetname__': 'my_img_txt_pair_dataset',
        '__dprank__': 0,
        '__workerid__': 1,
        '__seed__', 1234,
        '__url__': '/path/t001.tar',
        '__key__': '0000001'
    }, 
    # for itemized Pytorch Dataset
    {
        'tokens': tensor1,
        'position_ids': tensor2,
        ...,
        '__datasetname__': 'my_txt_binary_dataset',
        '__dprank__': 0,
        '__workerid__': 1,
        '__seed__', 1234,
        '__index__': 55
    },
    ...
]
```
后续读取之后在机器学习程序中灵活处理。


## 状态保存和恢复
可以使用一个dict来更新训练过数据中记录的状态来持续跟踪数据读取状态。
`to_state(sample)`函数会返回一个字典，用来更新状态。
```python
dataloader_states = {}
iterator = iter(dataset)
sample = next(iterator)
dataloader_states.update(to_state(sample))
# ...
# save & reload dataloader_states
iterator = dataset.__iter_from__(dataloader_states)
# iterator will be reloaded to the previous progress
```
* 注意对于状态，每个数据并行rank和Dataloader worker_id，都会变成不同的key分开存储。因此reload的时候不能改变dp_size 和num_workers。

* 当改变序列长度时（这意味着对于流式数据，特别是有在线随机的情况，分sample的情况会被改变；对于itemize数据暂时不支持），对于__iter_from__方法，需要传入`reload_from_url_level=True`从而在url级别重新加载。

* 重新加载后，对于每个数据集的进度可以保持，但是初始sample各个数据集的顺序可能会发生变化。

## 随机性的处理
在`MergedDataset`的构造函数中，有`shuffle_unstreaming_subsets`和`shuffle_unstreaming_subsets`选项。
* `shuffle_unstreaming_subsets` 对于存在`__getitem__`的数据集的shuffle方法，目前是每2000000个都进行某种相同的permute。
* `shuffle_streaming_subsets` 对于`IterableDataset`的每个url内部储存buffer进行online shuffling。
* 选择数据集的时候，根据percent作为权重，每次吐最低累积权重的数据集，保证均匀性。
* 每个epoch后刷新该数据集的random seed。

## 变数据并行规模加载
TODO

## 数据集长度统计和建议
TODO
