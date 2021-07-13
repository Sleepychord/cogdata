# Cogdata
## 目录结构
```
.
├── cogdata_workspace
│   ├── cogdata_config.json
│   ├── merged.bin
│   ├── dataset1
│   │   ├── processed.bin
│   │   └── meta_info.json
│   └── dataset2
│       ├── processed.bin
│       └── meta_info.json
├── dataset1
│   ├── cogdata_info.json
│   ├── dataset1.json
│   └── dataset1.rar
└── dataset2
    ├── cogdata_info.json
    ├── dataset2.json
    └── dataset2.zip
```

子目录cogdata_config.json 代表这个子目录有一个数据集。cogdata_workspace 是存放处理过的数据的地方，也代表这里有一个活跃的预处理task。

为了灵活支持多种task，我们抽离了task和process过程，但是为了避免引起混淆，同时只能进行一种task，其信息存在于`cogdata_workspace/cogdata_config.json`。将`cogdata_workspace`目录移走可以通过创建新task。

运行方法是`./cogdata/cli.py`。

需要写的逻辑都已经有描述了。