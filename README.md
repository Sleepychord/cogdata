# Cogdata

## Install
```
pip install cogdata
sudo `which install_unrarlib.sh`
```
## Directory Structure
```
.
├── cogdata_task_task1
│   ├── cogdata_config.json (indicating a task path)
│   ├── merged.bin
│   ├── dataset1
│   │   ├── dataset1.bin
│   │   └── meta_info.json
│   └── dataset2
│       ├── dataset2.bin
│       └── meta_info.json
├── dataset1
│   ├── cogdata_info.json (indicating a dataset path)
│   ├── dataset1.json
│   └── dataset1.rar
└── dataset2
    ├── cogdata_info.json
    ├── dataset2.json
    └── dataset2.zip
```

## Pipeline
The motivation of this project is to provide lightweight APIs for large-scale NN-based data-processing, e.g. ImageTokenization. The abstraction has 3 parts:
* **Dataset**: Raw dataset from other organization in various formats, e.g. rar, zip, etc. The information are recorded at `cogdata_info.json` in its split folder. 
* **Task**: A task is a collection of "configs, results for different datsets, logs, merged results, and evenly split results". The config of a task are recorded in `cogdata_info.json`. The states (processed, hanging/running, unprocessed)of a dataset in this tasks are in `meta_info.json`.
* **DataSaver**: The format of saved results. The first option is our `BinSaver`, which saves plain bytes with fixed length. It can be read or memmap very fast. The config of DataSaver are also with the task in `cogdata_info.json`. 

### Commands
```
cogdata create_dataset  [-h] [--description DESCRIPTION] --data_files DATA_FILES [DATA_FILES ...] --data_format DATA_FORMAT [--text_files TEXT_FILES [TEXT_FILES ...]] [--text_format TEXT_FORMAT] name
```
Alias: `cogdata data ...`. `data_format` is chosen from class names in cogdata.datasets, e.g. `StreamingRarDataset`. Texts related options are optional for text-image datasets.

```
cogdata create_task [-h] [--description DESCRIPTION] --task_type TASK_TYPE --saver_type SAVER_TYPE [--length_per_sample LENGTH_PER_SAMPLE] [--img_sizes IMG_SIZES [IMG_SIZES ...]] [--txt_len TXT_LEN]
                           [--dtype {int32,int64,float32,uint8,bool}] --model_path MODEL_PATH
                           task_id
```
Alias: `cogdata task ...`. `task_type` and `saver_type` is chosen from class names in cogdata, e.g. `ImageTextTokenizationTask` or `BinarySaver`.
```
cogdata process [-h] --task_id TASK_ID [--nproc NPROC] [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                       [--batch_size BATCH_SIZE] [--ratio RATIO]
                       [datasets [datasets ...]]
```
The i-th proc will be binded to the i-th GPU.

```
cogdata merge [-h] --task_id TASK_ID
```
Merge all the processed data.

```
cogdata list [-h] [--task_id TASK_ID]
```
List all the current datasets in this folder.
```
cogdata clean [-h] [--task_id TASK_ID]
```
Clean the unfinished states of the task.
### Customized Tasks
Add `--extra_code PATH_TO_CODE` after `cogdata `(e.g., `cogdata --extra_code ../examples/convert2tar_task.py [task or process]` to execute and register your own task before running the command. See `examples/` for details. 

## TODO List

* [ ] 支持多种不同格式文本处理
* [ ] sphinx 注释文档更详细撰写
* [ ] 更精细化的参数管理，将tokenization一般化
* [ ] PPT & 视频介绍
* [ ] Merge 视频处理 [Wenyi]
* [ ] Merge Object detection [Zhuoyi]


