Quick Start
===========
A simple example about usage of ``cogdata``.

Installation
------------
pip::

    pip install cogdata --index-url https://test.pypi.org/simple

Initialization 
--------------
Firstly, build a data folder and move the data files into it::

    .
    └── test_ds
        ├── infolist.json
        └── n10148035.tar


Create Dataset
--------------
Use ``cogdata create_dataset`` commend to create a dataset::

    cogdata create_dataset  [-h], [--help]                              # show the help and exit
                            [--description DESCRIPTION]                 # description of the handling dataset.
                            [--text_files TEXT_FILES [TEXT_FILES ...]]  # file names of the handling dataset.
                            [--text_format TEXT_FORMAT]                 # format of data files.
                            --data_files DATA_FILES [DATA_FILES ...]    # file name of the text.
                            --data_format DATA_FORMAT                   # format of the text file.

                            name                                        # name of the handling dataset.

This commend just creates a ``cogdata_info.json`` in "name" folder. Here let Dataset's name be same with the data folder.

Example::
    
    cogdata create_dataset --text_files infolist.json --text_format dict --data_files n10148035.tar --data_format TarDataset test_ds

Directory structure::

    .
    └── test_ds
        ├── cogdata_info.json
        ├── infolist.json
        └── n10148035.tar

Create Task
-----------
Use ``cogdata create_task`` commend to create a task::

    cogdata create_task [-h], [--help]                              # show the help and exit
                        [--description DESCRIPTION]                 # description of the new task.
                        [--length_per_sample LENGTH_PER_SAMPLE]     # data length of one sample (Bytes).
                        [--img_sizes IMG_SIZES [IMG_SIZES ...]]     # sizes of a pre-tokenized image.
                        [--txt_len TXT_LEN]                         # length of text in one sample.
                        [--dtype {int32,int64,float32,uint8,bool}]  # data type of samples.
                        [--model_path MODEL_PATH]                   # path of image tokeizer
                        --task_type TASK_TYPE                       # type of the handling task.
                        --saver_type SAVER_TYPE                     # saver mode.

                        task_id                                     # id of the new task.

Example::

    # Don't forget to modify "model_path"
    cogdata create_task --description test --task_type ImageTextTokenizationTask --saver_type BinarySaver --length_per_sample 1088 --img_sizes 256 --txt_len 64 --dtype int32 --model_path='/dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt' test_task


Directory structure::

    .
    ├── cogdata_task_test_task
    │   └── cogdata_config.json
    └── test_ds
        ├── cogdata_info.json
        ├── infolist.json
        └── n10148035.tar


Check Datasets and Tasks
-------------------------
Now we can use ``cogdata list`` commend to check::

    cogdata list [-h], [--help]                       # show the help and exit.
                 [-t TASK_ID], [ --task_id TASK_ID]   # id of the handling task.

Example: list dataset::

    cogdata list

Expected Output::

    --------------------------- All Raw Datasets --------------------------    
    test_ds(207.7MB)
    ------------------------------- Summary -------------------------------
    Total 1 datasets
    Total size: 207.7MB

Example: list task::

    cogdata list -t test_task

Expected Output::

    --------------------------- All Raw Datasets --------------------------    
    test_ds(207.7MB)
    ------------------------------- Summary -------------------------------
    Total 1 datasets
    Total size: 207.7MB
    ------------------------------ Task Info ------------------------------
    Task Id: test_task
    Task Type: ImageTextTokenizationTask
    Description: test
    Processed:  FORMAT: dataset_name(raw_size => processed_size)

    Hanging:  FORMAT: dataset_name(raw_size)[create_time]

    Additional:  FORMAT: dataset_name(processed_size)

    Unprocessed:  FORMAT: dataset_name(raw_size)
    test_ds(207.7MB) 

"test_ds" is in Unprocessed group.

Process
-------
Use ``cogdata process`` commend to process datasets::

    cogdata process 
                    [-h], [--help]                                      # show the help and exit
                    [--nproc NPROC]                                     # number of processes to launch.
                    [--dataloader_num_workers DATALOADER_NUM_WORKERS]   # number of processes for dataloader per computational process.
                    [--ratio RATIO]                                     # ratio of data to process.
                    -t TASK_ID, --task_id TASK_ID                       # id of the handling task.

                    [datasets [datasets ...]]                           # dataset names, None means all possible datasets.

Example::

    cogdata process --task_id test_task --nproc 2 --dataloader_num_workers 1 --ratio 1 test_ds

Expected Output::

    All datasets: test_ds
    Processing test_ds
    dataset: test_ds, rank 0:[#########################] 100%  Speed: 92.66 samples/s
    dataset: test_ds, rank 1:[#########################] 100%  Speed: 92.66 samples/s
    Waiting torch.launch to terminate...

Now "test_task" is processed. It can be examined by ``cogdata list -t test_task``::

    ------------------------------ Task Info ------------------------------
    Task Id: test_task
    Task Type: ImageTextTokenizationTask
    Description: test
    Processed:  FORMAT: dataset_name(raw_size => processed_size)
    test_ds(207.7MB => 5.4MB) 
    Hanging:  FORMAT: dataset_name(raw_size)[create_time]

    Additional:  FORMAT: dataset_name(processed_size)

    Unprocessed:  FORMAT: dataset_name(raw_size)    
    
Directory structure::

    .
    ├── cogdata_task_test_task
    │   ├── cogdata_config.json
    │   ├── main_pid_35218.log
    │   └── test_ds
    │       ├── logs
    │       │   ├── rank_0.log
    │       │   ├── rank_0.progress
    │       │   ├── rank_1.log
    │       │   └── rank_1.progress
    │       ├── meta_info.json
    │       ├── test_ds.bin.part_0.cogdata
    │       └── test_ds.bin.part_1.cogdata
    └── test_ds
        ├── cogdata_info.json
        ├── infolist.json
        └── n10148035.tar

Merge
------
There are 2 processed files now, ``test_ds.bin.part_0.cogdata`` and ``test_ds.bin.part_1.cogdata``. Because ``nproc=2`` in process.

So we need to merge them by ``cogdata merge``::

    cogdata merge [-h], [--help]                    # show the help message and exit
                  -t TASK_ID, --task_id TASK_ID     # id of the handling task

Example::

    cogdata merge -t test_task

Directory structure::

    .
    ├── cogdata_task_test_task
    │   ├── cogdata_config.json
    │   ├── main_pid_35218.log
    │   ├── merge.bin
    │   └── test_ds
    │       ├── logs
    │       │   ├── rank_0.log
    │       │   ├── rank_0.progress
    │       │   ├── rank_1.log
    │       │   └── rank_1.progress
    │       ├── meta_info.json
    │       ├── test_ds.bin.part_0.cogdata
    │       └── test_ds.bin.part_1.cogdata
    └── test_ds
        ├── cogdata_info.json
        ├── infolist.json
        └── n10148035.tar


Split
------
Use ``cogdata split`` to random split the merge result into some average subsets::

    cogdata split [-h], [--help]                    # show the help message and exit.
                  -t TASK_ID, --task_id TASK_ID     # id of the handling task.
                  n                                 # number of split pieces for the merge result.

Example::

    cogdata split -t test_task 3

Directory structure::

    .
    ├── cogdata_task_test_task
    │   ├── cogdata_config.json
    │   ├── main_pid_40494.log
    │   ├── merge.bin
    │   ├── split_merged_files
    │   │   ├── merge.bin.part0
    │   │   ├── merge.bin.part1
    │   │   └── merge.bin.part2
    │   └── test_ds
    │       ├── logs
    │       │   ├── rank_0.log
    │       │   ├── rank_0.progress
    │       │   ├── rank_1.log
    │       │   └── rank_1.progress
    │       ├── meta_info.json
    │       ├── test_ds.bin.part_0.cogdata
    │       └── test_ds.bin.part_1.cogdata
    └── test_ds
        ├── cogdata_info.json
        ├── infolist.json
        └── n10148035.tar

Clean
------
If a task crash or stay "Hanging" for too long, ``cogdata clean`` can help to remove damaged files in the task folder::
    
    cogdata clean [-h], [--help]                    # show the help message and exit
                  -t TASK_ID, --task_id TASK_ID     # id of the handling task

Example::

    cogdata clean -t test_task