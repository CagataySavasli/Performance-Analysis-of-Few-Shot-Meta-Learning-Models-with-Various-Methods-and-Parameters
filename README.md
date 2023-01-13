# Performance-Analysis-of-Few-Shot-Meta-Learning-Models-with-Various-Methods-and-Parameters



## Requirements
- Python >= 3.x
- Numpy >= 1.21.5
- Torch >= 1.13.0
- GPyTorch >= 1.9.0
- (optional) TensorboardX


## Experiments
These are the instructions to train and test the methods reported in the paper in the various conditions.

Two data sets, QMUL and AAF, were used in this project. The data set named AAF is downloaded and prepared together with cloning this repository. To download and prepare the QMUL data set:

```
cd filelists/QMUL/
sh download_QMUL.sh
```

The case to be run in the project must be defined in the `main.py` file. There are 4 parameters in this file: `method`, `dataset`, `model`, and `kernel type`. Changes to these parameters should be made in the `main.py` file and then run the python file.

```
train.main(method, dataset, model, kernel_type)
result = test.main(method, dataset, model, kernel_type)
```

**Method :** There are a few available methods that you can use: `DKT`, `gpnet`, and `transfer` (Feature Transfer Reggression). Baseline corresponds to feature transfer and `gpnet` is a modified method of `DKT`. You must use those exact strings at training and test time when you call the `main.py`.

**Data Set :** There are a few available data set that you can use: `QMUL` and `AAF`. You must use those exact strings at training and test time when you call the `main.py`.

**Model :** The script allows training and testing on different backbone networks. There are a few available model that you can use: `Conv3|4|6`, `ResNet10|18|34|50|101`. You must use those exact strings at training and test time when you call the `main.py`.

**Kernel Type :** There are a few available kernel types that you can use: `rbf`, `linear`, `matern`, `poli1`, and `poli2`. Although `gencheb` completes the kernel train phase without any problems, it receives a `gpytorch bug error` during the testing phase. *Work continues to fix the problem*. You must use those exact strings at training and test time when you call the `main.py`.


## Regression

**QMUL Head Pose Trajectory Regression** In order to run this experiment you first have to download and setup the QMUL dataset, this can be done automatically running the file `download_QMUL.sh` from the folder `filelists/QMUL`. The methods that can be used for regression are `DKT`, `gpnet` and `transfer` (feature transfer). In order to train these methods, use:

```
dataset = "QMUL" 
```
in `main.py`


**AAF Face and Age Dataset** The methods that can be used for regression are `DKT`, `gpnet` and `transfer` (feature transfer) too. In order to train these methods, use:

```
dataset = "QMUL" 
```
in `main.py`


**For Train :**
The number of training epochs can be set with `stop_epoch`.

**For Test :**
You can additionally specify the size of the support set with `n_support` (which defaults to 5), and the number of test epochs with `n_test_epochs` (which defaults to 10).


**After all settings :**
```
python main.py
```
or
```
python3 main.py
```
