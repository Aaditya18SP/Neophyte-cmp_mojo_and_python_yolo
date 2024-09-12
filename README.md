# Compare YOLO model using Python and Mojo
We are comparing the performance of the YOLO model on ***CPU*** implemented using Python and Mojo.


## MOJO setup
Mojo is installed using `Magic`. I use `Magic` to run mojo as well hence that implies that I am using the `Max` framework.

`magic` is based on `conda`. It uses `conda` under the hood and hence resolves to `conda-forge` and other related repositories for downloading packages

Packages to install using magic
1. pillow
2. pytorch (I didnt need to specify cpu or gpu version. I installed it and it works)
3. opencv

eg: `magic add pillow pytorch opencv`

To execute the mojo code. Just run the `run.mojo` file in the `Mojo-Yolo/MAX` directory. 
The run command is `magic run mojo run.mojo`

## PYTHON setup
Just execute the `yolo_predict.py` file in the `Python_Yolo` directory. 


## Dataset
I am using the 2017 Val Coco dataset which contains 5000 images
Link:- 

## Metrics measured
1. FPS
2. Average Cpu usage = Total usage of the process on all cores/ no of cores
3. Total model execution time. 
