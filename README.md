
# BB-SGD : background boostrapping using stochastic gradient descent 

This repository is the official implementation of the model BB-SGD described in the paper "BB-SGD : fast and accurate background reconstruction using background bootstrapping"


## Requirements

The model requires Pytorch (>= 1.7.1) and Torchvision with cuda capability 

The model also requires OpenCV (>=4.1) 

To install other requirements:

```setup
pip install -r requirements.txt
```
The model has been tested with  Nvidia RTX 2080 TI and Nvidia RTX 3090 GPU.

## How to use the model

the command to generate the background from a sequence of frames is 

```
python main.py --input_path your_input_path 
```

where your_input_path is the path to the folder where the frame sequence is saved.
Example : python main.py --input_path /workspace/Datasets/SBMnet_dataset/basic/511/input

the result background image will be stored in the current working directory.
To view options, type python main.py -h

## Evaluation

To evaluate the BB-SGD model on the SBMnet 2016 dataset: 

- download the SBMnet 2016 dataset from the following link : 
```
 http://pione.dinf.usherbrooke.ca/static/dataset/SBMnet_dataset.zip
```
and save it to some folder

- generate the backgrounds using the command 

```
python SBM_processing_pipeline.py datasetPath resultPath
```

where datasetPath is the path to the saved SBMnet dataset and resultPath is the path to the result folder where you want to store the results.

Example : python SBM_processing_pipeline.py  /workspace/Datasets/SBMnet_dataset  /workspace/Datasets/SBMnet_results

This pipeline will compute and save the 79 backgrounds associated to the 79 sequences of the dataset

In order to compute statistics for the sequences where a ground truth is provided with the SBMnet dataset, use the command

```
python SBM_UTILITY.py groundtruthtPath resultPath
```
example : python SBM_UTILITY.py  /workspace/Datasets/SBMnet_dataset  /workspace/Datasets/SBMnet_results
the evaluation statistics will be stored as cm.txt file in the resultPath folder


If you want also to complete this evaluation with other sequences where a ground truth background is also publicly available but not provided with the SBMnet dataset,
go the website https://sbmi2015.na.icar.cnr.it/SBIdataset.html, download the groundtruth images associated to the sequences
Toscana, Candela_m1.10,CaVignal,Foliage, People&Foliage, and add them to the appropriate folder of the SBM dataset before starting the evaluation utility.


Warning : Different runs of the model with the same inputs may lead to small differences in evaluation results ompared to the results  due to the random initialization of the image.



# Acknowledgment  
The processing and evaluation codes for the SBMnet dataset ( SBM_Evaluate.py, SBM_gauss.py, SBM_processing_pipeline.py and SBM_UTILITY.py) are adapted from the evaluation codes available on the SBMnet webiste
(links http://pione.dinf.usherbrooke.ca/code)





