# SJTU-EE228-project2
## ribfrac dectection, segmentation and classification
This a implementation of 3DUnet on pytorch for ribfrac detection task. The main parts of data preprocessing are inspired from https://github.com/M3DV/FracNet
### Getting start
All the data are from https://ribfrac.grand-challenge.org/dataset/ and the data should be placed under the directory like
```
data
|--ribfrac
|	|--train
|	|	 |--RibFrac1-image.nii.gz
|	|--val
|	|	|--RibFrac421-image.nii.gz
|	|--test
|	|	|--RibFrac501-image.nii.gz
|	|--train_label
|	|	|--RibFrac1-label.nii.gz
|	|--val_label
|	|	|--RibFrac421-label.nii.gz
SJTU-EE228-project2
```

To train a model, just simply run the scripts in ./script directory.
```
cd ./script/
bash train2.sh
```
The model parameter will be saved in ../output/${task_id}/

To get the prediction file on test set, use the command
```
python predict.py --task_id ${your task id} 
```
The output prediction result will be saved in ../output/${task_id}/

### test result
The prediction result on VALIDATION set is uploaded in ./prediction_directory
