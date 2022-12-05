# DI-prediction
This is the official implementation of our article "Combining frontal and profile view facial images with clinical data to predict difficult-to-intubate patients using AI".

## Data
Please create a folder named *./data*, and put your facial data into the corresponding folders.
Modify the folder names in <mydataset.py> before running.

## Train
```$ python DI_translrn.py --arch <> --data <> --split <> --code <> ```

## Evaluation
```$ python eval.py --arch <> --data <> --split <>```


Please reach out if you have any question or suggestion!
[email](zsu@wakehealth.edu) 
