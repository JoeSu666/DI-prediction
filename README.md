# DI-prediction
This is the official implementation of our article "Combining frontal and profile view facial images with clinical data to predict difficult-to-intubate patients using AI".

## Data
Please create folder named ./data, and put your facial data into the corresponding folders.
Modify the folder names in <mydataset.py>.

## Train
```$ python DI_translrn.py --arch <> --data <>```

## Evaluation
```$ python DI_translrn.py --arch <> --data <> --evaluate ```
