# land-use-image-classification
Multi-label classification of Land use images. Project for Python course at YSU Applied Statistics and Data Science master program.

### Setup

Install required packages.
- ```
  pip install -r requirements.txt
  ```

Download [UCMerced_LandUse](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and [EuroSAT_RGB](https://github.com/phelber/EuroSAT) datasets and put them in `data` folder.

### Data augmentation

**UCMerced_LandUse** is used for training. **EuroSAT_RGB** dataset is used for augmentation.

#### Steps to create augmented dataset

- Create multilabel datasets.

  - ```
    python scripts/data_preprocessing.py --shuffle
    ```

  - ```
    python scripts/data_preprocessing.py --data-path data/EuroSAT_RGB --width 64 --height 64 --batch-size 64 --output-path data/EuroSAT_RGB_processed
    ```

- Apply augmentations

  - ```
    python scripts/data_augmentation.py --affine_prop=0.5 --perspective_prop=0.7
    ```

  - **NOTE**:
    - If sum of **affine_prop** and **perspective_prop**   is greater than 1 both augmentations will be applied on some part of data. In the example above only affine transformation will be applied on 0.3 portion of data, only perspective transformation  will be applied on 0.5 portion of data, both augmentations will be applied on 0.2 portion of data
    - Images with augmentation will be saved with names `<image_id>_aff`, `<image_id>_pers` or `<image_id>_aff_pers` Corresponding `image_labels.json` will be created.


### Training
  Use  `train.py` to train a model. Different ResNet and DenseNet models are supported. Best model is saved in `models` folder. 

  ```
  usage: train.py [-h] --model MODEL --data DATA [--batch-size BATCH_SIZE] [--lr LR] [--train-portion TRAIN_PORTION] [--device DEVICE] [--epochs EPOCHS] [--seed SEED] [--num-classes NUM_CLASSES] [--width WIDTH] [--height HEIGHT] [--tensorboard]
  ```

  An example of starting a training. 

  - ```
    python scripts/train.py --model resnet18 --data data/UCMerced_LandUse_augmented  --batch-size 16 --epochs 40  --tensorboard
    ```
    
  Inpsect tensorboard.
  - ```
    tensorboard --logdir=runs
    ```
    
### Inference

  Use `inference.py` for an inference of a single image.

  ```
  usage: inference.py [-h] --image-path IMAGE_PATH [--model-name MODEL_NAME] --model-path MODEL_PATH [--device DEVICE] [--threshold THRESHOLD] [--width WIDTH] [--height HEIGHT]
  ```
  
  An example of inference command.
  - ```
    python scripts/inference.py --image-path data/UCMerced_LandUse_augmented/Images/002df11b.jpg --model-path models/resnet18_2023-05-28_22-15-28_best_weights.pt
    ```

  
### Demo

Run demo with following command

```
streamlit run demo/app.py
```

