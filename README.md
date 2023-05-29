# land-use-image-classification
Multi-label classification of Land use images. Project for Python course at YSU Applied Statistics and Data Science master program.

### Setup

Install required packages.
- ```
  pip install -r requirements.txt
  ```

Download [UCMerced_LandUse](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and [EuroSAT_RGB](https://github.com/phelber/EuroSAT) datasets and put them in `data` folder.

### Dataset creation

Multilabel dataset is created from multiclass one with `data_processing.py`

```
usage: data_preprocessing.py [-h] [--data-path DATA_PATH] [--output-path OUTPUT_PATH] [--extension EXTENSION] [--seed SEED] [--width WIDTH] [--height HEIGHT] [--batch-size BATCH_SIZE] [--shuffle]
```

`DATA_PATH` must have following structure

- `DATA_PATH`

  - `Images`

    - `class1`
      - `img1.jpg`
      - `img2.jpg`
      - ........
    - `class2`
      - `img1.jpg`
      - `img2.jpg`
      - ........

    - ........
    - `classN`
      - `img1.jpg`
      - `img2.jpg`
      - ........

The script concatenates images of different classes. The number of images to be combined in one is controlled by `BATCH_SIZE`.

As a result `OUTPUT_PATH` will be created with following structure

- `OUTPUT_PATH`
  - `Images`
    - `combined_img1.jpg`
    - `combined_img2.jpg`
    - .........
  - `image_lables.json`

`image_labels.json` contains labels(as one-hot vector) for created images.

### Data augmentation

As augmentation Image blending, affine and perspective transformations are used.

- Image blending is used for smoothing horizontal and vertical lines passing through center of an image (those are appeared because of concatenating images). As background satellite images can be used. Those also can be a images processed with the steps described in dataset creation section.
- Affine transformation is used to rotate images. The transformation is applied on enlarged image (with filled border, BORDER_REFLECT is used), then image crop (from center) is taken with original size.
- Perspective transformation is applied using the same logic described for affine transformation.

Augmentations are applied on multiclass dataset with `data_augmentation.py`

```
usage: data_augmentation.py [-h] [--data_path DATA_PATH] [--label_path LABEL_PATH] [--bg_data_path BG_DATA_PATH] [--output_path OUTPUT_PATH] [--extension EXTENSION] [--seed SEED] [--affine_prop AFFINE_PROP] [--perspective_prop PERSPECTIVE_PROP]
```

Proportions of dataset to be used for transformations are controlled by `AFFINE_PROP` and `PERSPECTIVE_PROP`.

#### Steps to create multilabel augmented dataset

We used **UCMerced_LandUse**  for training and **EuroSAT_RGB** as background for augmentation.

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

    - If sum of **affine_prop** and **perspective_prop** is greater than 1 both augmentations will be applied on some part of data. In the example above only affine transformation will be applied on 0.3 portion of data, only perspective transformation will be applied on 0.5 portion of data, both augmentations will be applied on 0.2 portion of data
    - Images with augmentations will be saved with names `<image_id>_aff`, `<image_id>_pers` or `<image_id>_aff_pers`. Corresponding `image_labels.json` will be created.


### Training
  Use  `train.py` to train a model. Different ResNet and DenseNet models are supported. Best model is saved in `models` folder. 

  ```
  usage: train.py [-h] --model MODEL --data DATA [--batch-size BATCH_SIZE] [--lr LR] [--train-portion TRAIN_PORTION] [--device DEVICE] [--epochs EPOCHS] [--seed SEED] [--num-classes NUM_CLASSES] [--width WIDTH] [--height HEIGHT] [--tensorboard]
  ```

  An example of starting a training. 

  - ```
    python scripts/train.py --model resnet18 --data data/UCMerced_LandUse_augmented  --batch-size 16 --epochs 40  --tensorboard
    ```
    

  Inspect tensorboard.
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

