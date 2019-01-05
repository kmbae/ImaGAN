# ImaGAN

## Download datasets (from [pix2pix](https://github.com/phillipi/pix2pix)) with:

    $ bash ./data/download_dataset.sh dataset_name

- `edges2shoes`: 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/).
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN).

## Start training

    $ python main.py

## Requirements
* pytorch 0.4.2
* tensorboardX
* tqdm
