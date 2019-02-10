# ImaGAN
This code is a fork from the code for "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks" by [carpedm20](https://github.com/carpedm20) available [here](https://github.com/carpedm20/DiscoGAN-pytorch).
## Download datasets (from [pix2pix](https://github.com/phillipi/pix2pix)) with:

    $ bash ./data/download_dataset.sh edges2shoes
    $ bash ./data/download_dataset.sh edges2handbags

- `edges2shoes`: 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/).
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN).

## Requirements

    $ pip install -r requirements.txt

## Start training

    $ python main.py

## To monitor results

    $ tensorboard --logdir runs

Check points are saved in **logs**

tensorboard summaries are saved in **runs**
