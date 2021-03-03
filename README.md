# ImaGAN: Unsupervised Training of Conditional Joint CycleGAN for Transferring Style with Core Structures in Content Preserved
This is an official repo for "ImaGAN: Unsupervised Training of Conditional Joint CycleGAN for Transferring Style with Core Structures in Content Preserved" implemented using PyTorch.
This code heavily borrows from the code for "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks" by [carpedm20](https://github.com/carpedm20) available [here](https://github.com/carpedm20/DiscoGAN-pytorch).
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

---

If you find our work useful please cite 
```bibtex
@InProceedings{10.1007/978-3-030-20890-5_29,
author = {Bae, Kangmin and Ma, Minuk and Jang, Hyunjun and Ju, Minjeong and Park, Hyoungwoo and Yoo, Chang},
year = {2019},
month = {06},
pages = {447-462},
booktitle={Asian Conference on Computer Vision 2018},
title = {ImaGAN: Unsupervised Training of Conditional Joint CycleGAN for Transferring Style with Core Structures in Content Preserved},
isbn = {978-3-030-20889-9},
doi = {10.1007/978-3-030-20890-5_29}
}
```
