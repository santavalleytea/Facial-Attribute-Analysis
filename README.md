# COMP3419-Facial-Attribute-Analysis

Install the following packages if not installed:

1. PyTorch: `pip install torch torchvision`
2. Pandas: `pip install pandas`
3. Pillow: `pip install Pillow`

Steps for running the code:

1. `python3 model.py` or `python model.py` depending on the system used.

Additional edits for fine-tuning methods:

1. If you would like to adjust the number of images the model goes through, adjust the `max_images` parameter in `train_loader`, `val_loader` and `test_loader` in `data_load.py`. Located in line 56.
2. If you would like to implement or exclude data augmentation, keep or comment out `transforms.RandomHorizontalFlip()` in `data_load.py`. Located in line 51.
3. If you would like to adjust the learning rate of the model, adjust the `lr` parameter within `optim.Adam(model.parameters(), lr=x)` in `model.py`. Located in line 39.
