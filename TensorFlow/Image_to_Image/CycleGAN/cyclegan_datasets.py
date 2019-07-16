"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'cityscapes_train': 2975,
    'cityscapes_test': 500
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'cityscapes_train': '.jpg',
    'cityscapes_test': '.jpg',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'cityscapes_train': './input/cityscapes/cityscapes_train.csv',
    'cityscapes_test': './input/cityscapes/cityscapes_test.csv',
}
