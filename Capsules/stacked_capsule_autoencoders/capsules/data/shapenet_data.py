import os
import sys
import pdb
import glob
sys.path.append(os.getcwd())

from PIL import Image
import os.path
import numpy as np
import argparse
import time
import random
import tensorflow_datasets.public_api as tfds


class ShapeNetImageDataset(tfds.core.GeneratorBasedBuilder):
    """Short description of my dataset."""

    VERSION = tfds.core.Version('0.1.3')

    def _info(self):
        print("Shapenet!!!!")
        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=("This is the dataset for Shapenet Images. It contains yyy. The "
                         "images are kept at their original dimensions."),
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                "image_description": tfds.features.Text(),
                "image": tfds.features.Image(shape = (128, 128, 1)),
                # Here, labels can be of 5 distinct values.
                "label": tfds.features.ClassLabel(num_classes=10),
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("image", "label"),
            # Homepage of the dataset for documentation
            # Bibtex citation for the dataset
            citation=r"""@article{shapenet-images-dataset-2020,
                                  author = {Zen, Luo},"}""",
        )

    def _split_generators(self, dl_manager):
        # Downloads the data and defines the splits
        # dl_manager is a tfds.download.DownloadManager that can be used to
        # download and extract URLs

        # Specify the splits
        self.root = "/hdd/zen/data/Reallite/Rendering/chair_cls1"
        classes = ['38436cce91dfe9a340b2974a4bd47901', 'ccfc857f35c138ede785b88cc9024b2a',
                        '3fdc09d3065fa3c524e7e8a625efb2a7', '795f38ce5d8519938077cafed2bb8242',
                        '30b0196b3b5431da2f95e2a1e9997b85', 'b405fa036403fbdb307776da88d1350f',
                        '3af3096611c8eb363d658c402d71b967', 'ce8e6c13899376e2f3c9c1464e55d580',
                        '4527cc19d2f09853a718067b9ac932e1', '124ef426dfa0aa38ff6069724068a578']
        self.class_2_label = { classes[i]:i for i in range(len(classes))}

        all_images = []
        for class_name in classes:
            all_image = sorted(glob.glob(os.path.join(self.root, "data", class_name, "*-color.png")))
            all_image = [os.path.join(self.root, "data", class_name, i.split("/")[-1].split("-")[0])  for i in all_image]
            all_images = all_images + all_image

        self.data_list = all_images
        split = 6/7

        split_num = np.floor((1- split) * len(self.data_list)).astype(int)
        self.test_data_list = self.data_list[:split_num]

        split_num = np.floor((split) * len(self.data_list)).astype(int)
        self.train_data_list = self.data_list[:split_num]

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_dir_path": self.train_data_list,
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "images_dir_path": self.test_data_list,
                },
            ),
        ]

    def _generate_examples(self, images_dir_path):
        for image_dir in images_dir_path:
#             img = Image.open('{}-color.png'.format(image_dir))
            img = Image.open('{}-color.png'.format(image_dir))
            img = np.array(img.convert('L'))[:,:,None]
            meta = np.load('{}-meta.npz'.format(image_dir))
            model_id = str(meta['model_id'])
            label = self.class_2_label[model_id]
            
            yield image_dir, {
                "image_description": image_dir,
                "image": img,
                "label": label,
            }

if __name__ == "__main__":
    sn_dataset_builder = ShapeNetImageDataset()
    sn_dataset_builder.download_and_prepare()

