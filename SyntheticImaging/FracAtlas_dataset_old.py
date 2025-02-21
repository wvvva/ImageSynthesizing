#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 23:13:51 2024

@author: houyuhan
"""


#Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FracAtlas Dataset Loader

This script provides a Hugging Face `datasets` loader for the FracAtlas dataset, a comprehensive collection
of musculoskeletal radiographs aimed at advancing research in fracture classification, localization, and segmentation.
The dataset includes high-quality X-Ray images accompanied by detailed annotations in COCO JSON format for segmentation
and bounding box information, as well as PASCAL VOC XML files for additional localization data.

The loader handles downloading and preparing the dataset, making it readily available for machine learning models and analysis
tasks in medical imaging, especially focusing on the detection and understanding of bone fractures.

License: CC-BY 4.0
"""


import csv
import json
import os
from typing import List
import datasets
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import xml.etree.ElementTree as ET
from datasets import load_dataset, DownloadConfig
import requests
import zipfile
from io import BytesIO


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:yh0701/FracAtlas_dataset,
title = {FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs},
author={Abedeen, Iftekharul; Rahman, Md. Ashiqur; Zohra Prottyasha, Fatema; Ahmed, Tasnim; Mohmud Chowdhury, Tareque; Shatabda, Swakkhar},
year={2023}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
The "FracAtlas" dataset is a collection of musculoskeletal radiographs for fracture classification, localization, and segmentation.
It includes 4,083 X-Ray images with annotations in multiple formats.The annotations include bbox, segmentations, and etc.
The dataset is intended for use in deep learning tasks in medical imaging, specifically targeting the understanding of bone fractures.
It is freely available under a CC-BY 4.0 license.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://figshare.com/articles/dataset/The_dataset/22363012"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "The dataset is licensed under a CC-BY 4.0 license."

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class FracAtlasDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""
    VERSION = datasets.Version("1.1.0")

    def _info(self):
      return datasets.DatasetInfo(
          description=_DESCRIPTION,
          features=datasets.Features(
              {
                  "image_id": datasets.Value("string"),
                  "image": datasets.Image(),
                  "hand": datasets.ClassLabel(num_classes=2,names=['no_hand','hand']),
                  "leg": datasets.ClassLabel(num_classes=2,names=['no_leg','leg']),
                  "hip": datasets.ClassLabel(num_classes=2,names=['no_hip','hip']),
                  "shoulder": datasets.ClassLabel(num_classes=2,names=['no_shoulder','shoulder']),
                  "mixed": datasets.ClassLabel(num_classes=2,names=['not_mixed','mixed']),
                  "hardware": datasets.ClassLabel(num_classes=2,names=['no_hardware','hardware']),
                  "multiscan": datasets.ClassLabel(num_classes=2,names=['not_multiscan','multiscan']),
                  "fractured": datasets.ClassLabel(num_classes=2,names=['not_fractured','fractured']),
                  "fracture_count": datasets.Value("int32"),
                  "frontal": datasets.ClassLabel(num_classes=2,names=['not_frontal','frontal']),
                  "lateral": datasets.ClassLabel(num_classes=2,names=['not_lateral','lateral']),
                  "oblique": datasets.ClassLabel(num_classes=2,names=['not_oblique','oblique']),
                  "localization_metadata": datasets.Features({
                    "width": datasets.Value("int32"),
                    "height": datasets.Value("int32"),
                    "depth": datasets.Value("int32"),
                }),
                  "segmentation_metadata": datasets.Features({
                      "segmentation": datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
                      "bbox": datasets.Sequence(datasets.Value("float")),
                      "area": datasets.Value("float")
                      }) or None
              }
          ),
          # No default supervised_keys (as we have to pass both question
          # and context as input).
          supervised_keys=None,
          homepage=_HOMEPAGE,
          citation=_CITATION
      )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
      downloaded_files = '/content/drive/MyDrive/Research/Yoon/SyntheticImaging'

      base_path = os.path.join(downloaded_files, "FracAtlas")

      # Split the dataset to train/test/validation by 0.7,0.15,0.15
      df = pd.read_csv(os.path.join(base_path, 'dataset.csv'))
      train_df, test_df = train_test_split(df, test_size=0.3)
      validation_df, test_df = train_test_split(test_df, test_size=0.5)

      # store them back as csv
      train_df.to_csv(os.path.join(base_path, 'train_dataset.csv'), index=False)
      validation_df.to_csv(os.path.join(base_path, 'validation_dataset.csv'), index=False)
      test_df.to_csv(os.path.join(base_path, 'test_dataset.csv'), index=False)

      annotations_path = os.path.join(base_path, 'Annotations/COCO JSON/COCO_fracture_masks.json')
      images_path = os.path.join(base_path, 'images')
      localization_path = os.path.join(base_path, 'Annotations/PASCAL VOC')

      return [
          datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"dataset_csv_path": os.path.join(base_path, 'train_dataset.csv'),
                                                                          "images_path": images_path,
                                                                         "annotations_path": annotations_path,
                                                                         "localization_path":localization_path
                                                                        }),
          datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"dataset_csv_path": os.path.join(base_path, 'validation_dataset.csv'),
                                                                               "images_path": images_path,
                                                                              "annotations_path": annotations_path,
                                                                              "localization_path":localization_path
                                                                             }),
          datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"dataset_csv_path": os.path.join(base_path, 'test_dataset.csv'),
                                                                         "images_path": images_path,
                                                                        "annotations_path": annotations_path,
                                                                        "localization_path":localization_path
                                                                        })
      ]

    def _generate_examples(self, annotations_path, images_path, dataset_csv_path,localization_path):
        logging.info("Generating examples from = %s", dataset_csv_path)
        split_df = pd.read_csv(dataset_csv_path)  # Load the DataFrame for the current split

        # Function to convert numeric ID to formatted string
        def format_image_id(numeric_id):
            return f"IMG{numeric_id:07d}.jpg"  # Adjust format as needed

        # Function to extract information from xml files
        def parse_xml(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

        # Extract the necessary information
            width = int(root.find("./size/width").text)
            height = int(root.find("./size/height").text)
            depth = int(root.find("./size/depth").text)
            segmented = int(root.find("./segmented").text)
            return width, height, depth, segmented

        # Load annotations
        with open(annotations_path) as file:
            annotations_json = json.load(file)

        for item in annotations_json['annotations']:
            item['image_id'] = format_image_id(item['image_id'])

        annotations = {item['image_id']: item for item in annotations_json['annotations']}


        # Iterate through each row in the split DataFrame
        for _, row in split_df.iterrows():
            image_id = row['image_id']
            # Determine the folder based on the 'fractured' column
            folder = 'Fractured' if row['fractured'] == 1 else 'Non_fractured'

            # Check if the formatted_image_id exists in annotations
            annotation = annotations.get(image_id)
            image_path = os.path.join(images_path, folder, image_id)

            # Initialize variables
            segmentation, bbox, area = None, None, None
            segmentation_metadata = None

            if annotation:
                segmentation = annotation.get('segmentation')
                bbox = annotation.get('bbox')
                area = annotation.get('area')

                segmentation_metadata = {
                    'segmentation':  segmentation,
                    'bbox':bbox,
                    'area': area
                    }
            else:
                segmentation_metadata = None # Default if not present

            xml_file_name = f"{image_id.split('.')[0]}.xml"
            xml_path = os.path.join(localization_path, xml_file_name)

            # Parse the XML file
            width, height, depth, _ = parse_xml(xml_path)

            localization_metadata = {
                'width': width,
                "height":height,
                'depth': depth
                }


            # Construct example data
            example_data = {
                "image_id": row['image_id'],
                "image":image_path,
                "hand": row["hand"],
                "leg": row["leg"],
                "hip": row["hip"],
                "shoulder": row["shoulder"],
                "mixed": row["mixed"],
                "hardware": row["hardware"],
                "multiscan": row["multiscan"],
                "fractured": row["fractured"],
                "fracture_count": row["fracture_count"],
                "frontal": row["frontal"],
                "lateral": row["lateral"],
                "oblique": row["oblique"],
                "localization_metadata": localization_metadata,
                'segmentation_metadata': segmentation_metadata
            }
            yield image_id, example_data
