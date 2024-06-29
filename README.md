# YOLOv10 Fine-Tuning for Kidney Stone Detection

This project focuses on fine-tuning the YOLOv10 model for the custom object detection of kidney stones. Leveraging a dataset sourced from Roboflow, this repository provides a comprehensive setup for training and deploying a YOLOv10 model specifically for medical imaging applications.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Object detection in medical imaging, particularly for identifying kidney stones, is a critical task that can aid in faster and more accurate diagnosis. This project utilizes the YOLOv10 model, a state-of-the-art object detection algorithm, fine-tuned with a specific dataset to detect kidney stones.

## Features

- Fine-tuned YOLOv10 model for kidney stone detection
- Training pipeline
- Easy-to-use inference script


## Dataset

The dataset used for this project is sourced from [Roboflow](https://roboflow.com/), which contains annotated images of kidney stones. The dataset is split into training, validation, and test sets. 

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/suhanisuha/yolov10_kidneystone.git
   cd yolov10_kidneystone
