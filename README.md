# Covid-Face-Mask-Detection

## Overview

This project is designed to detect in real-time whether a person is wearing a mask or not using computer vision techniques. It leverages MobileNetV2 transfer learning for feature extraction, OpenCV for image processing, and Haar Cascade for frontal face detection.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Demo](#demo)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before using or contributing to this project, ensure you have the following prerequisites installed:

- Python (3.x recommended)
- OpenCV
- TensorFlow (for MobileNetV2)
- Haar Cascade Frontal Face Classifier (provided in OpenCV)

## Getting Started

1. Clone this repository to your local machine:

```git clone https://github.com/Ansh1999/Covid-Face-Mask-Detection.git```

2. Install the required Python packages:

   ```pip install -r requirements.txt```

3. Run the mask detection script:

   ```python main.py```

## Usage

* Start the script, and it will use your computer's camera to perform real-time mask detection on people's faces in the video stream.
  
* Detected faces are first located using Haar Cascade frontal face detection.
  
* For each detected face, the model classifies whether the person is wearing a mask or not.

## Customization

You can customize this project in several ways:

*Train the MobileNetV2 model on your own dataset for improved accuracy.

*Adjust the model's confidence threshold to control the detection sensitivity.

*Modify the Haar Cascade classifier for specific face detection requirements.

*Integrate this project into other applications, such as access control systems or COVID-19 safety monitoring tools.





