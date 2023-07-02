# intel-oneAPI

#### Team Name - Delta-force
#### Problem Statement - MEDICAL IMAGE PROCESSING--SYCL-Accelerated Brain Tumor Segmentation and  Classification from MRI Images
#### Team Leader Email -  dineshjnld22@gmail.com
                         

## A Brief of the Prototype:
The DPC++ model is designed for brain tumor detection using deep learning techniques. Here's a brief overview of the model:

-->Load the TensorFlow Model: The code starts by loading a pre-trained TensorFlow model from a .pb file. The model is assumed to have been trained on a large dataset for brain tumor detection.

-->Selecting an Image: The code selects a random image from a specified folder containing brain images. This image will be used for inference.

-->Benchmarking Variables: The code initializes variables for benchmarking, including the number of processed images and the total latency.

-->Image Preprocessing: The selected image is loaded using OpenCV and preprocessed. It is converted to grayscale and resized to match the input size expected by the model.

-->Creating the Input Tensor: A TensorFlow input tensor is created with the appropriate shape and data type. The preprocessed image data is then copied to the input tensor.

-->Inference with DPC++: The DPC++ code executes the TensorFlow session to perform inference on the input tensor. The output tensor containing the tumor detection results is obtained.

-->Latency Calculation: The code measures the latency for the current image by capturing the start and end times of the inference process. The total latency is accumulated for all processed images.

-->Tumor Detection Interpretation: The output tensor is interpreted to determine the presence of a tumor. A threshold (e.g., 0.5) is applied to the tumor score, and if it exceeds the threshold, it is considered that the brain image contains a tumor.

-->Printing Results: The code prints the selected image path, whether a tumor is detected or not, and the benchmarking measures for the selected image.

-->Clean Up: The TensorFlow session is closed and resources are cleaned up.

![image](https://github.com/Dineshjnld/intel-oneAPI/assets/106725225/4c104756-12d7-4ecf-8e3f-e5f6ece34171)

## Tech Stack: 
   implementation of brain tumor detection from MRI images using the tech stack mentioned. Here's a breakdown of the code:
Programming Languages:
Python: The Python programming language is used for the Python implementation of the brain tumor detection model.
DPC++: DPC++ (Data Parallel C++) is used for the DPC++ implementation, which leverages SYCL for heterogeneous programming and parallel execution on hardware accelerators.
Libraries and Frameworks:

TensorFlow: The TensorFlow library is utilized for loading the pre-trained model and performing inference in both the Python and DPC++ implementations.
OpenCV: The OpenCV library is used for image loading, preprocessing, and resizing in the Python implementation.
CL/sycl: The CL/sycl library is used for programming with DPC++ and executing computations on heterogeneous hardware platforms.

Overall, the tech stack includes Python, TensorFlow, OpenCV, and CL/sycl for implementing and executing the brain tumor detection model in both Python and DPC++.
## Step-by-Step Code Execution Instructions:
To clone and run the brain tumor detection prototype, follow these step-by-step instructions:

To clone and run the brain tumor detection prototype, follow these step-by-step instructions:

Clone the Repository:

Open a command prompt or terminal.
Navigate to the directory where you want to clone the repository.
Run the following command to clone the repository:

Ensure that you have the necessary libraries installed, including Eigen, OpenCV, and the SYCL implementation such as Intel's DPC++ or ComputeCpp.
Refer to the respective library's documentation for installation instructions specific to your operating system.
Update the MRI Image Path:

Open the cloned project in your preferred code editor.
Locate the line cv::Mat mriImageMat = cv::imread("path/to/your/mri/image.jpg", cv::IMREAD_GRAYSCALE);.
Replace "path/to/your/mri/image.jpg" with the actual path to your MRI image file.
Ensure that the MRI image file is in grayscale format.
Build and Execute the Code:

git clone https://github.com/dineshjnld/intel-oneAPI.git
This will create a local copy of the repository on your machine.
Set up the Environment:

Ensure that you have Python and the required libraries installed. You can use pip to install the dependencies listed in the requirements.txt file:
pip install -r requirements.txt
Set up the DPC++ development environment with the necessary tools, libraries, and compilers. Refer to the documentation of your preferred DPC++ implementation for instructions on setting up the environment.
Prepare the Dataset:

Place your brain tumor dataset in a folder of your choice. Ensure that the images are properly labeled and organized.
Update the code to point to the correct dataset folder path.
Run the Python Implementation:

Open a command prompt or terminal.
Navigate to the cloned repository's directory.
Run the following command to execute the Python implementation:
python brain_tumor_detection.py
The Python code will load the dataset, train the model, perform brain tumor detection on a randomly selected image, and display the results.
Compile and Run the DPC++ Implementation:

Open a command prompt or terminal.
Navigate to the cloned repository's directory.
Compile the DPC++ code using the appropriate DPC++ compiler command. 
dpcpp brain_tumor_detection.cpp -o brain_tumor_detection
Run the compiled executable to execute the DPC++ implementation:
./brain_tumor_detection
The DPC++ code will load the dataset, perform brain tumor detection on a randomly selected image using DPC++, and display the results.

  
## What I Learned:
   From this brain tumor detection MODEL, I have  learnt several key concepts and techniques:

Data Loading and Preprocessing: I have understand how to load and preprocess image data using libraries like OpenCV and Pandas. This includes reading image files, resizing images, converting pixel values, and organizing data into a suitable format for training and inference.

Deep Learning Model Training: The code demonstrates how to train a deep learning model using the FastAI library.I have learn how to set up data loaders, define a model architecture, choose appropriate metrics, and train the model using a training loop.

Model Inference and Prediction: The code shows how to use a trained model to perform inference on new unseen data. I have see how to load a saved model, preprocess input images, pass them through the model, and interpret the model's predictions.

Integration with DPC++: The code showcases how to integrate DPC++ (Data Parallel C++) programming model into the brain tumor detection pipeline. I have  learnt how to leverage DPC++ to offload computationally intensive tasks to parallel processing devices, improving performance and efficiency.

Benchmarking and Performance Analysis: The code incorporates benchmarking measures to evaluate the performance of the brain tumor detection system.I have understand how to calculate latency, throughput, and average latency to assess the system's efficiency and make performance comparisons.

TensorFlow Integration: The code demonstrates the integration of a TensorFlow model into the brain tumor detection pipeline.I have  learnt how to load a pre-trained TensorFlow model and perform inference using TensorFlow's C++ API.

By studying and working with this brain tumor detection model, I have gained practical experience in image classification, deep learning model training, inference using both Python and DPC++, and performance analysis. These skills can be applied to various other computer vision and machine learning projects.




