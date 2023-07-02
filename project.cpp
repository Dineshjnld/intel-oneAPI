#include <CL/sycl.hpp>
#include <iostream>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <random>

namespace fs = std::filesystem;

int main() {
    // Create a SYCL queue for DPC++ computations
    cl::sycl::queue q;

// Load the TensorFlow model
    std::string model_path = "path/to/tensorflow_model.pb";
    tensorflow::SessionOptions session_options;
    tensorflow::Session* session = tensorflow::NewSession(session_options);
    tensorflow::Status status = session->Create(model_path);
    if (!status.ok()) {
     std::cerr << "Failed to load TensorFlow model: " << status.ToString() << std::endl;
        return 1;
    }

    // Path to the folder containing brain images
    std::string folder_path = "path/to/brain_images";

    // Get the list of image files in the folder
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            image_files.push_back(entry.path().string());
        }
    }

    // Select a random image from the list
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, image_files.size() - 1);
    int random_index = dis(gen);
    std::string selected_image_path = image_files[random_index];

    // Benchmarking variables
    int processed_images = 0;
    std::chrono::duration<double, std::milli> total_latency = std::chrono::duration<double, std::milli>::zero();

    // Load and preprocess the selected image
    cv::Mat image = cv::imread(selected_image_path, cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(224, 224));  // Resize image to match model input size

    // Convert image to float32 and normalize pixel values
    cv::Mat image_float;
    image.convertTo(image_float, CV_32F, 1.0 / 255.0);

    // Create a TensorFlow input tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 224, 224, 1}));
    float* input_data_ptr = input_tensor.flat<float>().data();

    // Copy image data to the input tensor
    std::memcpy(input_data_ptr, image_float.ptr(), image_float.total() * sizeof(float));

    // Create DPC++ event to capture execution time
    cl::sycl::event event;

    // Run TensorFlow session to perform inference
    std::vector<tensorflow::Tensor> outputs;
    auto start_time = std::chrono::high_resolution_clock::now();
    status = session->Run({{"input_tensor", input_tensor}}, {"output_tensor"}, {}, &outputs, &event);
    if (!status.ok()) {
        std::cerr << "Failed to run TensorFlow session: " << status.ToString() << std::endl;
        return 1;
    }

    // Wait for the DPC++ event to complete
    event.wait();
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate latency for the current image
    std::chrono::duration<double, std::milli> latency = end_time - start_time;
    total_latency += latency;

    // Get the output tensor containing the tumor detection results
    const tensorflow::Tensor& output_tensor = outputs[0];
    const float* output_data_ptr = output_tensor.flat<float>().data();

    // Interpret the output to determine tumor presence
    float tumor_score = output_data_ptr[0];
    bool has_tumor = tumor_score > 0.5;

    // Print the result for the selected image
    std::cout << "Selected Image: " << selected_image_path << std::endl;
    if (has_tumor) {
        std::cout << "The brain image contains a tumor." << std::endl;
    } else {
        std::cout << "The brain image does not contain a tumor." << std::endl;
    }
    std::cout << std::endl;

    // Calculate throughput and average latency
    double throughput = 1 / (total_latency.count() / 1000.0);
    double average_latency = total_latency.count() / processed_images;

    // Print benchmarking measures
    std::cout << "Benchmarking Measures:" << std::endl;
    std::cout << "Processed Images: " << processed_images << std::endl;
    std::cout << "Total Latency (ms): " << total_latency.count() << std::endl;
    std::cout << "Throughput (images/sec): " << throughput << std::endl;
    std::cout << "Average Latency (ms): " << average_latency << std::endl;

    // Clean up resources
    session->Close();
    delete session;

    return 0;
}
