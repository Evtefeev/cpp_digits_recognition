#include <fann.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>


char intToChar(int n) {
    if (n >= 0 && n <= 9) {
        return '0' + n;  // Convert to digit character
    } else if (n >= 10 && n <= 36) {
        return 'A' + (n - 10);  // Convert to uppercase letter
    } else {
        return '\0';  // Invalid input
    }
}
// Function to preprocess the image: convert to grayscale, resize, and normalize
std::vector<float> preprocessImage(const std::string &imagePath, int imgSize)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        throw std::runtime_error("Error: Unable to load image " + imagePath);
    }


    // Apply binary thresholding
    cv::Mat binaryImage;
    cv::threshold(img, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Resize the image to 28x28
    cv::resize(binaryImage, binaryImage, cv::Size(imgSize, imgSize));

    // Normalize pixel values to [0, 1]
    std::vector<float> imageData(imgSize * imgSize);
    for (int i = 0; i < binaryImage.rows; ++i)
    {
        for (int j = 0; j < binaryImage.cols; ++j)
        {
            imageData[i * binaryImage.cols + j] = 1.0f-(binaryImage.at<uchar>(i, j) / 255.0f);
        }
    }

    return imageData;
}

// Function to recognize the digit using the trained neural network
int recognizeDigit(std::vector<float> &imageData, const std::string &annFilePath, int class_count)
{
    struct fann *ann = fann_create_from_file(annFilePath.c_str());
    if (!ann)
    {
        throw std::runtime_error("Error: Unable to load ANN from file " + annFilePath);
    }
    // for (auto t : imageData)
    // {
    //     std::cout << t << std::endl;
    // }
    fann_type *output = fann_run(ann, imageData.data());
    for (int i = 0; i < class_count; i++)
    {

        std::cout << intToChar(i) << ": " << output[i] << std::endl;
    }
    int predictedDigit = std::max_element(output, output + class_count) - output;

    fann_destroy(ann);
    return predictedDigit;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path> <ann_path>" << std::endl;
        return 1;
    }

    const std::string imagePath = argv[1];
    const std::string annFilePath = argv[2];
    const int imgSize = 28;

    try
    {
        std::vector<float> imageData = preprocessImage(imagePath, imgSize);
        int digit = recognizeDigit(imageData, annFilePath, 36);
        std::cout << "Recognized digit: " << intToChar(digit) << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
