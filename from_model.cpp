#include "model3.c"
#include <vector>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>

void preprocessImage(
    float (&input)[56][56][1],
    const std::string &imagePath,
    int imgSize,
    bool binary = false,
    bool invert = false)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        throw std::runtime_error("Error: Unable to load image " + imagePath);
    }

    // Apply binary thresholding
    cv::Mat binaryImage = img;
    if (binary)
    {
        cv::threshold(img, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    // Resize the image to 28x28
    cv::resize(binaryImage, binaryImage, cv::Size(imgSize, imgSize));

    // Normalize pixel values to [0, 1]
    std::vector<float> imageData(imgSize * imgSize);
    for (int i = 0; i < binaryImage.rows; ++i)
    {
        for (int j = 0; j < binaryImage.cols; ++j)
        {
            if (invert)
            {
                input[i][j][1] = (1.0f - binaryImage.at<uchar>(i, j) / 255.0f);
            }
            else
            {
                input[i][j][1] = (binaryImage.at<uchar>(i, j) / 255.0f);
            }
        }
    }
}

char intToChar(int n)
{
    if (n >= 0 && n <= 9)
    {
        return '0' + n; // Convert to digit character
    }
    else if (n >= 10 && n <= 36)
    {
        return 'A' + (n - 10); // Convert to uppercase letter
    }
    else
    {
        return '\0'; // Invalid input
    }
}

int main(int argc, char **argv)
{

    const int class_count = 36;
    const int img_size = 56;

    if (argc < 2)
    {
        std::cout << "Usage: from_model [image.png]" << std::endl;
        return 1;
    }
    bool arg_binary = false;
    bool arg_invert = false;
    for (int i = 1; i < argc; i++)
    {
        if (argv[i][1] == 'b')
        {
            arg_binary = true;
        }
        if (argv[i][1] == 'i')
        {
            arg_invert = true;
        }
    }

    float res[class_count];
    float input[img_size][img_size][1];
    preprocessImage(input, argv[argc-1], img_size, arg_binary, arg_invert);

    entry(&input, &res);

    float max = 0.0;
    int index = 0;
    for (int i = 0; i < class_count; i++)
    {
        std::cout << intToChar(i) << ": " << res[i] << std::endl;
        if (res[i] > max)
        {
            max = res[i];
            index = i;
        }
    }
    std::cout << "Result: " << intToChar(index) << std::endl;
}