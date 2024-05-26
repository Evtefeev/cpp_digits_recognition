#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <png.h>
#include <opencv2/opencv.hpp>
#include <sstream>

// Namespace for convenience
namespace fs = std::filesystem;

// Function to read PNG image and return normalized pixel data resized to 28x28
std::vector<float> readPngImageAndResize(const std::string &filename)
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        std::cerr << "Error: Unable to open image " << filename << std::endl;
        return {};
    }

    // Resize the image to 28x28
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(28, 28));

    // Normalize pixel values and store them in a vector
    std::vector<float> image_data;
    image_data.reserve(28 * 28 * resizedImage.channels());
    for (int y = 0; y < resizedImage.rows; ++y)
    {
        for (int x = 0; x < resizedImage.cols * resizedImage.channels(); ++x)
        {
            image_data.push_back(resizedImage.at<cv::Vec4b>(y, x)[x] / 255.0f);
        }
    }

    return image_data;
}

void readChars74KData(const std::string &baseFolder, std::vector<std::vector<float>> &data, int numberOfImages, int dataOfAnImage)
{
    for (int sampleIdx = 1; sampleIdx <= 36; ++sampleIdx)
    {
        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << sampleIdx;
        std::string folderName = baseFolder + "/Sample" + oss.str();
        std::cout << folderName << std::endl;

        for (const auto &entry : fs::directory_iterator(folderName))
        {
            if (entry.path().extension() == ".png")
            {
                std::vector<float> image = readPngImageAndResize(entry.path().string());
                if (!image.empty())
                {
                    data.push_back(image);
                }
                else
                {
                    std::cerr << "Failed to read image: " << entry.path() << std::endl;
                }
            }
        }
    }
}

void readChars74KLabels(const std::string &filename, std::vector<int> &labels, int numberOfImages)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    labels.push_back(1);
    std::string buf;
    std::getline(file, buf);

    for (int i = 1; i < numberOfImages; ++i)
    {
        std::getline(file, buf);
        uint8_t label = atoi(buf.c_str());
        labels.push_back(label);
    }
}

void readMNISTData(const std::string &filename, std::vector<std::vector<float>> &data, int numberOfImages, int dataOfAnImage)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Skip the header
    file.ignore(16);

    for (int i = 0; i < numberOfImages; ++i)
    {
        std::vector<float> image(dataOfAnImage); // Initialize the vector with the required size
        for (int j = 0; j < dataOfAnImage; ++j)
        {
            uint8_t pixel = 0;
            if (!file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel)))
            {
                std::cerr << "Error: Unable to read pixel data for image " << i << std::endl;
                return;
            }
            image[j] = pixel / 255.0f; // Normalize pixel value
        }
        data.push_back(image);
    }
}

void readMNISTLabels(const std::string &filename, std::vector<int> &labels, int numberOfImages)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Skip the header
    file.ignore(8);

    for (int i = 0; i < numberOfImages; ++i)
    {
        uint8_t label = 0;
        if (!file.read(reinterpret_cast<char *>(&label), sizeof(label)))
        {
            std::cerr << "Error: Unable to read label data for image " << i << std::endl;
            return;
        }
        labels.push_back(label);
    }
}

void saveFANNData(const std::string &filename, const std::vector<std::vector<float>> &data, const std::vector<int> &labels, int class_count)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    file << data.size() << " " << data[0].size() << " " << class_count << "\n"; // Number of samples, inputs, outputs
    for (size_t i = 0; i < data.size(); ++i)
    {
        for (const auto &value : data[i])
        {
            file << value << " ";
        }
        file << "\n";
        for (int j = 0; j < class_count; ++j)
        {
            file << (j == labels[i] ? 1.0f : 0.0f) << " ";
        }
        file << "\n";
    }
}

int main()
{
    const int numberOfImages = 1016 * 36;
    const int dataOfAnImage = 28 * 28;

    std::vector<std::vector<float>> trainData;
    std::vector<int> trainLabels;

    // readMNISTData("./train-images.idx3-ubyte", trainData, numberOfImages, dataOfAnImage);
    // readMNISTLabels("./train-labels.idx1-ubyte", trainLabels, numberOfImages);
    std::cout << "readData" << std::endl;
    readChars74KData("English/Fnt/", trainData, numberOfImages, dataOfAnImage);
    std::cout << "readLabels" << std::endl;
    readChars74KLabels("list_English_Fnt.m", trainLabels, numberOfImages);
    std::cout << "saveData" << std::endl;
    saveFANNData("./Chars74K_train_data.fann", trainData, trainLabels, 36);

    return 0;
}
