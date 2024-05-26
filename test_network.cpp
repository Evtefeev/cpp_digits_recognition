#include <fann.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

void readMNISTTestData(const std::string& filename, std::vector<std::vector<float>>& data, int numberOfImages, int dataOfAnImage) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.ignore(16);
        for (int i = 0; i < numberOfImages; ++i) {
            std::vector<float> image(dataOfAnImage);
            for (int j = 0; j < dataOfAnImage; ++j) {
                uint8_t pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                image[j] = pixel / 255.0;
            }
            data.push_back(image);
        }
    }
}

void readMNISTTestLabels(const std::string& filename, std::vector<int>& labels, int numberOfImages) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.ignore(8);
        for (int i = 0; i < numberOfImages; ++i) {
            uint8_t label = 0;
            file.read(reinterpret_cast<char*>(&label), sizeof(label));
            labels.push_back(label);
        }
    }
}

int main() {
    const int numberOfImages = 10000;
    const int dataOfAnImage = 28 * 28;

    std::vector<std::vector<float>> testData;
    std::vector<int> testLabels;

    readMNISTTestData("t10k-images.idx3-ubyte", testData, numberOfImages, dataOfAnImage);
    readMNISTTestLabels("t10k-labels.idx1-ubyte", testLabels, numberOfImages);

    struct fann *ann = fann_create_from_file("mnist_net.fann");

    std::cout << testData.size() << std::endl;
    std::cout << testLabels.size() << std::endl;

    int correct = 0;
    for (size_t i = 0; i < testData.size(); ++i) {
        fann_type *output = fann_run(ann, testData[i].data());
        int predicted = std::max_element(output, output + 10) - output;
        if (predicted == testLabels[i]) {
            ++correct;
        }
    }
    
    std::cout << "Accuracy: " << (float)correct / (float)testData.size() * 100.0 << "%" << std::endl;

    fann_destroy(ann);

    return 0;
}
