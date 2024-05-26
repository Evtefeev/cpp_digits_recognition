#include <iostream>
#include <vector>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

int main() {
    // Initialize the ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXPredictionExample");

    try {
        // Create a session options object
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Path to the ONNX model file
        const char* model_path = "model.onnx";

        // Create a session object from the ONNX model file
        Ort::Session session(env, model_path, session_options);

        // Get the number of model input nodes
        size_t num_input_nodes = session.GetInputCount();

        // Assuming there's only one input node in this example
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

        // Prepare input data (assuming input is a float tensor)
        std::vector<float> input_data = { /* your input data */ };

        // Create input tensor object
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            env, input_data.data(), input_data.size(), input_tensor_info.GetShape().data(), 
            input_tensor_info.GetShape().size()
        );

        // Run the prediction
        const char* input_names[] = { session.GetInputName(0, env) };
        const char* output_names[] = { session.GetOutputName(0, env) };
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
        );

        // Assuming there's only one output tensor in this example
        Ort::TypeInfo output_type_info = output_tensors[0].GetTypeInfo();
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

        // Retrieve output data (assuming output is a float tensor)
        std::vector<float> output_data(output_tensor_info.GetElementCount());
        output_tensors[0].GetTensorMutableData<float>() = output_data.data();

        // Output the prediction result
        for (float val : output_data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

    } catch (const Ort::Exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}
