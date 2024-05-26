#include <fann.h>
#include <fann_data.h>
#include <iostream>

int main() {
    const unsigned int num_input = 28 * 28;
    const unsigned int num_output = 36;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 512;
    const float desired_error = (const float) 0.001;
    const unsigned int max_epochs = 100;
    const unsigned int epochs_between_reports = 10;

    struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);

    fann_train_on_file(ann, "Chars74K_train_data.fann", max_epochs, epochs_between_reports, desired_error);

    fann_save(ann, "Chars74K_net.fann");

    fann_destroy(ann);

    return 0;
}
