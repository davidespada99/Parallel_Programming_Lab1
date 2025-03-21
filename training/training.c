/*
 *  training.c
 *
 *  File reused from the High-Performance Computing course at the School of 
 *  Engineering of the Autonomous University of Barcelona.
 *  Created on: January 31, 2019
 *  Last modified: Fall 2024 (academic year 24-25)
 *  Author: ecesar, asikora
 *  Modified by: Blanca Llauradó, Christian Germer
 *
 *  Description:
 *  Functions for training the neural network.
 *
 */


#include "training.h"

#include <math.h>

/**
 * @brief Initializes the initial layer of the network (input layer) 
 * with the input we want to recognize.
 *
 * @param i Index of the element from the training set that we will use.
 */

void feed_input(int i) {
    #pragma omp parallel for // 1
    for (int j = 0; j < num_neurons[0]; j++)
        lay[0].actv[j] = input[i][j];
}

/**
 * @brief Propagation of neuron values from the input layer to the rest of 
 * the network layers until obtaining a prediction (output).
 *
 * @details The input layer (layer 0) has already been initialized with the 
 * input values that we want to recognize. Thus, the outer loop (over i) 
 * iterates through all layers of the network starting from the first hidden 
 * layer (layer 1). The inner loop (over j) iterates through the neurons 
 * in layer i, calculating their activation values [lay[i].actv[j]]. The 
 * activation value of each neuron depends on the excitation of the neuron 
 * computed in the innermost loop (over k) [lay[i].z[j]]. The excitation 
 * value is initialized with the bias of the corresponding neuron [j] 
 * (lay[i].bias[j]) and is calculated by multiplying the activation values 
 * of the neurons from the previous layer (i-1) by the weights of the 
 * connections (out_weights) between the two layers. Finally, the activation 
 * value of neuron (j) is computed using the RELU (Rectified Linear Unit) 
 * function if (j) is a hidden layer, or the Sigmoid function if it is the 
 * output layer.
 */

void forward_prop() {
    
    for (int i = 1; i < num_layers; i++) {
        #pragma omp parallel for //2
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].z[j] = lay[i].bias[j];
            #pragma omp parallel for //3
            for (int k = 0; k < num_neurons[i - 1]; k++)
                lay[i].z[j] +=
                    ((lay[i - 1].out_weights[j * num_neurons[i - 1] + k]) *
                     (lay[i - 1].actv[k]));

            if (i <
                num_layers - 1)  // Relu Activation Function for Hidden Layers
                lay[i].actv[j] = ((lay[i].z[j]) < 0) ? 0 : lay[i].z[j];
            else  // Sigmoid Activation Function for Output Layer
                lay[i].actv[j] = 1 / (1 + exp(-lay[i].z[j]));
        }
    }
}

/**
 * @brief Calculates the gradient that needs to be applied to the weights of the
 * connections between neurons to correct prediction errors.
 *
 * @details It calculates two correction vectors for each layer of the network: one to
 * correct the weights of the connections from neuron (j) to the previous layer
 * (lay[i-1].dw[j]), and a second to correct the bias of each neuron in the current layer
 * (lay[i].bias[j]). There is a different treatment for the output layer (num_layers -1)
 * because this is the only case where the error is known 
 * (lay[num_layers-1].actv[j] - desired_outputs[p][j]). This can be seen in the first two loops.
 * For all hidden layers, the expected activation value of each neuron cannot be known, 
 * so an estimation is made. This calculation is performed in the nested loops that iterate 
 * through all the hidden layers (over i) and neuron by neuron (over j). It can be seen 
 * that for each case, an estimation of the activations of the neurons in the previous layer 
 * is made (lay[i-1].dactv[k] = lay[i-1].out_weights[j * num_neurons[i-1] + k] * lay[i].dz[j];),
 * except for the input layer (input layer) which is known (input image).
 *
 */

void back_prop(int p) {
    // Output Layer
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        lay[num_layers - 1].dz[j] =
            (lay[num_layers - 1].actv[j] - desired_outputs[p][j]) *
            (lay[num_layers - 1].actv[j]) * (1 - lay[num_layers - 1].actv[j]);
        lay[num_layers - 1].dbias[j] = lay[num_layers - 1].dz[j];
    }

    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] =
                (lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k]);
            lay[num_layers - 2].dactv[k] =
                lay[num_layers - 2]
                    .out_weights[j * num_neurons[num_layers - 2] + k] *
                lay[num_layers - 1].dz[j];
        }
    }

    // Hidden Layers
    for (int i = num_layers - 2; i > 0; i--) {
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].dz[j] = (lay[i].z[j] >= 0) ? lay[i].dactv[j] : 0;

            for (int k = 0; k < num_neurons[i - 1]; k++) {
                lay[i - 1].dw[j * num_neurons[i - 1] + k] =
                    lay[i].dz[j] * lay[i - 1].actv[k];

                if (i > 1)
                    lay[i - 1].dactv[k] =
                        lay[i - 1].out_weights[j * num_neurons[i - 1] + k] *
                        lay[i].dz[j];
            }
            lay[i].dbias[j] = lay[i].dz[j];
        }
    }
}

/**
 * @brief Updates the weight vectors (out_weights) and bias vectors (bias) of each layer
 * according to the calculations made in the back_prop function and the learning rate alpha.
 *
 * @see back_prop
 */

void update_weights(void) {
    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < num_neurons[i + 1]; j++)
            for (int k = 0; k < num_neurons[i]; k++)  // Update Weights
                lay[i].out_weights[j * num_neurons[i] + k] =
                    (lay[i].out_weights[j * num_neurons[i] + k]) -
                    (alpha * lay[i].dw[j * num_neurons[i] + k]);

        for (int j = 0; j < num_neurons[i]; j++)  // Update Bias
            lay[i].bias[j] = lay[i].bias[j] - (alpha * lay[i].dbias[j]);
    }
}
