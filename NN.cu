#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <cassert>

//ReLU activation function
__device__
void activationFunction(int* n){
    if (*n < 0) {
        *n = 0;
    }
}

//Mean Squared Error function
__device__
void meanSquaredError(int* n, int* x) {
    *x = 0;
    int target = 25;
    for (int i = 0; i < 2; i++) {
        *x += (target - n[i]) * (target - n[i]);
    }
    *x = *x / 2;
}

//Mean Squared Error Derivative to calculate backpropagation
__device__
void meanSquaredErrorDerivative(int* outputLayer, int* errTermDer) {
    int target = 25;
    *errTermDer = *outputLayer-target ;
}


//Forward propagation from Input Layer to Hidden Layer
__global__
void forwardPropItoH(int* vecInput, int* vecHidden, int* vecBias, 
                     float* vecWeightsx1, float* vecWeightsx2) {
    int index = threadIdx.x;

    if (index < 3) {    
        vecHidden[index] = (vecInput[0] * vecWeightsx1[index]) + 
                           (vecInput[1] * vecWeightsx2[index]) + vecBias[0];
        activationFunction(&vecHidden[index]);
    }
}

//Forward propagation from Hidden Layer to Output Layer
__global__
void forwardPropHtoO(int* vecHidden, int* vecOutput, int* vecBias, 
                     float* vecWeightsy1, float* vecWeightsy2) {
    int index = threadIdx.x;

    if (index < 2) {
        if (index == 0) {
            vecOutput[index] = (vecHidden[0] * vecWeightsy1[0]) + 
                               (vecHidden[1] * vecWeightsy1[1]) + 
                               (vecHidden[2] * vecWeightsy1[2]) + 
                                vecBias[0];
            activationFunction(&vecOutput[index]);
        }
        if (index == 1) {
            vecOutput[index] = (vecHidden[0] * vecWeightsy2[0]) + 
                               (vecHidden[1] * vecWeightsy2[1]) + 
                               (vecHidden[2] * vecWeightsy2[2]) + 
                                vecBias[0];
            activationFunction(&vecOutput[index]);
        }
    }
}

//Activation Function Derivative to calculate backpropagation
__device__
void activationFunctionDerivative(int* output, int* derTerm){
    if (*output < 0) {
        *derTerm = 0;
    }
    *derTerm = 1;
}

//Calculating the error for each node in the Hidden Layer
__device__
void backPropErrorH(float* weights1, float* weights2, int* errTerm, int* reluder_Hidden, int* outErrHidden) {
  int sumCounter = 0;
  for(int i = 0; i<3; i++){
    for(int j = 0; j<2; j++){
        sumCounter = (errTerm[j] * weights1[i]) + (errTerm[j] * weights2[i]);
    }
    // activationFunctionDerivative()
    outErrHidden[i] =  reluder_Hidden[i] * sumCounter;
    sumCounter = 0;
  }
}

//Calculating and adjusting the weights based on the weight change formula
__device__
void calculateNewWeight(float* weights1, float* weights2, int* vecOut, int* outputErr, float* lRate) {
    for(int i = 0; i < 3; i++) {
            weights1[i] = weights1[i] - ((*lRate) * static_cast<float>(outputErr[0]) * static_cast<float>(vecOut[0]));
            weights2[i] = weights2[i] - ((*lRate) * static_cast<float>(outputErr[1]) * static_cast<float>(vecOut[1]));
    } 
}

//Backpropagation from Output Layer to Hidden Layer
__global__
void backPropOtoH(int* vecHidden, int* vecOutput, int* vecBias, 
                  float* vecWeightsy1, float* vecWeightsy2, float* learning_Rate) {
    int index = threadIdx.x;

    //Array to hold the errors of each output node
    __shared__ int err_term[2];

    //Array to hold activation layer derivative values of the nodes in the Hidden Layer
    __shared__ int reluDerHidden[3];

    //Array to hold total error values of the nodes in the Hidden Layer
    __shared__ int total_err_per_hidden[3];

    if(index<3){
      activationFunctionDerivative(&vecHidden[index],&reluDerHidden[index]);
    }

    if(index==0){
      meanSquaredErrorDerivative(&vecOutput[index],&err_term[index]);
    }
    if(index==1){
      meanSquaredErrorDerivative(&vecOutput[index],&err_term[index]);
    }
    if(index==2){
      backPropErrorH(vecWeightsy1, vecWeightsy2, err_term, reluDerHidden, total_err_per_hidden);
      calculateNewWeight(vecWeightsy1, vecWeightsy2, vecOutput, err_term,learning_Rate);
    }
}

int main(void) {
    //input layer values
    int x1 = 7;
    int x2 = 9;

    //hidden layer initial values
    int y1 = 0;
    int y2 = 0;
    int y3 = 0;

    //output layer initial values
    int o1 = 0;
    int o2 = 0;

    //weights connecting each node in the input layer to each node in the hidden layer
    float wx11 = 0.65;
    float wx12 = 0.55;
    float wx13 = 0.43;
    float wx21 = 0.91;
    float wx22 = 0.80;
    float wx23 = 0.80;

    //weights connecting each node in the hidden layer to each node in the output layer
    float wy11 = 0.66;
    float wy12 = 0.45;
    float wy13 = 0.75;
    float wy21 = 0.30;
    float wy22 = 0.77;
    float wy23 = 0.63;

    //bias node values
    int b1 = 1;

    //learning rate value for back propagation
    float learningrate = 0.1;

    //input layer vector
    std::vector<int> input_layer({x1, x2});

    //hidden layer vector
    std::vector<int> hidden_layer({y1, y2, y3});

    //output layer vector
    std::vector<int> output_layer({o1, o2});

    //bias vector
    std::vector<int> bias({b1});

    //vector of  weights connecting first node node in the input layer to each node in the hidden layer
    std::vector<float> weightsx1({wx11, wx12, wx13});

    //vector of  weights connecting second node node in the input layer to each node in the hidden layer
    std::vector<float> weightsx2({wx21, wx22, wx23});

    //vector of  weights connecting each node in hidden layer to the first output node in the output layer
    std::vector<float> weightsy1({wy11, wy12, wy13});

    //vector of weights connecting each node in hidden layer to the second output node 2 in the output layer
    std::vector<float> weightsy2({wy21, wy22, wy23});

    //int pointer to input layer vector
    int* ptr_input;
    
    //int pointer to hidden layer vector
    int* ptr_hidden;
    
    //int pointer to output layer vector
    int* ptr_output;
    
    //int pointer to bias node
    int* ptr_bias;
    
    //float pointer to weightsx1 vector
    float* ptr_wx1;

    //float pointer to weightsx2 vector
    float* ptr_wx2;

    //float pointer to weightsy1 vector
    float* ptr_wy1;
    
    //float pointer to weightsy2 vector
    float* ptr_wy2;

    //float pointer to learning rate
    float* ptr_learningRate;


    cudaMallocManaged(&ptr_input, input_layer.size() * sizeof(int));
    cudaMallocManaged(&ptr_hidden, hidden_layer.size() * sizeof(int));
    cudaMallocManaged(&ptr_output, output_layer.size() * sizeof(int));
    cudaMallocManaged(&ptr_bias, bias.size() * sizeof(int));

    cudaMallocManaged(&ptr_wx1, weightsx1.size() * sizeof(float));
    cudaMallocManaged(&ptr_wx2, weightsx2.size() * sizeof(float));
    cudaMallocManaged(&ptr_wy1, weightsy1.size() * sizeof(float));
    cudaMallocManaged(&ptr_wy2, weightsy2.size() * sizeof(float));

    cudaMallocManaged(&ptr_learningRate, sizeof(float));


    cudaMemcpy(ptr_input, input_layer.data(), input_layer.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_hidden, hidden_layer.data(), hidden_layer.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_output, output_layer.data(), output_layer.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_bias, bias.data(), bias.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(ptr_wx1, weightsx1.data(), weightsx1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_wx2, weightsx2.data(), weightsx2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_wy1, weightsy1.data(), weightsy1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr_wy2, weightsy2.data(), weightsy2.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(ptr_learningRate, &learningrate,sizeof(float), cudaMemcpyHostToDevice);


    // Running the forward propagation from input layer to hidden layer
    forwardPropItoH<<<1, 3>>>(ptr_input, ptr_hidden, ptr_bias, ptr_wx1, ptr_wx2);

    // Waiting for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Running the forward propagation from hidden layer to output layer
    forwardPropHtoO<<<1, 2>>>(ptr_hidden, ptr_output, ptr_bias, ptr_wy1, ptr_wy2);

    // Waiting for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Running the backward propagation from output layer to hidden layer
    backPropOtoH<<<1, 3>>>(ptr_hidden, ptr_output, ptr_bias, ptr_wy1, ptr_wy2, ptr_learningRate);

    // Waiting for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaFree(ptr_input);
    cudaFree(ptr_hidden);
    cudaFree(ptr_output);
    cudaFree(ptr_bias);
    cudaFree(ptr_wx1);
    cudaFree(ptr_wx2);
    cudaFree(ptr_wy1);
    cudaFree(ptr_wy2);

    return 0;
}

