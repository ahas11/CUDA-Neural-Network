#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <cassert>

__device__
void activationFunction(int* n){
    if (*n < 0) {
        *n = 0;
    }
}

__device__
void meanSquaredError(int* n, int* x) {
    *x = 0;
    int target = 25;
    for (int i = 0; i < 2; i++) {
        *x += (target - n[i]) * (target - n[i]);
    }
    *x = *x / 2;
}

__device__
void meanSquaredErrorDerivative(int* outputLayer, int* errTermDer) {
    int target = 25;
    *errTermDer = *outputLayer-target ;
}



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
__device__
void activationFunctionDerivative(int* output, int* derTerm){
    if (*output < 0) {
        *derTerm = 0;
    }
    *derTerm = 1;
}

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

__device__
void calculateNewWeight(float* weights1, float* weights2, int* vecOut, int* outputErr, float* lRate) {
    for(int i = 0; i < 3; i++) {
            weights1[i] = weights1[i] - ((*lRate) * static_cast<float>(outputErr[0]) * static_cast<float>(vecOut[0]));
            weights2[i] = weights2[i] - ((*lRate) * static_cast<float>(outputErr[1]) * static_cast<float>(vecOut[1]));
    } 
}

__global__
void backPropOtoH(int* vecHidden, int* vecOutput, int* vecBias, 
                  float* vecWeightsy1, float* vecWeightsy2, float* learning_Rate) {
    int index = threadIdx.x;

    __shared__ int err_term[2];
    __shared__ int reluDerHidden[3];
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
      printf("%f\n%f\n%f\n", vecWeightsy1[0], vecWeightsy1[1], vecWeightsy1[2] );
    }
}

int main(void) {
    int x1 = 7;
    int x2 = 9;

    float wx11 = 0.65;
    float wx12 = 0.55;
    float wx13 = 0.43;
    float wx21 = 0.91;
    float wx22 = 0.80;
    float wx23 = 0.80;

    float wy11 = 0.66;
    float wy12 = 0.45;
    float wy13 = 0.75;

    float wy21 = 0.30;
    float wy22 = 0.77;
    float wy23 = 0.63;

    int y1 = 0;
    int y2 = 0;
    int y3 = 0;

    int o1 = 0;
    int o2 = 0;

    int b1 = 1;
    float learningrate = 0.1;

    std::vector<int> input_layer({x1, x2});
    std::vector<int> hidden_layer({y1, y2, y3});
    std::vector<int> output_layer({o1, o2});
    std::vector<int> bias({b1});
    std::vector<float> weightsx1({wx11, wx12, wx13});
    std::vector<float> weightsx2({wx21, wx22, wx23});
    std::vector<float> weightsy1({wy11, wy12, wy13});
    std::vector<float> weightsy2({wy21, wy22, wy23});

    int* ptr_input;
    int* ptr_hidden;
    int* ptr_output;
    int* ptr_bias;
    float* ptr_wx1;
    float* ptr_wx2;
    float* ptr_wy1;
    float* ptr_wy2;
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


    // Run the first kernel
    forwardPropItoH<<<1, 3>>>(ptr_input, ptr_hidden, ptr_bias, ptr_wx1, ptr_wx2);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Run the second kernel
    forwardPropHtoO<<<1, 2>>>(ptr_hidden, ptr_output, ptr_bias, ptr_wy1, ptr_wy2);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    backPropOtoH<<<1, 3>>>(ptr_hidden, ptr_output, ptr_bias, ptr_wy1, ptr_wy2, ptr_learningRate);

    // Wait for GPU to finish before accessing on host
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
