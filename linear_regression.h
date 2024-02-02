#include<stdio.h>
#include <stdlib.h> 

float model_result(float* feature, float* parameters, int size); // ok
float* model_results(float** example, float* parameters, int size, int number_example); //ok

float unitary_cost(float* feature, float* parameters, int size, float label);
float compute_cost(float** example, float* parameters,float* target, int size, int number_example);

void gradient_descent(float alpha, float** example, float* parameters,float* target, int size, int number_example);

