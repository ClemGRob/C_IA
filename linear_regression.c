#include"linear_regression.h"
#include <math.h>
/*linear regression*/
float model_result(float* feature, float* parameters, int size)
{
    float results = parameters[size]; 
    for(int i = 0; i<size; i++)
    {
        results+=feature[i]*parameters[i];
    }
    return results;
}

float* model_results(float** example, float* parameters, int size, int number_example)
{
    float* results = malloc(number_example*sizeof(float));
    
    for(int i = 0; i<number_example; i++)
    {
        results[i]=model_result(example[i], parameters, size);
    }
    return results;
}




/*cost */
float unitary_cost(float* feature, float* parameters, int size, float target)
{
    float a = model_result(feature, parameters, size) - target;
    return a;
}

float compute_cost(float** example, float* parameters,float* target, int size, int number_example)
{
    float results = 0.0;
    
    for(int i = 0; i<number_example; i++)
    {
        results+=pow(unitary_cost(example[i], parameters, size, target[i]), 2);
    }
    return results/(2*number_example);
}

void gradient_descent(float alpha, float** example, float* parameters,float* target, int size, int number_example)
{
    float* results = malloc(size*sizeof(float));

    for(int i = 0; i < size; i++)
    {
        results[i] = 0;
        for(int j = 0; j < number_example; j++)
        {
            results[i] +=unitary_cost(example[j], parameters, size, target[j])*example[j][i];
        }
        results[i] = parameters[i] - alpha*results[i]/number_example;
    }
    
    // last param
    results[size] = 0;
    for(int j = 0; j < number_example; j++)
    {
        results[size] +=unitary_cost(example[j], parameters, size, target[j]);
    }

    results[size] = parameters[size] - alpha*results[size]/number_example;
    for(int i = 0; i < size+1; i++)
    {
        parameters[i]=results[i];
    }
    free(results);
    
}

