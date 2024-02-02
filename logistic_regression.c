#include "logistic_regression.h"
#include <math.h>

double sig_model_result(float* feature, double* parameters, int size) // ok
/*
    calcul dot
*/
{
    double results = parameters[size]; 
    for(int i = 0; i<size; i++)
    {
        results+=(double)feature[i]*parameters[i];
    }
    return results;
}


double model_sigmoid_result(float** example, double* parameters, int size, int example_index) //ok
/*
    fw,b(x(i)) or g(z) or g(w.x(i)+b)
*/
{
    double results = 1/(1+exp(-1*sig_model_result(example[example_index], parameters, size)));
    // printf("%f\n", results);
    return results;
}

double loss(float** example, double* parameters, double* target, int size,  int example_index)//ok
{
    double fx = model_sigmoid_result(example, parameters, size, example_index);
    // double a =  -1*log(fx);
    // double b = -1*log(1-fx);
    if(target[example_index] == 1.0)return -1*log(fx);
    return -1*log(1-fx);
}

double compute_cost_logistic_regression(float** example, double* parameters,double* target, int size, int number_example)
{
    double ret = 0.0;
    for(int i =0; i<number_example; i++)
    {
        ret += loss(example, parameters,  target, size, i);
    }
    return ret/number_example;
}
/*cost */

void gradient_descent_logistic(float alpha, float** example, double* parameters,double* target, int size, int number_example)
{
    double* results = malloc(size*sizeof(double));
    
    for(int i = 0; i < size; i++)
    {
        
        results[i] = 0;
        
        for(int j = 0; j < number_example; j++)
        {
            double sig_res = model_sigmoid_result(example, parameters, size, j);
            results[i] += (model_sigmoid_result(example, parameters, size, j)-target[j])*example[j][i];
            
        }
        results[i] = parameters[i] - alpha*results[i]/number_example;
        
    }
    // last param
    results[size] = 0;
    for(int j = 0; j < number_example; j++)
    {
        
        results[size] += (model_sigmoid_result(example, parameters, size, j)-target[j]);
    }

    results[size] = parameters[size] - alpha*results[size]/number_example;
    
    for(int i = 0; i < size+1; i++)
    {
        parameters[i]=results[i];
    }
    free(results);
}