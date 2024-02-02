#include"linear_regression.h"
#include"logistic_regression.h"
#include"model_dealer.h"
#include <time.h>

const int size = 4;
const int number_example = 150;


int test_linear_regression()
{
    FILE *file;
    file = fopen("/home/clem/Documents/courcera_deeplearning/C_IA/model.txt", "w");
    if (file == NULL) {
        printf("Error opening the file!\n");
        return 1; // Exit the program with an error code
    }

    float **example;
    example = (float **)malloc(number_example * sizeof(float *));
    for (int i = 0; i < number_example ; i++) {
        example[i] = (float *)malloc( size * sizeof(float));
    }


    float *parameters;
    parameters = malloc((size+1) * sizeof(float *));

    float *parameters_wanted;
    parameters_wanted = malloc((size+1) * sizeof(float *));

    float* target = malloc(number_example * sizeof(float *));

    float *result;
    random_example_generator(example, number_example, size, -50.0, 50.0, file);

    parameters[0] = 2.0;
    parameters[1] = 4.0;
    parameters[2] = 2.0;
    parameters[3] = -5.0;
    parameters[4] = 2.0;

    //wanted : 
    parameters_wanted[0] = 1.0;
    parameters_wanted[1] = 7.0;
    parameters_wanted[2] = 8.0;
    parameters_wanted[3] = 0.0;
    parameters_wanted[4] = -8.0;

    for(int i = 0; i<number_example; i++)
    {
        target[i] = parameters_wanted[size];
        for(int j = 0; j< size; j++)
        {
            target[i]+=parameters_wanted[j]*example[i][j];
            // if(j%3==0) example[i][j] += 0.5;
            // if(i%3==0) example[i][j] += 0.4;
        }
    }
    float a = compute_cost(example, parameters,target, size, number_example);
    fprintf(file,"start:\n");
    print_params(parameters, size, file); 
    fprintf(file,"compute cost : %f\n", a);
    for(int i = 1; i<1500; i++)
    {
        
        gradient_descent(0.001, example, parameters,target, size, number_example);
        a = compute_cost(example, parameters,target, size, number_example);
        fprintf(file,"iteration nb : %d\n", i);
        print_params(parameters, size, file); 
        fprintf(file,"compute cost : %f\n", a);     
    }
    fclose(file);
    free(example);
    free(parameters);
    
    free(parameters_wanted);
    return 0;
}

int test_logistic_regression()
{
    FILE *file = NULL;
    file = fopen("/home/clem/Documents/courcera_deeplearning/C_IA/model.txt", "w");
    if (file == NULL) {
        printf("Error opening the file!\n");
        return 1; // Exit the program with an error code
    }

    float **example;
    example = (float **)malloc(number_example * sizeof(float *));
    for (int i = 0; i < number_example ; i++) {
        example[i] = (float *)malloc( size * sizeof(float));
    }
    random_example_generator_sigmoid(example, number_example, size, -5.0, 5.0, file);

    double* parameters;
    parameters = malloc((size+1) * sizeof(double *));

    double* parameters_wanted;
    parameters_wanted = malloc((size+1) * sizeof(double *));

    double* target = malloc(number_example * sizeof(double *));

    parameters[0] = 1.0;
    parameters[1] = 1.0;
    parameters[2] = 1.0;
    parameters[3] = 1.0;
    parameters[4] = 1.0;

    //wanted : 
    parameters_wanted[0] = 3.0;
    parameters_wanted[1] = -1.0;
    parameters_wanted[2] = 1.0;
    parameters_wanted[3] = -2.0;
    parameters_wanted[4] = 3.5;
    
    for(int i = 0; i<number_example; i++)
    {
        double a = model_sigmoid_result(example, parameters_wanted, size, i);
        printf("%lf\n", a);
        if (a>0.5)target[i] = 1;
        else target[i] = 0;
    }
    file_preparation(example, target, number_example, size, file);
    double result=0.0;
    double result2=0.0;
    double result3=0.0;
    printf("\n\n\n");
    for(int i = 0; i<number_example; i++)
    {
        result = sig_model_result(example[i], parameters, size);
        result2 = model_sigmoid_result(example, parameters, size, i);
        result3 = loss(example, parameters, target,  size, i);
        printf("%d : %lf  %lf  %lf\n", i, result, result2, result3);

    }

    double a = compute_cost_logistic_regression(example, parameters,target, size,  number_example );
    fprintf(file,"start:\n");
    print_params_double(parameters, size, file); 
    fprintf(file,"compute cost : %lf\n", a);

    for(int i = 1; i<10000; i++)
    {
        gradient_descent_logistic(0.01, example, parameters,target, size, number_example);
        a = compute_cost_logistic_regression(example, parameters, target , size,  number_example);
        fprintf(file,"iteration nb : %d\n", i);
        print_params_double(parameters, size, file); 
        fprintf(file,"compute cost : %f\n", a);     
    }
    fclose(file);
    for (int i = 0; i < number_example; i++) 
    {
        free(example[i]);  // Free each row
    }
    free(example);
    free(parameters);
    free(parameters_wanted);
    free(target);
    
    printf("free\n");

    return 0;
}

int main()
{
    printf("aa");
    int res = test_logistic_regression();
    if(res == 0) printf("test valide\n");
    return 0;
}