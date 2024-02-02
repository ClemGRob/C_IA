#include "model_dealer.h"

void print_params(float* params, int size,FILE *file)
{
    // printf("new params : ");
    for(int i = 0; i<size+1; i++)
    {
        // printf("%f ", params[i]);
        fprintf(file, "%f ", params[i]);
    }
    fprintf(file,"\n");
}

void print_params_double(double* params, int size,FILE *file)
{
    // printf("new params : ");
    for(int i = 0; i<size+1; i++)
    {
        // printf("%f ", params[i]);
        fprintf(file, "%f ", params[i]);
    }
    fprintf(file,"\n");
}

void file_preparation(float **example, double* target, int number_example, int size,FILE *file)
{
    for(int i = 0; i< number_example; i++)
    {
        for(int j = 0; j < size; j++)
        {   
            fprintf(file,"%lf ", example[i][j]);

        }
        fprintf(file,"%lf\n",target[i]);
    }
    fprintf(file,"\n");
}

void random_example_generator(float **example, int number_example, int size, float min, float max, FILE *file)
{
    srand((unsigned int)time(NULL));
    fprintf(file,"traning examples : \n");
    for(int i = 0; i< number_example; i++)
    {
        for(int j = 0; j < size; j++)
        {   
            int randomInt = rand();
            example[i][j]=((float)randomInt / RAND_MAX) * (max - (min)) + (min);
            fprintf(file,"%f ", example[i][j]);

        }
        fprintf(file,"\n");
    }
    fprintf(file,"\n");
}

void random_example_generator_sigmoid(float **example, int number_example, int size, float min, float max, FILE *file)
{
    srand((unsigned int)time(NULL));
    for(int i = 0; i< number_example; i++)
    {
        for(int j = 0; j < size; j++)
        {   
            int randomInt = rand();
            float rang_generated = ((float)randomInt / RAND_MAX) * (max - (min)) + (min);
            example[i][j] = rang_generated;

        }
    }
}

//TO DO
void round_training_set(float **example, int number_example, int size, float min, float max)
{
    for(int i = 0; i< number_example; i++)
    {
        for(int j = 0; j < size; j++)
        {   
            int randomInt = rand();
            example[i][j]=((float)randomInt / RAND_MAX) * (max - (min)) + (min);

        }
    }
}