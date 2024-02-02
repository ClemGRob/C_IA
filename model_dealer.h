#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void print_params(float* params, int size, FILE *file);
void print_params_double(double* params, int size,FILE *file);
void random_example_generator(float **example, int number_example, int size, float min, float max, FILE *file);
void round_training_set(float **example, int number_example, int size, float min, float max);
void random_example_generator_sigmoid(float **example, int number_example, int size, float min, float max, FILE *file);
void file_preparation(float **example, double* target, int number_example, int size,FILE *file);