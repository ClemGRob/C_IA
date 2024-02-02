double model_sigmoid_result(float** example, double* parameters, int size, int number_example);
double loss(float** example, double* parameters, double* target,int size,int example_index);
double compute_cost_logistic_regression(float** example, double* parameters,double* target, int size, int number_example);
void gradient_descent_logistic(float alpha, float** example, double* parameters,double* target, int size, int number_example);
double sig_model_result(float* feature, double* parameters, int size);