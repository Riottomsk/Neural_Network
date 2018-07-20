void calculate_loss_with_gpu(float * d_label, int label_size, int batch_size, float * d_loss);
void elementwise_multiplication(float *d_a, float * d_b, float * d_c, int size);
void elementwise_sqrt(float *d_a, float * d_c, int size);
void elementwise_add(float *d_a, float * d_c, float b, int size);
void elementwise_devision(float *d_a, float * d_b, float * d_c, int size);

void comp_loss_with_gpu(float * d_data, float* d_label, int size, float * d_loss);
void comp_loss_from_unknown_area(float * original, float * d_data, float* d_label, int batch_size, int size, float * d_loss);