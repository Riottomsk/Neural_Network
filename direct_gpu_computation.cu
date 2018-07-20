#include "direct_gpu_computation.cuh"
#include <cmath>
// block width for computation - arbitrary parameter - can be changed
#define BW 512

/**
* Computes the backpropagation results of the Softmax loss for each result in a batch.
* Uses the softmax values obtained from forward propagation to compute the difference.
*
* @param label The training batch label values.
* @param num_labels The number of possible labels.
* @param batch_size The size of the trained batch.
* @param diff The resulting gradient.
*/
__global__ void SoftmaxLoss(const float *label, int num_labels, int batch_size, float *diff) // or also Cross-Entropy
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	const int label_value = static_cast<int>(label[idx]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}

/**
* Computes ceil(x / y) for integral nonnegative values.
*/
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}


void calculate_loss_with_gpu(float * d_label, int label_size, int batch_size, float * d_loss)
{
	SoftmaxLoss <<<RoundUp(batch_size, BW), BW >>>(d_label, label_size, batch_size, d_loss);
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ewm(float *a, float *b, float *c, int size)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < size)
        c[id] = a[id] * b[id];
}

void elementwise_multiplication(float *d_a, float * d_b, float * d_c, int size)
{
	int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 512;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)size/blockSize);
	ewm<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
}


// CUDA kernel. Each thread takes care of one element of c
__global__ void ew_sqrt(float *a, float *c, int size)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < size)
        c[id] = std::sqrt(a[id]);
}

void elementwise_sqrt(float *d_a, float * d_c, int size)
{
	int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 512;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)size/blockSize);
	ew_sqrt<<<gridSize, blockSize>>>(d_a, d_c, size);
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ew_add(float *a, float *c, float b, int size)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < size)
        c[id] = a[id] + b;
}

void elementwise_add(float *d_a, float * d_c, float b, int size)
{
	int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 512;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)size/blockSize);
	ew_add<<<gridSize, blockSize>>>(d_a, d_c, b, size);
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ew_devision(float *a, float *b, float *c, int size)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < size)
        c[id] = a[id] / b[id];
}

void elementwise_devision(float *d_a, float * d_b, float * d_c, int size)
{
	int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 512;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)size/blockSize);
	ew_devision<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
}

__global__ void CompLoss(const float * data, const float *label, int size, float *loss)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	loss[idx] = (data[idx] - label[idx]) / static_cast<float>(std::sqrt(std::pow(data[idx]-label[idx],2) + 1.0e-12));	
}

void comp_loss_with_gpu(float * d_data, float* d_label, int size, float * d_loss)
{
	int blockSize, gridSize;
	blockSize = BW;
	// Number of thread blocks in grid
    gridSize = (int)ceil((float)size/blockSize);	
	CompLoss <<<gridSize, blockSize>>>(d_data, d_label, size, d_loss);
}

__global__ void CompLossUnknown(const float *original, const float * data, const float *label, int batch_size, int size, float *loss)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size * batch_size)
		return;

	int batch_num = int(idx/size);
	int pix_num = idx % size;

	//if (original[idx * 4] != 0.5)
	if(original[batch_num*size*4 + pix_num*4] != 0.5)
		return;
	loss[idx] = (data[idx] - label[idx]) / static_cast<float>(std::sqrt(std::pow(data[idx]-label[idx],2) + 1.0e-12));	
}

void comp_loss_from_unknown_area(float * original, float * d_data, float* d_label, int batch_size, int size, float * d_loss)
{
	int blockSize, gridSize;
	blockSize = BW;
	// Number of thread blocks in grid
    gridSize = (int)ceil((float)(size * batch_size)/blockSize);	
	CompLossUnknown<<<gridSize, blockSize>>>(original, d_data, d_label, batch_size, size, d_loss);
}