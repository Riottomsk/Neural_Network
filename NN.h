#pragma once
#include <cstdio>
#include <cstdlib>

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <sstream>
#include <string>
#include <cassert>
#include <iostream>
//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
	system("pause");												   \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status << " " << cudaGetErrorString((cudaError_t)status);                            \
	/*std::cerr << "\n" << cudaGetErrorString((cudaError_t)status) << std::endl;*/		   \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


///////////////////////////////////////////////////////////////////////////////////////////

void show_some_data(float* d_output, int size); // debug function

// this class becoming too big - TODO: use PointerToImplementation pattern or split to several classes...
class Neural_Network
{
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;

	int m_gpuid;
	int m_batchSize;
	size_t m_workspaceSize;
	void *cudnn_workspace = nullptr;

	float learning_rate;
	bool pretrained = false;
	float *d_onevec = nullptr;
	size_t gpu_memory_allocated = 0;
	unsigned update_type = SGD;
	unsigned loss_type = CROSS_ENTROPY;


	// Disable copying
	Neural_Network& operator=(const Neural_Network&) = delete;
	Neural_Network(const Neural_Network&) = delete;

	struct Layer;
	std::vector<Layer *> layers; // can be switched to shared_ptr
	
		
	void forwardPropagation(float * data);
	void backPropagation();
	double calculate_loss(float * d_label);

	void show_weights();

	

public:

	Neural_Network(int gpuid, int batch_size, bool pretrained = false, int update_type_ = SGD, int loss_type_ = CROSS_ENTROPY);
	void addInput(int channels, int height, int width, std::string name = "");
	void addConv(int channels, int kernel_size, int pad, std::string name = ""); // square convolution
	void addPool(int size, int stride, int pad = 0, std::string name = ""); // MaxPool
	void addReLU(std::string name = "");
	void addFC(int channels, std::string name = "");
	void addDropout(float dropout_ratio, std::string name = "");
	void addSoftMax(std::string name = "");

	// this layer will be automatically linked to the mirrored pool layer
	// so be sure that you do not including more unpool layers than you have pools
	// TODO : name linkage
	void addUnpool(std::string name = "");

	//// Main functions
	void train(float * data, float * label, size_t iter, bool visualize = false, bool backup = false);
	void inference(float * d_data);
	////


	void save_weights(const char* nn_name);
	void load_weights(const char* nn_name); // rename to ~load_model

	void load_only_weights(const char* nn_name);
	
	void setWorkSpace();
	float * getResult();

	// following procedures are meant to br used with ONLY_INFERENCE
	void clearTensors();
	void redefineTensors(int height, int width, int channels);
	
	void getOutputDimensions(int &height, int &width, int &classes);

	// you can specify the pathto the json file with class names, if not only numbers will be shown
	std::vector<float> interpret_softmax_result(const char* path= nullptr, bool show = true);
	

	// if you see this function - DELETE IT IMMEDIATELY
	void convert_VGG16_weights();

	void load_specified_weights(float *data, size_t size, int layer_num);
	
	// public parameters

	struct
	{			
		float alpha = 0.001; // value from ADAM paper
		//float alpha = 0.00001; // value from DIM paper
		float alpha_t; // this will be calcualted at each step
		float beta1 = 0.9;
		float beta2 = 0.999;
		float epsilon = 1.0e-8;
		float epsilon_t;
		size_t t = 0;
	} adam;

	struct
	{
		float momentum = 0.9; // momentum
		float L2 = 0.0005; // L2 or weight decay
		float lr = 0.01; // learning rate
		size_t t = 0;
	} msgd;



	enum
	{
		SGD = 0, // Stochastic Gradient Descent
		mSGD, // mini-batch Gradient Descent with momentum
		ADAM,
		ONLY_INFERENCE
	} update;

	enum
	{		
		CROSS_ENTROPY = 0, // for classification mainly
		COMPOSITIONAL
	} loss;

	// 0 - least debug info, 
	// debug > 0 - total allocated memory,
	// debug > 1 - workspace memory on each conv layer
	// debug > 2 - memory allocated in each layer
	int DEBUG_LEVEL = 1; 

	size_t it = 0; // TODO: iteration counter - better here than separate in updates

	~Neural_Network();
private:

	enum layer_type
	{		
		CONV = 0,
		POOL,
		RELU,
		INPUT,
		FC,
		SOFTMAX,
		DROPOUT,
		UNPOOL
	};
	
	struct Layer
	{
		std::string name;
		int type;
		Layer(std::string name_ = "");
		virtual void set_forward_propagation(Layer *src, Neural_Network *nn) {}
		virtual void forward_propagation(Layer *src, Neural_Network *nn) {}
		virtual void back_propagation(Layer *dst, Neural_Network *nn) {}
			
		//int in_channels, in_width, in_height;
		int out_channels, out_width, out_height;
		cudnnTensorDescriptor_t tensor; // output tensor of the layer
		float *data = nullptr;			// output data of the layer
		void define_tensor(Neural_Network *nn);	
		// for ONLY_INFERENCE
		void redefine_tensor(Neural_Network *nn);
		void link_tensors(Layer *src, Neural_Network *nn);
		float *diff_data = nullptr;

		void clear();
		virtual void redefine(Layer *src, Neural_Network *nn) {}

		virtual ~Layer();
	};

	struct Input_layer : Layer
	{
		Input_layer(int height_, int width_, int channels_, Neural_Network *nn, std::string name = "");
		void input_data(float *data_, Neural_Network *nn);
		void redefine(int height_, int width_, int channels_, Neural_Network *nn);
		~Input_layer() {}
	};

	struct Conv_layer : Layer
	{		
		cudnnConvolutionDescriptor_t desc;
		cudnnConvolutionFwdAlgo_t fw_algo;
		cudnnConvolutionBwdFilterAlgo_t bwF_algo;
		cudnnConvolutionBwdDataAlgo_t bwD_algo;

		int pad = 0;

		// temporary here
		int in_channels, in_width, in_height;

		int kernel_size;		
		std::vector<float> kernel_value; // actual kernel / weights - that would be trained
		cudnnFilterDescriptor_t filterDesc; // == kernelDesc
		float *filter_data = nullptr; // filter data pointer
		float *gfdata = nullptr; // filter gradient data pointer

		std::vector<float> bias_value;
		cudnnTensorDescriptor_t bias_tensor;
		float * d_bias = nullptr;
		float * d_gbias = nullptr;
	
		Conv_layer(int out_channels_,int kernel_size_, int pad_, std::string name = "");
		void set_forward_propagation(Layer *src, Neural_Network *nn);		
		virtual void forward_propagation(Layer *src, Neural_Network *nn);
		void back_propagation(Layer *dst, Neural_Network *nn);

		void load_weights();
		// TODO: fix
		void convert_and_load();
		void save_weights(const char * nn_name);

		void redefine(Layer *src, Neural_Network *nn);

		void load_specified_weights(float *data, size_t size);
		void show_weights();


		struct // Adam parameters
		{
			//size_t t = 0;
			float * d_m_f = nullptr; // Adam moment parameter for filter
			float * d_v_f = nullptr;

			float * d_m_b = nullptr; // Adam moment parameter for bias
			float * d_v_b = nullptr;

			float * tmp_f = nullptr; // for g*g usage and other uses
			float * tmp_b = nullptr;

		} adam;

		struct // mSGD parameters
		{
			float * d_v_f = nullptr;
			float * d_v_b = nullptr;
		} msgd;

		~Conv_layer();
	};

	struct Pool_layer : Layer // MaxPool
	{
		int size, stride;
		cudnnPoolingDescriptor_t desc;
		Pool_layer(int size_, int stride_, int pad_ = 0, std::string name = ""); // square stride assumed
		void set_forward_propagation(Layer *src, Neural_Network *nn);
		void forward_propagation(Layer *src, Neural_Network *nn);
		void back_propagation(Layer *dst, Neural_Network *nn);

		void redefine(Layer *src, Neural_Network *nn);

		~Pool_layer();
	};

	struct Unpool_layer : Layer
	{
		int size, stride;
		Pool_layer *pool_ref;
		Layer * pre_pool_ref;
		cudnnPoolingDescriptor_t desc;
		Unpool_layer(Pool_layer *src, Layer * pre_pool_ref_, std::string name = ""); // square stride assumed
		void set_forward_propagation(Layer *src, Neural_Network *nn);
		void forward_propagation(Layer *src, Neural_Network *nn);
		void back_propagation(Layer *dst, Neural_Network *nn);

		//void redefine(Layer *src, Neural_Network *nn);

		~Unpool_layer() {}
	};

	struct RELU_layer : Layer
	{
		cudnnActivationDescriptor_t relu_desc;
		RELU_layer(std::string name = "");
		void set_forward_propagation(Layer *src, Neural_Network *nn);
		void forward_propagation(Layer *src, Neural_Network *nn);
		void back_propagation(Layer *dst, Neural_Network *nn);

		void redefine(Layer *src, Neural_Network *nn);

		~RELU_layer();
	};

	// unfinished - you should probably use conv instead of FC, since there is an easy FC->CONV transformation
	struct FC_layer : Layer
	{
		// temporary here
		int in_channels, in_width, in_height;
		std::vector<float> neurons; // initial kernel / weights
		float *neurons_data = nullptr; // filter data pointer
		std::vector<float> bias;
		float *d_bias;
		void set_forward_propagation(Layer *src, Neural_Network *nn);
		FC_layer(int out_channels_, std::string name = "") : Layer(name) { out_channels = out_channels_; type = FC; }
		void forward_propagation(Layer *src, Neural_Network *nn);
		void convert_and_load();
		void load_weights();
		~FC_layer();
	};

	struct SoftMax_layer : Layer
	{
		SoftMax_layer(std::string name = "") : Layer(name) { type = SOFTMAX; }
		void set_forward_propagation(Layer *src, Neural_Network *nn);
		void forward_propagation(Layer *src, Neural_Network *nn);
		void back_propagation(Layer *dst, Neural_Network *nn);

		void redefine(Layer *src, Neural_Network *nn);

		~SoftMax_layer() {}
	};

	struct Dropout_layer : Layer
	{
		float dropout;
		cudnnDropoutDescriptor_t desc;
		size_t reservedSpace_size;
		void * d_reservedSpace = nullptr;
		size_t statesSpace_size;
		void * d_statesSpace = nullptr;

		Dropout_layer(float dropout_ratio, std::string name = "");
		void set_forward_propagation(Layer *src, Neural_Network *nn);
		void forward_propagation(Layer *src, Neural_Network *nn);
		void back_propagation(Layer *dst, Neural_Network *nn);
		
		~Dropout_layer();
	};


};

