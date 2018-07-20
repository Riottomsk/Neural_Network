#pragma once
#include <cstdio>
#include <cstdlib>

#include <chrono>
#include <random>

#include "readubyte.h"
#include "NN.h"

int LeNet()
{	
	int gpuid = 1;
	checkCudaErrors(cudaSetDevice(gpuid));

	std::string abs_path = "C:\\cinetec\\Kirill_test_projects\\NN_base\\";
	std::string FLAGS_train_images = abs_path + "train-images.idx3-ubyte";
	std::string FLAGS_train_labels = abs_path + "train-labels.idx1-ubyte";
	std::string FLAGS_test_images = abs_path + "t10k-images.idx3-ubyte";
	std::string FLAGS_test_labels = abs_path + "t10k-labels.idx1-ubyte";
	size_t width, height, channels = 1;
	

	// Open input data
	printf("Reading input data\n");

	// Read dataset sizes
	size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
	size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
	if (train_size == 0)
		return 1;

	std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
	std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

	// Read data from datasets
	if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
		return 2;
	if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
		return 3;

	printf("Done. Training dataset size: %d, Test dataset size: %d\n%d %d\n", (int)train_size, (int)test_size, train_images.size(), train_labels.size());

	bool pretrained = false;
	int batchSize = 256;
	size_t epoch = 10;
	size_t iter_num = epoch * train_size / batchSize;
	//size_t iter_num = 200;
	std::cout << "There will be " << iter_num << " iterations\n";
	
	////preparing workspace
	int byte_size = sizeof(float) * batchSize * channels * height * width;


	float *data = nullptr, *d_labels = nullptr;
	checkCudaErrors(cudaMalloc(&data, byte_size));
	checkCudaErrors(cudaMalloc(&d_labels, sizeof(float) * batchSize));
	////
	
	bool train = true;

	if (train)
	{		
		//int update_type = Neural_Network::SGD;
		int update_type = Neural_Network::ADAM;
		Neural_Network nn(gpuid, batchSize, pretrained, update_type);
		nn.adam.alpha = 1.0e-3;
		//nn.msgd.lr /= 10;
		//nn.adam.alpha /= 10;
		//nn.adam.t = 9000;
		nn.addInput(channels, height, width);		
		nn.addConv(32, 5, 2, "conv1_1");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(64, 5, 2, "conv2_1");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(1024, 7, 0, "conv3_1"); // conv -> fc
		nn.addReLU();
		nn.addDropout(0.4);
		nn.addConv(10, 1, 0, "conv3_2"); // conv -> fc
		nn.addReLU();
		nn.addSoftMax();

		nn.setWorkSpace();

		if (pretrained)
		{
			nn.load_weights("MNIST");
			std::cout << "Weights loaded. Resuming training from " << nn.adam.t << " iteration" << std::endl;
		}

		printf("Preparing dataset\n");

		// Normalize training set to be in [0,1]
		std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
		for (size_t i = 0; i < train_size * channels * width * height; ++i)
			train_images_float[i] = (float)train_images[i] / 255.0f;

		for (size_t i = 0; i < train_size; ++i)
			train_labels_float[i] = (float)train_labels[i];

		printf("Training...\n");				

		std::vector<float> label(batchSize * 10, 0);
		
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int iter = 0; iter < iter_num; ++iter)
		{
			// Train
			int imageid = iter % (train_size / batchSize);

			// Prepare current batch on device
			checkCudaErrors(cudaMemcpyAsync(data, &train_images_float[imageid * batchSize * width*height*channels],
				byte_size, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * batchSize],
				sizeof(float) * batchSize, cudaMemcpyHostToDevice));
			if ((iter) % 50 == 0 )
				nn.train(data,d_labels,iter,true);
			else				
				nn.train(data, d_labels, iter);
		}
		//checkCudaErrors(cudaDeviceSynchronize());
		auto t2 = std::chrono::high_resolution_clock::now();

		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iter_num);

		//return 0;

		nn.save_weights("MNIST");
	}

	printf("Testing...\n");

	Neural_Network nn1(0, 1, true,Neural_Network::ONLY_INFERENCE);
	nn1.addInput(channels, height, width);
	nn1.addConv(32, 5, 2, "conv1_1");
	nn1.addReLU();
	nn1.addPool(2, 2);
	nn1.addConv(64, 5, 2, "conv2_1");
	nn1.addReLU();
	nn1.addPool(2, 2);
	nn1.addConv(1024, 7, 0, "conv3_1"); // conv -> fc
	nn1.addReLU();
	nn1.addDropout(0.4);
	nn1.addConv(10, 1, 0, "conv3_2"); // conv -> fc
	nn1.addReLU();
	nn1.addSoftMax();

	nn1.setWorkSpace();
	nn1.load_weights("MNIST");

	float classification_error = 1.0f;

	int classifications = 10000;
	int num_errors = 0;
	for (int i = 0; i < classifications; ++i)
	{
		std::vector<float> test_image_float(width * height);
		// Normalize image to be in [0,1]
		for (int j = 0; j < width * height; ++j)
			test_image_float[j] = (float)test_images[i * width*height*channels + j] / 255.0f;
		checkCudaErrors(cudaMemcpyAsync(data, &test_image_float[0],
			sizeof(float) * channels * width * height, cudaMemcpyHostToDevice));
		nn1.inference(data);

		auto res = nn1.getResult();
		// Perform classification
		std::vector<float> class_vec(10);

		// Copy back result
		checkCudaErrors(cudaMemcpy(&class_vec[0], res, sizeof(float) * 10, cudaMemcpyDeviceToHost));

		// Determine classification according to maximal response
		int chosen = 0;
		for (int id = 1; id < 10; ++id)
		{
			if (class_vec[chosen] < class_vec[id]) chosen = id;
		}

		if (chosen != test_labels[i])
			++num_errors;
	}
	classification_error = (float)num_errors / (float)classifications;
	printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);

	checkCudaErrors(cudaFree(data));
	checkCudaErrors(cudaFree(d_labels));
}