#include "NN.h"
#include "Image_preparation.h"
#include "CIFAR-10.h"

void Cifar_train()
{
	std::cout << "CIFAR-10 trainig\n";
	int gpuid = 0;
	checkCudaErrors(cudaSetDevice(gpuid));
	std::string train_path = "E:\\Kirill_testing\\CIFAR_data\\Train\\";
	int thread_num = 1;
	int batchSize = 128;
	int train_size = 5.0e+4;
	int epoch = 50;
	bool pretrained = false;
	int channel = 3, height = 24, width = 24;
	int byte_size = batchSize * channel * height * width * sizeof(float);
	float *d_batch, *d_label;
	checkCudaErrors(cudaMalloc(&d_batch, byte_size));
	checkCudaErrors(cudaMalloc(&d_label, byte_size));
	
	Image_preparator prep(thread_num);
	prep.cifar_preparation(train_path, batchSize, byte_size, true);
	prep.start_next_cifar_epoch();

	int update_type = Neural_Network::mSGD;
	Neural_Network nn(gpuid, batchSize, pretrained, update_type);
	//nn.DEBUG_LEVEL = 3;
	nn.adam.alpha = 1.0e-6;
	nn.msgd.lr = 1.0e-4;
	nn.addInput(channel, height, width);
	nn.addConv(64, 5, 2, "conv1");
	nn.addReLU();
	nn.addPool(3, 2, 1);
	nn.addConv(64, 5, 2, "conv2");
	nn.addReLU();
	nn.addPool(3, 2, 1);
	nn.addConv(384, 6, 0, "local3");
	nn.addReLU();
	//nn.addDropout(0.4);
	nn.addConv(192, 1, 0, "local4");
	nn.addReLU();
	//nn.addDropout(0.4);
	nn.addConv(10, 1, 0, "local5");
	nn.addReLU();
	nn.addSoftMax();

	nn.setWorkSpace();

	size_t start_iter = 0;
	if (pretrained)
	{
		nn.load_weights("CIFAR-10");
		start_iter = nn.it;
		std::cout << "Start straining from " << start_iter << " iteration" << std::endl;
	}
	
	size_t iteration = epoch * train_size / batchSize;
	//size_t iteration = 10;
	std::cout << "There will be total: " << iteration - start_iter << " iterations" << std::endl;
	//std::cout << "It will take about " << (iteration - start_iter) * 0.3 << "sec" << std::endl;

	auto t1 = std::chrono::high_resolution_clock::now();
	for (int iter = start_iter; iter < iteration; iter++)
	{
		prep.load_next_batch(d_batch, d_label);
		
		//show_some_data(d_batch, 10);

		if ((iter + 1) % 200 == 0 && 0)
			nn.train(d_batch, d_label, iter, true, false);
		else
		{
			if (iter % 50 == 0)
				nn.train(d_batch, d_label, iter, true, false);
			else
				nn.train(d_batch, d_label, iter,false, false);
		}
		if ((iter + 1) % (train_size / batchSize) == 0 && ((iter + 1) < iteration))
		{	
			prep.start_next_cifar_epoch();	
		}

		//checkCudaErrors(cudaDeviceSynchronize());
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	printf("Avr Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iteration);
	
	nn.save_weights("CIFAR-10");

	checkCudaErrors(cudaFree(d_batch));
	checkCudaErrors(cudaFree(d_label));
}

void Cifar_test()
{
	std::cout << "CIFAR-10 testing\n";
	int gpuid = 0;
	checkCudaErrors(cudaSetDevice(gpuid));
	std::string train_path = "E:\\Kirill_testing\\CIFAR_data\\Test\\";
	int thread_num = 1;
	int batchSize = 1;	
	
	bool pretrained = true;
	int channel = 3, height = 24, width = 24;
	int byte_size = batchSize * channel * height * width * sizeof(float);
	float *d_batch, *d_label;
	checkCudaErrors(cudaMalloc(&d_batch, byte_size));
	checkCudaErrors(cudaMalloc(&d_label, byte_size));

	Image_preparator prep(thread_num);
	prep.cifar_preparation(train_path, batchSize, byte_size, false);
	prep.start_next_cifar_epoch();

	int update_type = Neural_Network::ONLY_INFERENCE;
	Neural_Network nn(gpuid, batchSize, pretrained, update_type);	
	nn.addInput(channel, height, width);
	nn.addConv(64, 5, 2, "conv1");
	nn.addReLU();
	nn.addPool(3, 2, 1);
	nn.addConv(64, 5, 2, "conv2");
	nn.addReLU();
	nn.addPool(3, 2, 1);
	nn.addConv(384, 6, 0, "local3");
	nn.addReLU();
	nn.addConv(192, 1, 0, "local4");
	nn.addReLU();
	nn.addConv(10, 1, 0, "local5");
	nn.addReLU();
	nn.addSoftMax();

	nn.setWorkSpace();

	nn.load_weights("CIFAR-10");

	

	//size_t iteration = train_size;
	//size_t iteration = 10;
	//std::cout << "There will be total: " << iteration << " iterations" << std::endl;
	//std::cout << "It will take about " << (iteration - start_iter) * 0.3 << "sec" << std::endl;

	float classification_error = 1.0f;

	int classifications = 10000;
	int num_errors = 0;
	float label;
	auto t1 = std::chrono::high_resolution_clock::now();
	for (int iter = 0; iter < classifications; iter++)
	{
		prep.load_next_batch(d_batch, d_label);

		nn.inference(d_batch);
		
		auto res = nn.getResult();

		// Perform classification
		std::vector<float> class_vec(10);
		//auto class_vec = nn.interpret_softmax_result(nullptr,false);

		// Copy back result
		checkCudaErrors(cudaMemcpy(&class_vec[0], res, sizeof(float) * 10, cudaMemcpyDeviceToHost));

		// Determine classification according to maximal response
		int chosen = 0;
		for (int id = 1; id < 10; ++id)
		{
			if (class_vec[chosen] < class_vec[id]) chosen = id;
		}
		checkCudaErrors(cudaMemcpy(&label, d_label, sizeof(float), cudaMemcpyDeviceToHost));
		if (chosen != (int)label)
			++num_errors;

		//checkCudaErrors(cudaDeviceSynchronize());
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	printf("Avr Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / classifications);

	classification_error = (float)num_errors / (float)classifications;
	printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
	

	checkCudaErrors(cudaFree(d_batch));
	checkCudaErrors(cudaFree(d_label));
}

void Cifar_10()
{
	Cifar_train();
	Cifar_test();
}