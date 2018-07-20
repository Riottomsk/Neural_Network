#include "DeepImageMatting.h"
#include <chrono>
#include "Image_preparation.h"

void DIM_train()
{
	int gpuid = 0;
	checkCudaErrors(cudaSetDevice(gpuid));
	auto image_path = "E:\\Kirill_testing\\data\\data";
	unsigned channels = 4, height = 320, width = 320;
	
	int batchSize = 4;
	int thread_num = 8;
	int epoch = 500;
	int train_size = 3712;
	//int train_size = 32;
	
	bool pretrained = true;


	size_t byte_size = channels	 * height * width * batchSize * sizeof(float);
	float *d_batch;
	float *d_label;
	checkCudaErrors(cudaMalloc(&d_batch, byte_size));
	checkCudaErrors(cudaMalloc(&d_label, byte_size / channels));
	Image_preparator prep(thread_num);
	prep.set_dim_train_param(image_path, batchSize, byte_size, train_size);
	prep.next_dim_epoch();

	int update_type = Neural_Network::ADAM;
	int loss_type = Neural_Network::COMPOSITIONAL;
	Neural_Network nn(gpuid, batchSize, pretrained, update_type,loss_type);	
	{
		nn.adam.alpha = 1.0e-8;
		nn.adam.epsilon = 1.0e-8;
		nn.adam.beta2 = 0.9999;
		nn.DEBUG_LEVEL = 1;
		nn.addInput(channels, height, width);
		nn.addConv(64, 3, 1, "conv1_1");
		nn.addReLU();
		nn.addConv(64, 3, 1, "conv1_2");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(128, 3, 1,"conv2_1");
		nn.addReLU();
		nn.addConv(128, 3, 1, "conv2_2");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(256, 3, 1, "conv3_1");
		nn.addReLU();
		nn.addConv(256, 3, 1, "conv3_2");
		nn.addReLU();
		nn.addConv(256, 3, 1, "conv3_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(512, 3, 1, "conv4_1");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv4_2");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv4_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(512, 3, 1, "conv5_1");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv5_2");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv5_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(4096, 7, 3, "fc6");
		nn.addReLU();
		// end of encoder
		
		if (!pretrained)
			nn.load_only_weights("VGG16");

		nn.addConv(512, 1, 0, "Deconv6");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(512, 5, 2, "Deconv5");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(256, 5, 2, "Deconv4");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(128, 5, 2, "Deconv3");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(64, 5, 2, "Deconv2");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(64, 5, 2, "Deconv1");
		nn.addReLU();
		nn.addConv(1, 5, 2, "Raw_Aplha_Pred");
		// end of decoder

		nn.setWorkSpace();

		size_t start_iter = 0;
		if (pretrained)
		{
			nn.load_weights("tmp_DIM_80K");
			start_iter = nn.it;
		}

		int ckp = 10;

		size_t iterations = train_size * epoch / batchSize;
		//size_t iterations = 0;
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int iter = start_iter; iter < iterations; iter++)
		{			
			//std::cout << "Iter: " << iter << std::endl;

			prep.load_next_batch(d_batch, d_label);		
			if ((iter + 1) % 1000 == 0)
			{
				std::cout << std::endl;
				nn.train(d_batch, d_label, iter, true, true);				
			}
			else
			{
				if (iter % ckp == 0)
				{
					std::cout << std::endl;
					nn.train(d_batch, d_label, iter, true, false);
					auto res = nn.getResult();
					prep.check_dim_and_stay(res, d_label, d_batch, height, width);
				}
				else
					nn.train(d_batch, d_label, iter, false, false);
			}
			//if ((iter + 1) % (train_size/batchSize) == 0)
			//{
			//	prep.next_dim_epoch();
			//	nn.save_weights("tmp_epoch");
			//}

			if (iter % (ckp / 10) == 0)
				std::cout << float((iter % ckp)) / (float(ckp)/100) << "% ";

		}
		auto t2 = std::chrono::high_resolution_clock::now();
		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / (iterations - start_iter));
	}

	nn.save_weights("DIM");
	//auto res = nn.getResult();	
	//prep.check_dim_image(res, d_label);
	prep.stop();
	
	// refinement stage - TODO
	if(0)
	{
		Neural_Network refinement(0, batchSize, pretrained);
		//refinement.update_type = Neural_Network::ADAM;
		refinement.addInput(channels, height, width);
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(1, 3, 1);

		nn.setWorkSpace();
		refinement.setWorkSpace();

		std::vector<float> vec(channels * height * width, 1);

		auto t1 = std::chrono::high_resolution_clock::now();
		nn.inference(vec.data());
		checkCudaErrors(cudaDeviceSynchronize());
		refinement.inference(vec.data());
		auto t2 = std::chrono::high_resolution_clock::now();

		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / 1);
	}

	checkCudaErrors(cudaFree(d_batch));
	checkCudaErrors(cudaFree(d_label));
}

void DIM_eval()
{
	int gpuid = 0;
	checkCudaErrors(cudaSetDevice(gpuid));
	auto image_path = "E:\\Kirill_testing\\data\\data";
	unsigned channels = 4, height = 1080, width = 1920;

	int batchSize = 1;
	int thread_num = 1;
	//int epoch = 50;
	//int train_size = 3000;

	bool pretrained = true;


	size_t byte_size = channels	 * height * width * batchSize * sizeof(float);
	float *d_batch;
	float *d_label;
	checkCudaErrors(cudaMalloc(&d_batch, byte_size));
	checkCudaErrors(cudaMalloc(&d_label, byte_size / channels));
	Image_preparator prep(thread_num);
	prep.set_dim_train_param(image_path, batchSize, byte_size);
	//prep.next_dim_epoch();
	prep.dim_eval();

	int update_type = Neural_Network::ONLY_INFERENCE;	
	Neural_Network nn(gpuid, batchSize, pretrained, update_type);
	{		
		nn.DEBUG_LEVEL = 1;
		nn.addInput(channels, height, width);
		nn.addConv(64, 3, 1, "conv1_1");
		nn.addReLU();
		nn.addConv(64, 3, 1, "conv1_2");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(128, 3, 1, "conv2_1");
		nn.addReLU();
		nn.addConv(128, 3, 1, "conv2_2");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(256, 3, 1, "conv3_1");
		nn.addReLU();
		nn.addConv(256, 3, 1, "conv3_2");
		nn.addReLU();
		nn.addConv(256, 3, 1, "conv3_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(512, 3, 1, "conv4_1");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv4_2");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv4_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(512, 3, 1, "conv5_1");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv5_2");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv5_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(4096, 7, 3, "fc6");
		nn.addReLU();
		// end of encoder

		nn.addConv(512, 1, 0, "Deconv6");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(512, 5, 2, "Deconv5");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(256, 5, 2, "Deconv4");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(128, 5, 2, "Deconv3");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(64, 5, 2, "Deconv2");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(64, 5, 2, "Deconv1");
		nn.addReLU();
		nn.addConv(1, 5, 2, "Raw_Aplha_Pred");
		// end of decoder

		nn.setWorkSpace();

		size_t start_iter = 0;
		
		nn.load_only_weights("tmp");
		
		//size_t iterations = train_size * epoch / batchSize;
		size_t iterations = 1;
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int iter = start_iter; iter < iterations; iter++)
		{
			prep.load_next_batch(d_batch, d_label);			
			//nn.train(d_batch, d_label, iter, true, false);			
			nn.inference(d_batch);
		}
		auto t2 = std::chrono::high_resolution_clock::now();
		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / (iterations - start_iter));
	}

	auto res = nn.getResult();
	//prep.load_next_batch(d_batch, d_label);
	prep.check_dim_and_stay(res, d_label, d_batch, height, width);
	prep.stop();

	// refinement stage - TODO
	if (0)
	{
		Neural_Network refinement(0, batchSize, pretrained);
		//refinement.update_type = Neural_Network::ADAM;
		refinement.addInput(channels, height, width);
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(1, 3, 1);

		nn.setWorkSpace();
		refinement.setWorkSpace();

		std::vector<float> vec(channels * height * width, 1);

		auto t1 = std::chrono::high_resolution_clock::now();
		nn.inference(vec.data());
		checkCudaErrors(cudaDeviceSynchronize());
		refinement.inference(vec.data());
		auto t2 = std::chrono::high_resolution_clock::now();

		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / 1);
	}

	checkCudaErrors(cudaFree(d_batch));
	checkCudaErrors(cudaFree(d_label));
}

void DIM_sequence(const char* image_path)
{
	int gpuid = 0;
	checkCudaErrors(cudaSetDevice(gpuid));
	//auto image_path = "E:\\Kirill_testing\\data\\data\\24";
	auto save_path = "E:\\Kirill_testing\\DIM_result\\Pic";
	unsigned channels = 4, height = 1080, width = 1920;

	int batchSize = 1;
	int thread_num = 1;
	//int epoch = 50;
	int seq_size;

	bool pretrained = true;


	size_t byte_size = channels	 * height * width * batchSize * sizeof(float);
	float *d_batch;
	float *d_label;
	checkCudaErrors(cudaMalloc(&d_batch, byte_size));
	checkCudaErrors(cudaMalloc(&d_label, byte_size / channels));
	Image_preparator prep(thread_num);
	seq_size = prep.set_dim_seq_param(image_path, batchSize, byte_size);
	//prep.next_dim_epoch();
	prep.dim_eval();

	int update_type = Neural_Network::ONLY_INFERENCE;
	Neural_Network nn(gpuid, batchSize, pretrained, update_type);
	{
		nn.DEBUG_LEVEL = 1;
		nn.addInput(channels, height, width);
		nn.addConv(64, 3, 1, "conv1_1");
		nn.addReLU();
		nn.addConv(64, 3, 1, "conv1_2");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(128, 3, 1, "conv2_1");
		nn.addReLU();
		nn.addConv(128, 3, 1, "conv2_2");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(256, 3, 1, "conv3_1");
		nn.addReLU();
		nn.addConv(256, 3, 1, "conv3_2");
		nn.addReLU();
		nn.addConv(256, 3, 1, "conv3_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(512, 3, 1, "conv4_1");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv4_2");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv4_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(512, 3, 1, "conv5_1");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv5_2");
		nn.addReLU();
		nn.addConv(512, 3, 1, "conv5_3");
		nn.addReLU();
		nn.addPool(2, 2);
		nn.addConv(4096, 7, 3, "fc6");
		nn.addReLU();
		// end of encoder

		nn.addConv(512, 1, 0, "Deconv6");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(512, 5, 2, "Deconv5");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(256, 5, 2, "Deconv4");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(128, 5, 2, "Deconv3");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(64, 5, 2, "Deconv2");
		nn.addReLU();
		nn.addUnpool();
		nn.addConv(64, 5, 2, "Deconv1");
		nn.addReLU();
		nn.addConv(1, 5, 2, "Raw_Aplha_Pred");
		// end of decoder

		nn.setWorkSpace();

		size_t start_iter = 0;

		nn.load_only_weights("tmp");

		//size_t iterations = train_size * epoch / batchSize;
		size_t iterations = seq_size;
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int iter = start_iter; iter < iterations; iter++)
		{
			prep.load_next_batch(d_batch, d_label);
			nn.inference(d_batch);
			auto res = nn.getResult();
			//prep.check_dim_and_stay(res, d_label, d_batch, height, width);
			prep.save_dim_pic(save_path, res, d_batch, height, width, iter);
		}
		auto t2 = std::chrono::high_resolution_clock::now();
		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / (iterations - start_iter));
	}


	prep.stop();

	// refinement stage - TODO
	if (0)
	{
		Neural_Network refinement(0, batchSize, pretrained);
		//refinement.update_type = Neural_Network::ADAM;
		refinement.addInput(channels, height, width);
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(64, 3, 1);
		refinement.addReLU();
		refinement.addConv(1, 3, 1);

		nn.setWorkSpace();
		refinement.setWorkSpace();

		std::vector<float> vec(channels * height * width, 1);

		auto t1 = std::chrono::high_resolution_clock::now();
		nn.inference(vec.data());
		checkCudaErrors(cudaDeviceSynchronize());
		refinement.inference(vec.data());
		auto t2 = std::chrono::high_resolution_clock::now();

		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / 1);
	}

	checkCudaErrors(cudaFree(d_batch));
	checkCudaErrors(cudaFree(d_label));
}


void DIM()
{
	DIM_train();
	//DIM_eval();
	//DIM_sequence("E:\\Kirill_testing\\data\\data\\3");
}