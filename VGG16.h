#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>    // std::sort
#include <functional>
#include <array>
#include <chrono>
#include <random>
#include "NN.h"
#include "json.hpp"
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2\cudaimgproc.hpp"

#include <thread>
#include <mutex>
#include <chrono>
std::mutex m1;
std::mutex m2;

#include"Image_preparation.h"

void VGG16_training()
{
	std::cout << "VGG16 training stage\n";
	int num_gpus;
	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	printf("%i\n", num_gpus);

	cudaDeviceProp prop;

	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	std::cout << prop.name << ":" << std::endl << prop.multiProcessorCount << std::endl;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 1));
	std::cout << prop.name << ":" << std::endl << prop.multiProcessorCount << std::endl;

	int gpuid = 0;
	checkCudaErrors(cudaSetDevice(gpuid));

	size_t train_size = 1281024;
	//size_t train_size = 64;
	int batchSize = 16;
	bool pretrained = false;
	int channels = 3, height = 224, width = 224;

	std::string im_path = "E:\\Kirill_testing\\Image_for_training\\ILSVRC12_task2_validation\\";
	std::string l_path = "E:\\Kirill_testing\\Image_for_training\\ILSVRC2012_validation_with_2014.txt";

	std::string train_im_path = "E:\\Kirill_testing\\Image_for_training\\ILSVRC\\ILSVRC2012_img_train\\";
	std::string train_l_path = "E:\\Kirill_testing\\Image_for_training\\imagenet_class_index.json";

	size_t byte_size = sizeof(float) * channels * height * width * batchSize;

	int thread_num = 8;
	float *d_batch, *d_label;

	Image_preparator prep(thread_num);
	//prep.set_train_param(train_im_path, train_l_path, batchSize, byte_size, train_size);
	//prep.start_next_epoch();

	checkCudaErrors(cudaMalloc(&d_batch, byte_size));
	checkCudaErrors(cudaMalloc(&d_label, sizeof(float) * batchSize));

	// training
	{
		//prep.set_train_param_test(im_path, l_path,batchSize,byte_size,train_size);
		prep.set_train_param(train_im_path, train_l_path, batchSize, byte_size, train_size);
		prep.start_next_epoch();

		//int update_type = Neural_Network::ADAM;
		int update_type = Neural_Network::mSGD;
		Neural_Network nn(gpuid, batchSize, pretrained, update_type);
		nn.msgd.lr = 1.0e-2;
		nn.adam.alpha = 1.0e-5; // left ADAM for fine-tuning
		nn.adam.epsilon = 1.0e-8;
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
		nn.addConv(4096, 7, 0, "fc6");
		nn.addReLU();
		nn.addDropout(0.5);
		nn.addConv(4096, 1, 0, "fc7");
		nn.addReLU();
		nn.addDropout(0.5);
		nn.addConv(1000, 1, 0, "fc8");
		nn.addReLU();
		nn.addSoftMax();

		nn.setWorkSpace();

		size_t start_iter = 0;

		if (pretrained)
			nn.load_weights("tmp");

		size_t epoch = 10;
		size_t iteration = epoch * train_size / batchSize;
		//size_t iteration = 3;
		std::cout << "There will be " << iteration << " iterations" << std::endl;
		std::cout << "It will take about " << iteration * 0.8 << "sec" << std::endl;

		auto t1 = std::chrono::high_resolution_clock::now();
		for (size_t iter = 0; iter < iteration; iter++)
		{			
			auto t1 = std::chrono::high_resolution_clock::now();
			prep.load_next_batch(d_batch, d_label);
			auto t2 = std::chrono::high_resolution_clock::now();
			//if ((iter) % 10 == 0)
				//printf("Image getting time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);
			//show_some_data(d_label, 1);
			//system("pause");

			t1 = std::chrono::high_resolution_clock::now();
			if ((iter+1) % /*(train_size / batchSize)*/ 1000 == 0)
				nn.train(d_batch /*d_currBatch*/ /*currBatch->data()*/, d_label, iter, true, true);
			else
			{
				if ((iter) % 50 == 0 )
					nn.train(d_batch, d_label, iter, true, false);
				else
					nn.train(d_batch /*d_currBatch*/ /*currBatch->data()*/, d_label, iter);
			}
			t2 = std::chrono::high_resolution_clock::now();
			//if ((iter) % 10 == 0)
				//printf("Train iter time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);

			// Second part makes no sense so far - look at iteration initialization...
			if ((iter + 1) % (train_size / batchSize) == 0 && ((iter + 1) < iteration))
			{
				t1 = std::chrono::high_resolution_clock::now();
				prep.start_next_epoch();
				t2 = std::chrono::high_resolution_clock::now();
				//printf("New epoch rand time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);
			}

			//checkCudaErrors(cudaDeviceSynchronize());
		}
		auto t2 = std::chrono::high_resolution_clock::now();
		printf("Avr Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iteration);

		nn.save_weights("VGG16_mSGD");
		
		checkCudaErrors(cudaFree(d_label));
		checkCudaErrors(cudaFree(d_batch));
	}

	std::cout << "VGG16 training stage over\n";
	return;

	{
		batchSize = 1;
		std::cout << "Test stage\n";
		channels = 3, height = 800, width = 1024;
		prep.validation_preparation();
	
		int update_type = Neural_Network::ONLY_INFERENCE;		
		Neural_Network nn(0, batchSize, true, update_type);				
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
		nn.addConv(4096, 7, 0, "fc6");
		nn.addReLU();
		nn.addDropout(0.5);
		nn.addConv(4096, 1, 0, "fc7");
		nn.addReLU();
		nn.addDropout(0.5);
		nn.addConv(1000, 1, 0, "fc8");
		nn.addReLU();
		nn.addSoftMax();
	
		nn.setWorkSpace();
		
		if (pretrained)
			nn.load_weights("VGG16_ADAM_256_size");

		size_t top1 = 0;
		size_t top5 = 0;

		float * d_data;

		float * d_img = nullptr;
		float * d_flipped = nullptr;
		int label;

		size_t fixed_size = 384 * 1600 * 3 * sizeof(float);

		checkCudaErrors(cudaMalloc(&d_img, fixed_size));
		checkCudaErrors(cudaMalloc(&d_flipped, fixed_size));
				
		size_t iterations = train_size;
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int iter = 0; iter < iterations; iter++)
		{
			
			prep.load_next_val_pair(d_img, d_flipped, height, width, label);
			//std::cout << "Img param: " << height << " " << width << " " << label << std::endl;
			nn.clearTensors();
			
			nn.redefineTensors(height, width, channels);	

			nn.inference(d_img);
			auto res_orig = nn.interpret_softmax_result(nullptr, false);
			
			//nn.inference(d_flipped);
			//auto res_flipped = nn.interpret_softmax_result(nullptr, false);

			std::vector<std::pair<float, int>> res_avr;
			for (int i = 0; i < res_orig.size(); i++)
				res_avr.push_back(std::make_pair((res_orig[i])/* + res_flipped[i]) / 2*/, i));
			if (res_avr.size() != 1000)
				std::cout << "Something wrong! " << res_avr.size() << std::endl;
			
			std::sort(res_avr.begin(), res_avr.end(), [](std::pair<float, int> a, std::pair<float, int> b) {
				return a > b;
			});

			if (res_avr[0].second == label)
				top1++;

			for (int i = 0; i < 5; i++)
			{
				if (res_avr[i].second == label)
					top5++;
			}
			checkCudaErrors(cudaDeviceSynchronize());

		}
		auto t2 = std::chrono::high_resolution_clock::now();
		printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / (iterations));
		std::cout << "Top1 error: " << float(iterations - top1) / float(iterations) * 100
			<< "% Top5 error: " << float(iterations - top5) / float(iterations) * 100 << "%\n";

		prep.stop();

		checkCudaErrors(cudaFree(d_img));
		checkCudaErrors(cudaFree(d_flipped));

	}

}

void get_next_test_image(const char* filename, int &height, int &width, std::vector<float> &img, bool flip)
{
	using namespace cv;
	using namespace std;

	Mat image;
	image = imread(filename, IMREAD_COLOR);

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}

	cv::cvtColor(image, image, CV_BGR2RGB);
	Mat im_f;
	image.convertTo(im_f, CV_32FC3, 1 / 255.0);
	image = im_f;

	float m = min(image.rows, image.cols);

	Mat outIm;

	
	int Q = 384;
	float ratio = float(Q) / m;		
	cv::resize(image, outIm, cv::Size(), ratio, ratio);
		
	auto size = outIm.size();	
	cv::Size newsize(size.width, size.height * 3);
	cv::Mat destination(newsize, CV_32FC1);	
	if (flip)
		cv::flip(outIm, im_f, 1);
	else
		im_f = outIm;

	//copy the channels from the source image to the destination
	for (int i = 0; i < im_f.channels(); ++i)
	{
		cv::extractChannel(
			im_f,
			cv::Mat(
				size.height,
				size.width,
				CV_32FC1,
				&(destination.at<float>(size.height*size.width*i))),
			i);
	}
	height = size.height;
	width = size.width;
	
	img.clear();	
	if (destination.isContinuous())
		img.assign((float*)destination.datastart, (float*)destination.dataend);
	else
		for (int i = 0; i < destination.rows; ++i)
			img.insert(img.end(), destination.ptr<float>(i), destination.ptr<float>(i) + destination.cols);
}

void image_preparation_thread(int &height, int &width, int &height_flip, int &width_flip, std::vector<float> &img, std::vector<float> &flipped, bool &stop)
{	
	std::unique_lock<std::mutex> lock1(m1, std::defer_lock);
	std::unique_lock<std::mutex> lock2(m2, std::defer_lock);
	bool flip;
	namespace fs = std::experimental::filesystem;
	std::string path = "E:\\Kirill_testing\\Image_for_training\\ILSVRC12_task2_validation\\";
	for (auto & p : fs::directory_iterator(path))
	{
		fs::path str = p;		
		//std::cout << str << std::endl;
		lock1.lock();
		flip = false;
		get_next_test_image(str.generic_string().c_str(), height, width, img, flip);
		lock1.unlock();
		lock2.lock();
		if (stop)
			return;
		flip = true;
		get_next_test_image(str.generic_string().c_str(), height_flip, width_flip, flipped, flip); 
		lock2.unlock();		
	}
}

void VGG16_test()
{
	std::cout << "Test stage\n";
	int channels = 3, height = 640, width = 800, height_flip = 0 , width_flip = 0;

	//unsigned w_old, h_old;
	//auto image_old = load_train_data_CHW("training\\laska0.png", w_old, h_old);

	//std::vector<float> img, flipped, *image;
	//std::unique_lock<std::mutex> lock1(m1, std::defer_lock);
	//std::unique_lock<std::mutex> lock2(m2, std::defer_lock);
	//lock2.lock();
	//bool stop = false;

	//std::thread t(image_preparation_thread, 
	//	std::ref(height), std::ref(width), std::ref(height_flip), std::ref(width_flip), 
	//	std::ref(img), std::ref(flipped), std::ref(stop));
	//t.detach();
	////image_preparation_thread(height, width, height_flip, width_flip, img, flipped);
	//


	//std::cout << height<< " " << width<< " " << img.size()<<std::endl;

	//for (int i = 0; i < 20; i++)
	//	std::cout << image_old[i] << " " << img[i] << std::endl;
	//
	//float im_old = 0;
	//float diff = 0;
	//float im_new = 0;
	//
	//if (image_old == img)
	//	std::cout << "Nice!";
	//else
	//	for(int i=0;i<image_old.size();i++)
	//		if (image_old[i] != img[i])
	//		{
	//			im_old += image_old[i];
	//			diff += std::abs(image_old[i] - img[i]);
	//			im_new += img[i];
	//			//if (std::abs(image_old[i] - img[i]) > 1.0e-8)
	//				//std::cout << "Wow!";
	//			//std::cout << "Here: " << i<<" "<< image_old[i] << " " << img[i]<<" "<< image_old[i] - img[i];
	//			//break;
	//		}
	//std::cout << diff << " " << im_old << " " << im_new << " " << diff / (im_old + im_new) << std::endl;
	//return;

	int gpuid = 1;
	checkCudaErrors(cudaSetDevice(gpuid));

	int thread_num = 4;
	int val_size = 6000;
	std::string im_path = "E:\\Kirill_testing\\Image_for_training\\ILSVRC12_task2_validation\\";
	std::string labels_path("E:\\Kirill_testing\\Image_for_training\\ILSVRC2012_validation_with_2014.txt");
	//std::cout << "What\n";
	Image_preparator prep(thread_num);
	prep.set_validation_param(im_path, labels_path, val_size);
	prep.start_validation();

	int batchSize = 1;
	bool pretrained = true;
	int update_type = Neural_Network::ONLY_INFERENCE;

	Neural_Network nn(gpuid, batchSize, pretrained, update_type);

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
	nn.addConv(4096, 7, 0, "fc6");
	nn.addReLU();
	nn.addDropout(0.5);
	nn.addConv(4096, 1, 0, "fc7");
	nn.addReLU();
	nn.addDropout(0.5);
	nn.addConv(1000, 1, 0, "fc8");
	nn.addReLU();
	nn.addSoftMax();

	nn.setWorkSpace();
	
	nn.load_weights("tmp");
	//nn.convert_VGG16_weights();

	// read a JSON file
	const char* path = "VGG16\\classes\\imagenet_class_index.json";	
	std::ifstream in(path);
	using json = nlohmann::json;
	json j;
	in >> j;
	std::vector<std::string> object_names(1000);
	for (json::iterator it = j.begin(); it != j.end(); ++it)
		object_names[atoi(it.key().c_str())] = it.value()[1].dump();

	std::ifstream labels("E:\\Kirill_testing\\Image_for_training\\ILSVRC2012_validation_with_2014.txt");
	
	size_t top1 = 0;
	size_t top5 = 0;

	float * d_data;
	
	float * d_img = nullptr;
	float * d_flipped = nullptr;
	int label;

	size_t fixed_size = 384 * 1600 * 3 * sizeof(float);

	checkCudaErrors(cudaMalloc(&d_img, fixed_size));
	checkCudaErrors(cudaMalloc(&d_flipped, fixed_size));

	//size_t test_size = 128;
	size_t iterations = val_size;
	auto t1 = std::chrono::high_resolution_clock::now();
	for (int iter = 0; iter < iterations; iter++)
	{	
		//std::cout << iter << std::endl;
		prep.load_next_val_pair(d_img, d_flipped, height, width, label);
		nn.clearTensors();
		//std::cout << height << " " << width << std::endl;
		nn.redefineTensors(height,width,channels);

		//checkCudaErrors(cudaMalloc(&d_data, sizeof(float) * channels * height * width));
		//checkCudaErrors(cudaMemcpy(d_data, d_img, sizeof(float) * channels * height * width, cudaMemcpyDeviceToDevice));

		//nn.inference(d_data);
		//
		////nn.inference(image->data());
		//checkCudaErrors(cudaDeviceSynchronize());
		//auto res_orig = nn.interpret_softmax_result("VGG16\\classes\\imagenet_class_index.json", false);
		//
		//lock1.unlock();
		//lock2.lock();				
		//image = &flipped;
		//
		//checkCudaErrors(cudaMemcpy(d_data, img.data(), sizeof(float) * channels * height * width, cudaMemcpyHostToDevice));
		//nn.inference(d_data);
		//
		//nn.inference(image->data());		

		nn.inference(d_img);
		auto res_orig = nn.interpret_softmax_result(/*nullptr*/ "VGG16\\classes\\imagenet_class_index.json", false);
		//checkCudaErrors(cudaMemcpy(d_data, d_flipped, sizeof(float) * channels * height * width, cudaMemcpyDeviceToDevice));
		nn.inference(d_flipped);
		auto res_flipped = nn.interpret_softmax_result(/*nullptr*/ "VGG16\\classes\\imagenet_class_index.json", false);

		std::vector<std::pair<float, int>> res_avr;
		for (int i = 0; i < res_orig.size(); i++)
			res_avr.push_back(std::make_pair((res_orig[i] + res_flipped[i]) / 2, i));

		//std::cout << res_avr.size() << std::endl;

		std::sort(res_avr.begin(), res_avr.end(), [](std::pair<float, int> a, std::pair<float, int> b) {
			return a > b;
		});
		//labels >> label;
		if (res_avr[0].second == label)
			top1++;
		//std::cout << "Answer is: " << label << " " << object_names[label] << std::endl;
		for (int i = 0; i < 5; i++)
		{
			if (res_avr[i].second == label)
				top5++;
			//std::cout << res_avr[i].first * 100 << " " << res_avr[i].second << " " << object_names[res_avr[i].second] << std::endl;
		}
		//std::cout << std::endl;		

		//checkCudaErrors(cudaFree(d_data));
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / (val_size));	
	std::cout << "Top1 error: " << float(val_size - top1) / float(val_size) * 100 
		<< "% Top5 error: " << float(val_size - top5) / float(val_size) * 100<<"%\n";

	prep.stop();

	checkCudaErrors(cudaFree(d_img));
	checkCudaErrors(cudaFree(d_flipped));
	//stop = true;
	//lock2.unlock();
	//using namespace std::chrono_literals;
	//std::this_thread::sleep_for(1s);
}

int VGG16()
{
	//VGG16_training();
	VGG16_test();
	return 0;
}