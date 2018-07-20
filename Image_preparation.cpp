#include "Image_preparation.h"
#include "json.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include <random>

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

Image_preparator::Image_preparator(int thread_num_)
{
	//std::cout << "IS\n";
	thread_num = thread_num_;
	mutices = new std::vector<std::mutex>(thread_num);
	ready_to_load.resize(thread_num, true);
	load_cvs = new std::vector<std::condition_variable>(thread_num);
	batches.resize(thread_num);
	labels.resize(thread_num);
	th_file_map.resize(thread_num);

}

void Image_preparator::set_train_param(std::string image_path_, std::string labels_path_, int batchSize_, size_t byte_size_, size_t train_size_)
{
	train_image_path = image_path_;
	train_labels_path = labels_path_;
	batchSize = batchSize_;
	byte_size = byte_size_;
	train_size = train_size_;

	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaMallocHost(&batches[i], byte_size));
		checkCudaErrors(cudaMallocHost(&labels[i], sizeof(float) * batchSize));
	}
	prepare_filenames();
	//map_and_rand();
}

void Image_preparator::set_train_param_test(std::string image_path_, std::string labels_path_, int batchSize_, size_t byte_size_, size_t train_size_)
{
	train_image_path = image_path_;
	train_labels_path = labels_path_;
	batchSize = batchSize_;
	byte_size = byte_size_;
	train_size = train_size_;

	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaMallocHost(&batches[i], byte_size));
		checkCudaErrors(cudaMallocHost(&labels[i], sizeof(float) * batchSize));
	}
	//std::cout << "Prep filenames\n";
	//prepare_filenames_test();
	//std::cout << "Split filenames\n";
	size_t pic_per_th = train_size / thread_num;
	
	for (int i = 0; i < thread_num; i++)
	{
		auto start_itr = std::next(file_map.cbegin(), i*pic_per_th);
		auto end_itr = std::next(file_map.cbegin(), (i + 1)*pic_per_th);

		std::copy(start_itr, end_itr, th_file_map[i].begin());
	}
	//std::cout << "Split over\n";
}

void Image_preparator::set_validation_param(std::string image_path_, std::string labels_path_, size_t validation_size_)
{
	//std::cout << "Going\n";
	validation_image_path = image_path_;
	validation_labels_path = labels_path_;
	batchSize = 1;
	//byte_size = byte_size_;
	validation_size = validation_size_;
	val_param.resize(thread_num);
	flipped.resize(thread_num);
	size_t max_bytes = sizeof(float) * 384 * 1600 * 3; // assume that no img will be bigger than 1600*384*3 after preparation
	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaMallocHost(&batches[i], max_bytes));
		checkCudaErrors(cudaMallocHost(&flipped[i], max_bytes));
		checkCudaErrors(cudaMallocHost(&labels[i], sizeof(float)));
	}
}

void Image_preparator::set_dim_train_param(std::string image_path_, int batchSize_, size_t byte_size_, size_t train_size_)
{
	train_image_path = image_path_;
	batchSize = batchSize_;
	byte_size = byte_size_;
	train_size = train_size_;
	th_dim_files.resize(thread_num);

	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaMallocHost(&batches[i], byte_size));
		checkCudaErrors(cudaMallocHost(&labels[i], byte_size / 4));
	}
	prepare_dim_filenames();
	using namespace cv;
	namedWindow("Ground Truth Alpha", WINDOW_NORMAL); // Create a window for display.
	moveWindow("Ground Truth Alpha", 1930, 10);
	resizeWindow("Ground Truth Alpha", 320, 320);
	namedWindow("Alpa Prediction", WINDOW_NORMAL); // Create a window for display.
	moveWindow("Alpa Prediction", 1920 + 360, 10);
	resizeWindow("Alpa Prediction", 320, 320);
	namedWindow("Original Image", WINDOW_NORMAL); // Create a window for display.	
	moveWindow("Original Image", 1930, 360);
	resizeWindow("Original Image", 320, 320);
	namedWindow("Trimap", WINDOW_NORMAL); // Create a window for display.	
	moveWindow("Trimap", 1920 + 360, 360);
	resizeWindow("Trimap", 320, 320);
	namedWindow("Composition", WINDOW_NORMAL); // Create a window for display.	
	moveWindow("Composition", 1920 + 360 + 360, 10);
	resizeWindow("Composition", 320, 320);
	namedWindow("Composition with threshold", WINDOW_NORMAL); // Create a window for display.	
	moveWindow("Composition with threshold", 1920 + 360 + 360, 360);
	resizeWindow("Composition with threshold", 320, 320);
}

int Image_preparator::set_dim_seq_param(std::string image_path_, int batchSize_, size_t byte_size_, size_t train_size_)
{
	train_image_path = image_path_;
	batchSize = batchSize_;
	byte_size = byte_size_;
	train_size = train_size_;
	th_dim_files.resize(thread_num);

	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaMallocHost(&batches[i], byte_size));
		checkCudaErrors(cudaMallocHost(&labels[i], byte_size / 4));
	}
	return prepare_dim_seq();
}

void Image_preparator::next_dim_epoch()
{
	stop();
	rand_dim_files();
	run = true;
	for (int i = 0; i < thread_num; i++)
		threads.emplace_back(&Image_preparator::train_dim_prep_th_no_white, this, i);
}

void Image_preparator::dim_eval()
{
	stop();
	//rand_dim_files();
	run = true;
	for (int i = 0; i < thread_num; i++)
		threads.emplace_back(&Image_preparator::eval_dim_th, this, i);
}

void Image_preparator::start_validation()
{		
	//std::cout << "On?\n";
	namespace fs = std::experimental::filesystem;	
		
	for (auto & p : fs::directory_iterator(validation_image_path))
	{
		fs::path str = p;
		//std::cout << str << std::endl;
		filenames.push_back(str);		
	}
	
	std::ifstream labels_file(validation_labels_path);
	vec_labels.resize(validation_size);
	for (size_t i = 0; i < validation_size; i++)
	{
		labels_file >> vec_labels[i];		
	}

	for (int i = 0; i < thread_num; i++)
	{
		threads.emplace_back(&Image_preparator::val_img_prep_th, this, i);
	}	
}

void Image_preparator::start_next_epoch()
{
	stop();	
	//std::cout << "Last epoch ended" << std::endl;
	map_and_rand();
	//std::cout << "Pictures randomized" << std::endl;
	//std::cout << th_file_map[0][0].first << " " << th_file_map[0][0].second << "\n" <<
	//	th_file_map[1][0].first << " " << th_file_map[1][0].second << "\n" <<
	//	th_file_map[2][0].first << " " << th_file_map[2][0].second << "\n";
	run = true;
	for (int i = 0; i < thread_num; i++)	
		threads.emplace_back(&Image_preparator::train_img_prep_th, this, i);
}

void Image_preparator::map_and_rand()
{
	//size_t pic_per_th = file_map.size() / thread_num;
	size_t pic_per_th = train_size / thread_num;
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(file_map.begin(), file_map.end(), g);	

	if (train_size < file_map.size())
		file_map.erase(file_map.begin() + train_size, file_map.end());

	//for (int i = 0; i < file_map.size(); i++)
		//std::cout << file_map[i].first << " " << file_map[i].second << std::endl;

	//system("pause");

	for (int i = 0; i < thread_num; i++)
	{
		auto start_itr = std::next(file_map.cbegin(), i*pic_per_th);
		auto end_itr = std::next(file_map.cbegin(), (i+1)*pic_per_th);
				
		std::copy(start_itr, end_itr, th_file_map[i].begin());
	}
}


void Image_preparator::load_next_batch(float * d_batch, float * d_label)
{
	while(!load_next_batch_internal(d_batch, d_label))	
	{
		std::cout << "Waiting batch for too long, restaring epoch..." << std::endl;
		next_dim_epoch();		
		load_next_batch(d_batch, d_label); // we still need data
	}
}

bool Image_preparator::load_next_batch_internal(float * d_batch, float * d_label)
{
	using namespace std::chrono_literals;
	std::unique_lock<std::mutex> lk(q_mutex);
	//cv.wait(lk, [this] {return !queue.empty(); });
	if (cv.wait_for(lk, 2000ms, [this] { return !queue.empty(); }))
	{
		int num = queue.back();
		queue.pop_back();
		//std::cout << "l_n_b: " << num << std::endl;

		checkCudaErrors(cudaMemcpyAsync(d_batch, batches[num], byte_size, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpyAsync(d_label, labels[num], batchSize * sizeof(float), cudaMemcpyHostToDevice)); // classification
		checkCudaErrors(cudaMemcpyAsync(d_label, labels[num], byte_size / 4, cudaMemcpyHostToDevice)); // composition

		ready_to_load[num] = true;
		(*load_cvs)[num].notify_one();
		return true;
	}
	else
		return false;
}

void Image_preparator::load_next_val_pair(float * d_img, float * d_flipped, int & height, int & width, int & label)
{
	//std::cout << "Here!0\n";
	std::unique_lock<std::mutex> lk(q_mutex);
	cv.wait(lk, [this] {return !queue.empty(); });
	int num = queue.back();
	queue.pop_back();
	//std::cout << "Here!1\n";
	height = val_param[num].first;
	width = val_param[num].second;

	size_t size = height * width * 3 * sizeof(float);

	//if (d_img != nullptr)
	//	checkCudaErrors(cudaFree(d_img));
	//if (d_flipped != nullptr)
	//	checkCudaErrors(cudaFree(d_flipped));
	//
	//checkCudaErrors(cudaMalloc(&d_img, size));
	//checkCudaErrors(cudaMalloc(&d_flipped, size));
	
	checkCudaErrors(cudaMemcpy(d_img, batches[num], size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_flipped, flipped[num], size, cudaMemcpyHostToDevice));
	
	label = labels[num][0];

	ready_to_load[num] = true;
	(*load_cvs)[num].notify_one();
}

void Image_preparator::stop()
{	
	if (!threads.empty())
	{
		run = false;
		for (int i = 0; i < thread_num; i++)
		{
			{
				std::lock_guard<std::mutex> lk((*mutices)[i]);
				ready_to_load[i] = true;
			}
			(*load_cvs)[i].notify_one();
			//std::cout << "W8 join" << std::endl;
			threads[i].join();
		}
		threads.clear();
	}
}

Image_preparator::~Image_preparator()
{
	stop();
	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaFreeHost(batches[i]));
		checkCudaErrors(cudaFreeHost(labels[i]));
	}

}

void Image_preparator::test_func()
{
	auto img_name = "E:\\Kirill_testing\\data\\data\\1\\image\\A476C008_161208_L42L_V5-0040_00000.png";
	auto gt_name = "E:\\Kirill_testing\\data\\data\\34\\alpha\\B493C009_161210_R161_V5-0539_00000.png";
	std::vector<float> img, gt;
	get_next_dim_train_image(img_name, gt_name, img, gt);
}

void Image_preparator::check_dim_image(float * res, float * label, int height, int width)
{
	checkCudaErrors(cudaDeviceSynchronize());
	float* h_res;
	float* h_label;

	h_res = new float[height * width];
	h_label = new float[height * width];
	checkCudaErrors(cudaMemcpy(h_res, res, height * width * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_label, label, height * width * sizeof(float), cudaMemcpyDeviceToHost));

	using namespace cv;

	Mat r(height, width, CV_32FC1);
	Mat l(height, width, CV_32FC1);
	memcpy(r.data, h_res, height * width * sizeof(float));
	memcpy(l.data, h_label, height * width * sizeof(float));

	r.convertTo(r, CV_8UC1);
	l.convertTo(l, CV_8UC1);

	int rc=0, lc=0;
	for (int i = 0; i < height * width; ++i)
	{
		if (h_res[i] > 128)
			rc++;
		if (h_label[i] > 128)
			lc++;
	}
	std::cout << rc << " " << lc << std::endl;
	//std::cout << r << std::endl;

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", l); // Show our image inside it.
	namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window2", r); // Show our image inside it.


	//for (int i = 0; i < 320 * 2; ++i)
	//	std::cout << h_res[i] << " " << h_label[i] << "; ";
	//std::cout<<std::endl;

	//resize(r, r, Size(), 1.0f / 10.0f, 1.0f / 10.0f);

	//std::cout << r << std::endl;
	//namedWindow("Display window3", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window3", r); // Show our image inside it.


	waitKey(0); // Wait for a keystroke in the window
	destroyWindow("Display window");



}

void Image_preparator::check_dim_and_stay(float * res, float * label, float * image, int height, int width)
{
	checkCudaErrors(cudaDeviceSynchronize());
	float* h_res;
	float* h_label;
	float* h_image;

	h_res = new float[height*width];
	h_label = new float[height*width];
	h_image = new float[height*width * 4];
	checkCudaErrors(cudaMemcpy(h_res, res, height*width * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_label, label, height*width * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_image, image, height*width * 4 * sizeof(float), cudaMemcpyDeviceToHost));

	using namespace cv;

	Mat r(height, width, CV_32FC1);
	Mat l(height, width, CV_32FC1);
	memcpy(r.data, h_res, height*width * sizeof(float));
	memcpy(l.data, h_label, height*width * sizeof(float));
	std::vector<Mat> splitted_im, alpha_split, ast;

	for (int i = 0; i < 3; i++)
	{		
		Mat tmp(height, width, CV_32FC1);
		memcpy(tmp.data, &h_image[i*height*width], height*width * sizeof(float));
		splitted_im.push_back(tmp);	
		Mat tmp2(height, width, CV_32FC1);
		memcpy(tmp2.data, h_res, height*width * sizeof(float));
		alpha_split.push_back(tmp2);
		Mat tmp3(height, width, CV_32FC1);
		memcpy(tmp3.data, h_res, height*width * sizeof(float));
		threshold(tmp3, tmp3, 0.5f, 1.0f, CV_THRESH_BINARY);
		ast.push_back(tmp3);
	}
	Mat trimap(height, width, CV_32FC1);
	memcpy(trimap.data, &h_image[3*height*width], height*width * sizeof(float));

	Mat im, res_im(height,width,CV_32FC3), res_im_thres(height, width, CV_32FC3);
	merge(splitted_im, im);	
	//splitted_im.push_back(r);
	//divide(splitted_im[3], Scalar(1.0f / 255.0f), splitted_im[3]);
	//merge(splitted_im, res_im);
	im = im + cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2]);
	//res_im = res_im + cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2],0);
	
	Mat alpha(height, width, CV_32FC3);
	merge(alpha_split, alpha);
	Mat at(height, width, CV_32FC3);
	merge(ast, at);
	//divide(alpha, Scalar(255.0f, 255.0f, 255.0f), alpha);
	//memcpy(alpha.data, h_res, height*width * sizeof(float));
	//alpha.convertTo(alpha, CV_32FC3, 1.0f / 255.0f);
	//std::cout << alpha.size << " " << alpha.channels() << " " << alpha.type()<< " "<<
		//im.size << " " << im.channels() << " " << im.type()<<" "<< res_im.size() << " " << res_im.channels() << std::endl;
	multiply(alpha, im, res_im);
	multiply(at, im, res_im_thres);
	
	//std::cout << im.channels() << std::endl;
	
	//memcpy(im.data, h_image, height*width * 3 * sizeof(float));	
	//r.convertTo(r, CV_8UC1);
	//l.convertTo(l, CV_8UC1);
	//trimap.convertTo(trimap, CV_8UC1);
	im.convertTo(im, CV_8UC3);
	//cvtColor(res_im, res_im, CV_BGR2BGRA);
	res_im.convertTo(res_im, CV_8UC3);
	res_im_thres.convertTo(res_im_thres, CV_8UC3);

	//namedWindow("Ground Truth Alpha", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Ground Truth Alpha", l); // Show our image inside it.
	//namedWindow("Alpa Prediction", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Alpa Prediction", r); // Show our image inside it.	
	//namedWindow("Original Image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original Image", im); // Show our image inside it.
	imshow("Trimap", trimap); // Show our image inside it.
	imshow("Composition", res_im); // Show our image inside it.
	imshow("Composition with threshold", res_im_thres); // Show our image inside it.

	//resize(r, r, Size(), 1.0f / 5.0f, 1.0f / 5.0f);
	//std::cout << r << std::endl;
	
	waitKey(10); // Wait for a keystroke in the window	
	delete[] h_res;
	delete[] h_label;
	delete[] h_image;
}

void Image_preparator::save_dim_pic(const char * save_path, float * res, float * image, int height, int width, int num)
{
	checkCudaErrors(cudaDeviceSynchronize());
	float* h_res;
	float* h_image;

	h_res = new float[height*width];
	h_image = new float[height*width * 4];
	checkCudaErrors(cudaMemcpy(h_res, res, height*width * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_image, image, height*width * 4 * sizeof(float), cudaMemcpyDeviceToHost));

	using namespace cv;

	Mat r(height, width, CV_32FC1);
	memcpy(r.data, h_res, height*width * sizeof(float));
	std::vector<Mat> splitted_im, alpha_split;
	for (int i = 0; i < 3; i++)
	{
		Mat tmp(height, width, CV_32FC1);
		memcpy(tmp.data, &h_image[i*height*width], height*width * sizeof(float));
		splitted_im.push_back(tmp);
		Mat tmp2(height, width, CV_32FC1);
		memcpy(tmp2.data, h_res, height*width * sizeof(float));
		threshold(tmp2, tmp2, 0.5f, 1.0f, CV_THRESH_BINARY);
		alpha_split.push_back(tmp2);
	}

	Mat im, res_im(height, width, CV_32FC3);
	merge(splitted_im, im);
	splitted_im.push_back(r);
	im = im + cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2]);

	Mat alpha(height, width, CV_32FC3);
	merge(alpha_split, alpha);
	//divide(alpha, Scalar(255.0f, 255.0f, 255.0f), alpha);
	multiply(alpha, im, res_im);


	//r.convertTo(r, CV_8UC1);
	//im.convertTo(im, CV_8UC3);	
	res_im.convertTo(res_im, CV_8UC3);
	std::string filename = save_path + std::string("\\") + std::to_string(num) + std::string(".png");
	imwrite(filename, res_im);
	delete[] h_res;
	delete[] h_image;
}

void Image_preparator::val_img_prep_th(int num)
{
	assert(num < thread_num);
	std::cout << "Validation image prep thread #" << num << " started!" << std::endl;

	size_t iter = validation_size / thread_num;
	std::vector<float> img;
	std::vector<float> fl_img;
	float *batch = batches[num];
	float *label = labels[num];
	int &height = val_param[num].first;
	int &width = val_param[num].second;
	float *flipped_im = flipped[num];

	for (int i = 0; i < iter; i++)
	{
		//std::cout << "w8 for prep\n";
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });

		if (!run)
			return;

		get_next_val_image(filenames[i*thread_num + num].generic_string().c_str(), height, width, img, fl_img);
		std::copy(img.begin(), img.end(), &batch[0]);
		std::copy(fl_img.begin(), fl_img.end(), &flipped_im[0]);
		label[0] = vec_labels[i*thread_num + num];

		ready_to_load[num] = false;
		lock.unlock();
		//std::cout << "ready prep\n";
		std::lock_guard<std::mutex> q_lock(q_mutex);
		queue.push_back(num);
		cv.notify_one();
	}

}

void Image_preparator::val_img_prep_th_2(int num)
{
	assert(num < thread_num);
	std::cout << "Validation image prep thread #" << num << " started!" << std::endl;
	
	size_t iter = train_size / thread_num;
	std::vector<float> img;
	std::vector<float> fl_img;
	float *batch = batches[num];
	float *label = labels[num];
	int &height = val_param[num].first;
	int &width = val_param[num].second;
	float *flipped_im = flipped[num];

	for (int i = 0; i < iter; i++)
	{
		//std::cout << "w8 for prep\n";
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });

		if (!run)
			return;

		get_next_val_image(th_file_map[num][i].first.c_str(), height, width, img, fl_img);
		std::copy(img.begin(), img.end(), &batch[0]);
		std::copy(fl_img.begin(), fl_img.end(), &flipped_im[0]);
		label[0] = th_file_map[num][i].second;

		ready_to_load[num] = false;
		lock.unlock();
		//std::cout << "ready prep\n";
		std::lock_guard<std::mutex> q_lock(q_mutex);
		queue.push_back(num);
		cv.notify_one();
	}
}

void Image_preparator::train_img_prep_th(int num)
{
	assert(num < thread_num);
	//std::cout << "Train image prep thread #" << num << " started!" << std::endl;

	size_t iter = th_file_map[num].size() / batchSize;
	//std::cout << iter << " " << num << std::endl;
	std::vector<float> img;
	float *batch = batches[num];
	float *label = labels[num];

	for (int i = 0; i < iter; i++)
	{
		//std::cout << "w8 " << num << std::endl;
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });
		//std::cout << "end w8 " << num << std::endl;

		if (!run)
			return;

		for (int j = 0; j < batchSize; j++)
		{
			get_next_train_image(th_file_map[num][i*batchSize + j].first.c_str(), img);
			std::copy(img.begin(), img.end(), &batch[img.size() * j]);
			label[j] = th_file_map[num][i*batchSize + j].second;
			//std::cout << label[j] << std::endl;
		}

		ready_to_load[num] = false;
		lock.unlock();
		std::lock_guard<std::mutex> q_lock(q_mutex);
		queue.push_back(num);
		//std::cout << "notify " << num << std::endl;
		cv.notify_one();
	}

}

void Image_preparator::get_next_train_image(const char * filename, std::vector<float>& img)
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

	image.convertTo(image, CV_32FC3/*, 1 / 255.0*/);
	
	
	int m = min(image.rows, image.cols);

	std::random_device rd;
	std::mt19937 gen(rd());

	int maxS = 512;
	int minS = 256;

	std::uniform_int_distribution<> d(minS, maxS);

	int S = d(gen);
	S = 256;

	float ratio = float(S) / float(m);
	//int xsize = image.cols * ratio;
	//int ysize = image.rows * ratio;			
	cv::resize(image, image, cv::Size(), ratio, ratio);	

	int xstart, ystart;
	std::uniform_int_distribution<> xr(0, S - 224);
	std::uniform_int_distribution<> yr(0, S - 224);

	xstart = xr(gen);
	ystart = yr(gen);
	//xstart = 0;
	//ystart = 0;

	////

	Rect r(xstart, ystart, 224, 224);
	image = image(r);

	Scalar mean, stddev;
	Scalar adj_stddev(3);
	meanStdDev(image, mean, stddev);
	//outIm.convertTo(outIm, CV_32FC1);
	image = image - mean; // subtract mean
	adj_stddev[0] = std::max(stddev[0], 1.0 / std::sqrt(image.cols * image.rows)); // TF suggest to use this to prevent division by 0
	adj_stddev[1] = std::max(stddev[1], 1.0 / std::sqrt(image.cols * image.rows)); // TF suggest to use this to prevent division by 0
	adj_stddev[2] = std::max(stddev[2], 1.0 / std::sqrt(image.cols * image.rows)); // TF suggest to use this to prevent division by 0

	divide(image, adj_stddev, image); // divide by standard deviation
	//// Subtracting mean RGB value
	//outIm = outIm - cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2]);
		
	auto size = image.size();
	
	cv::Size newsize(size.width, size.height * 3);
	cv::Mat destination(newsize, CV_32FC1);
	
	Mat flipped;

	std::uniform_int_distribution<> fl(0, 1);

	int flip = fl(gen);
	//flip = 0;

	if (flip)
	{
		cv::flip(image, image, 1);		
	}

	
	//copy the channels from the source image to the destination
	for (int i = 0; i < image.channels(); ++i)
	{
		cv::extractChannel(
			image,
			cv::Mat(
				size.height,
				size.width,
				CV_32FC1,
				&(destination.at<float>(size.height*size.width*i))),
			i);
	}
		
	img.clear();
	if (destination.isContinuous())
		img.assign((float*)destination.datastart, (float*)destination.dataend);
	else
		for (int i = 0; i < destination.rows; ++i)
			img.insert(img.end(), destination.ptr<float>(i), destination.ptr<float>(i) + destination.cols);	
	//cv::cvtColor(im_f, im_f, CV_RGB2BGR);
		
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", destination); // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window
	//destroyWindow("Display window");
}

void Image_preparator::get_next_val_image(const char* filename, int &height, int &width, std::vector<float> &img, std::vector<float> &flipped)
{
	//std::cout << filename << std::endl;

	using namespace cv;
	using namespace std;

	Mat image;
	image = imread(filename, IMREAD_COLOR);

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}

	//// Note: these convertion should NOT be done on dataset for VGG since they expect data in BGR and [0,255]-mean!!!!
	//cv::cvtColor(image, image, CV_BGR2RGB);
	//Mat im_f;
	image.convertTo(image, CV_32FC3);
	//image = im_f;
	////

	float m = min(image.rows, image.cols);

	Mat outIm;
	//int Q = 256; // in case of static S
	int Q = 384; // in case of S = rand(256,512)

	//not sure where it should happend
	//// Subtracting mean RGB value
	//outIm = outIm - cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2]);

	float ratio = float(Q) / m;
	cv::resize(image, outIm, cv::Size(), ratio, ratio);

	//// Subtracting mean RGB value
	outIm = outIm - cv::Scalar(VGG_MEAN[0],VGG_MEAN[1],VGG_MEAN[2]);

	auto size = outIm.size();
	cv::Size newsize(size.width, size.height * 3);
	cv::Mat destination(newsize, CV_32FC1);
	//std::cout << size<<newsize << std::endl;
	//copy the channels from the source image to the destination
	for (int i = 0; i < outIm.channels(); ++i)
	{
		cv::extractChannel(
			outIm,
			cv::Mat(
				size.height,
				size.width,
				CV_32FC1,
				&(destination.at<float>(size.height*size.width*i))),
			i);
	}
	height = size.height;
	width = size.width;

	//std::cout << "Chan for im" << std::endl;

	img.clear();
	if (destination.isContinuous())
		img.assign((float*)destination.datastart, (float*)destination.dataend);
	else
		for (int i = 0; i < destination.rows; ++i)
			img.insert(img.end(), destination.ptr<float>(i), destination.ptr<float>(i) + destination.cols);

	cv::Mat im_f;
	cv::Mat flip_dest(newsize, CV_32FC1);
	cv::flip(outIm, im_f, 1);
	//copy the channels from the source image to the destination
	for (int i = 0; i < im_f.channels(); ++i)
	{
		cv::extractChannel(
			im_f,
			cv::Mat(
				size.height,
				size.width,
				CV_32FC1,
				&(flip_dest.at<float>(size.height*size.width*i))),
			i);
	}
	//std::cout << "Chan for flipped" << std::endl;
	flipped.clear();
	if (flip_dest.isContinuous())
		flipped.assign((float*)flip_dest.datastart, (float*)flip_dest.dataend);
	else
		for (int i = 0; i < flip_dest.rows; ++i)
			flipped.insert(flipped.end(), flip_dest.ptr<float>(i), flip_dest.ptr<float>(i) + flip_dest.cols);
}

void Image_preparator::next_cifar_train_img(int num, std::vector<float>& img)
{
	auto &data = cifar_data[num].second;
	//std::cout << data.size();
	img.clear();

	std::random_device rd;
	std::mt19937 gen(rd());
	int xstart, ystart;
	std::uniform_int_distribution<> xr(0, 8);
	std::uniform_int_distribution<> yr(0, 8);

	xstart = xr(gen);
	ystart = yr(gen);

	std::uniform_int_distribution<> fl(0, 1);
	int flip = fl(gen);
	if (flip)
	{
		for (int c = 0; c < 3; c++)
			for (int h = ystart; h < ystart + 24; ++h)
				for (int w = xstart; w < xstart + 24; ++w)
					img.push_back(data[c * 32 * 32 + h * 32 + w]);
	}
	else
	{
		for (int c = 0; c < 3; c++)
			for (int h = ystart; h < ystart + 24; ++h)
				for (int w = xstart + 23; w >= xstart; --w)
					img.push_back(data[c * 32 * 32 + h * 32 + w]);
	}

	//using namespace cv;
	//cv::Mat res(24*3, 24, CV_32FC1, (float*)img.data());
	//resize(res, res, cv::Size(), 10, 10);
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", res); // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window
	//destroyWindow("Display window");

	//std::cout << img.size() << std::endl;

}

void Image_preparator::next_cifar_test_img(int num, std::vector<float>& img)
{
	auto &data = cifar_data[num].second;
	img.clear();

	// now image is centred
	int xstart, ystart;
	xstart = 4;
	ystart = 4;

	for (int c = 0; c < 3; c++)
		for (int h = ystart; h < ystart + 24; ++h)
			for (int w = xstart; w < xstart + 24; ++w)
				img.push_back(data[c * 32 * 32 + h * 32 + w]);
}

void Image_preparator::cifar_prep_th()
{
	//std::cout << "Train image prep thread #" << num << " started!" << std::endl;

	int num = 0; // there should be only 1 thread
	int iter = cifar_data.size() / batchSize;
	std::vector<float> img;
	float *batch = batches[num];
	float *label = labels[num];

	for (int i = 0; i < iter; i++)
	{
		//std::cout << "w8 " << num << std::endl;
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });
		//std::cout << "end w8 " << num << std::endl;

		if (!run)
			return;

		for (int j = 0; j < batchSize; j++)
		{
			next_cifar_train_img(i + j, img);
			//std::cout << img.size() << std::endl;
			std::copy(img.begin(), img.end(), &batch[img.size() * j]);
			label[j] = cifar_data[i + j].first;
		}
		//std::cout << sizeof(batch) << std::endl;
		ready_to_load[num] = false;
		lock.unlock();
		std::lock_guard<std::mutex> q_lock(q_mutex);
		queue.push_back(num);
		//std::cout << "notify " << num << std::endl;
		cv.notify_one();
	}
}

void Image_preparator::cifar_prep_val_th()
{
	int num = 0; // there should be only 1 thread
	int iter = cifar_data.size() / batchSize;
	std::vector<float> img;
	float *batch = batches[num];
	float *label = labels[num];

	for (int i = 0; i < iter; i++)
	{
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });

		if (!run)
			return;

		for (int j = 0; j < batchSize; j++)
		{
			next_cifar_test_img(i + j, img);
			std::copy(img.begin(), img.end(), &batch[img.size() * j]);
			label[j] = cifar_data[i + j].first;
		}
		ready_to_load[num] = false;
		lock.unlock();
		std::lock_guard<std::mutex> q_lock(q_mutex);
		queue.push_back(num);
		cv.notify_one();
	}
}

void Image_preparator::train_dim_prep_th(int num)
{
	assert(num < thread_num);
	std::cout << "Prep dim thread #" << num << " started!" << std::endl;
	size_t iter = th_dim_files[num].size() / batchSize;	
	std::vector<float> img;
	std::vector<float> gt;
	float *batch = batches[num];
	float *label = labels[num];

	for (int i = 0; i < iter; i++)
	{		
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });
	
		if (!run)
			return;

		for (int j = 0; j < batchSize; j++)
		{
			auto p = th_dim_files[num][i*batchSize + j];
			//get_next_train_image(th_dim_files[num][i*batchSize + j].first.c_str(), img);
			get_next_dim_train_image(p.first.c_str(), p.second.c_str(), img, gt);
			//get_next_dim_train_image(dim_files[i*batchSize*thread_num + num*batchSize + j].first.c_str(), dim_files[i*batchSize*thread_num + num*batchSize + j].second.c_str(), img, gt);
			std::copy(img.begin(), img.end(), &batch[img.size() * j]);
			std::copy(gt.begin(), gt.end(), &label[gt.size() * j]);
			//label[j] = th_file_map[num][i*batchSize + j].second;
			//std::cout << label[j] << std::endl;
		}

		ready_to_load[num] = false;
		lock.unlock();
		std::lock_guard<std::mutex> q_lock(q_mutex);
		queue.push_back(num);	
		cv.notify_one();
	}
}

void Image_preparator::train_dim_prep_th_no_white(int num)
{
	assert(num < thread_num);
	//std::cout << "Prep dim thread #" << num << " started!" << std::endl;
	//size_t iter = th_dim_files[num].size() / batchSize;
	std::vector<float> img;
	std::vector<float> gt;
	float *batch = batches[num];
	float *label = labels[num];

	while (!th_dim_files[num].empty())
	{
		//std::cout << "W8 lock" << std::endl;
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });
		//std::cout << "W8 lock done" << std::endl;

		if (!run)
			break;

		int j = 0;
		while (j < batchSize && !th_dim_files[num].empty())
		{
			auto p = th_dim_files[num].back();
			if (get_next_dim_train_image(p.first.c_str(), p.second.c_str(), img, gt))
			{
				std::copy(img.begin(), img.end(), &batch[img.size() * j]);
				std::copy(gt.begin(), gt.end(), &label[gt.size() * j]);
				j++;
			}
			th_dim_files[num].pop_back();
		}
		if (j == batchSize)
		{
			ready_to_load[num] = false;
			lock.unlock();
			std::lock_guard<std::mutex> q_lock(q_mutex);
			queue.push_back(num);
			cv.notify_one();
		}
	}
	//std::cout << "Prep dim thread #" << num << " over!" << std::endl;
}

void Image_preparator::eval_dim_th(int num)
{
	assert(num < thread_num);
	std::cout << "Eval DIM thread #" << num << " started!" << std::endl;
	size_t iter = dim_files.size() / (thread_num * batchSize);
	std::vector<float> img;
	std::vector<float> gt;
	float *batch = batches[num];
	float *label = labels[num];

	for (int i = 0; i < iter; i++)
	{
		std::unique_lock<std::mutex> lock((*mutices)[num]);
		(*load_cvs)[num].wait(lock, [this, num] {return ready_to_load[num]; });

		if (!run)
			return;

		for (int j = 0; j < batchSize; j++)
		{
			get_next_dim_eval_image(dim_files[i*batchSize*thread_num + num*batchSize + j].first.c_str(), dim_files[i*batchSize*thread_num + num*batchSize + j].second.c_str(), img, gt);
			std::copy(img.begin(), img.end(), &batch[img.size() * j]);
			std::copy(gt.begin(), gt.end(), &label[gt.size() * j]);
			//label[j] = th_file_map[num][i*batchSize + j].second;
			//std::cout << label[j] << std::endl;
		}

		ready_to_load[num] = false;
		lock.unlock();
		std::lock_guard<std::mutex> q_lock(q_mutex);
		queue.push_back(num);
		cv.notify_one();
	}
}


cv::Mat create_trimap(cv::Mat &grd_tr)
{
	using namespace cv;
	/// Global variables
	Mat src, dilation_dst;	
	//threshold(grd_tr, grd_tr, 50, 255, CV_THRESH_BINARY);
	grd_tr.copyTo(src);
	//threshold(grd_tr, grd_tr, 50, 255, CV_THRESH_BINARY);
	//threshold(grd_tr, grd_tr, 50, 255, CV_THRESH_TOZERO);

	int dilation_elem = 0;
	int dilation_size = 10;
	int const max_elem = 2;
	int const max_kernel_size = 21;

	std::random_device rd;
	std::mt19937 gen(rd());

	int min_d = 8;
	int max_d = 15;

	std::uniform_int_distribution<> d(min_d, max_d);
	dilation_size = d(gen);
	
	Mat src_gray;
	Mat dst, detected_edges;

	int edgeThresh = 1;
	int lowThreshold = 0;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;

	if (!src.data)
	{
		std::cout << "No data!" << std::endl;
		system("pause");
		exit(5);
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale
	//cvtColor(src, src_gray, CV_BGR2GRAY);	
	src_gray = src;

	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));
	//detected_edges = src_gray;
	
	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	Mat origin = src;

	src = dst;

	int dilation_type;
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	dilate(src, dilation_dst, element);
	//imshow("Dilation Demo", dilation_dst);

	origin.convertTo(origin, CV_32FC1, 1.0 / 255);
	grd_tr.convertTo(grd_tr, CV_32FC1, 1.0 / 255);
	Mat I = origin;
	//std::cout << origin.size << dilation_dst.size << std::endl;

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

	if (I.isContinuous() && dilation_dst.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	int i, j;
	//uchar* p, *pp, *ppp;
	float* p;
	uchar *pp;
	for (i = 0; i < nRows; ++i)
	{
		//std::cout << nRows << " " << nCols << std::endl;
		p = I.ptr<float>(i);
		pp = dilation_dst.ptr<uchar>(i);
		//ppp = detected_edges.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{			
			if (pp[j] != 0)
			{
				p[j] = 0.5;
				continue;
			}

			// fixing dilation not working with vertical lines
			//if (ppp != 0)
			//{

			//}
		}
	}
	//imshow("test1", grd_tr);	
	//imshow("test2", origin);
	//waitKey(0);
	return origin;
}

bool crop(cv::Mat &image, cv::Mat &grd_tr, cv::Mat &trimap)
{
	using namespace cv;
	using namespace std;
		
	int channels = trimap.channels();
	//cout << channels << " " << trimap.size << endl;
	assert(channels == 1);
	assert(trimap.rows == 1080);
	assert(trimap.cols == 1920);
	int nRows = trimap.rows;
	int nCols = trimap.cols;
	
	std::random_device rd;
	std::mt19937 gen(rd());

	int min = 320;
	int max = 640;
	
	std::uniform_int_distribution<> d(min, max);
	int crop_size = d(gen);

	int px = 0, py = 0;

	std::uniform_int_distribution<> dx(crop_size / 2 + 1, nCols - crop_size / 2 - 1);
	std::uniform_int_distribution<> dy(crop_size / 2 + 1, nRows - crop_size / 2 - 1);

	int counter = 0;
	// I dont like it to center on unkown region... yet following the paper
	std::vector<std::pair<int, int>> unkn_reg;

	for (int h = crop_size / 2 + 1; h < nRows - crop_size / 2 - 1; ++h)
		for (int w = crop_size / 2 + 1; w < nCols - crop_size / 2 - 1; ++w)
			if (trimap.at<float>(h, w) == 0.5)
				unkn_reg.push_back(std::make_pair(h, w));

	if (unkn_reg.empty())
		return false;
	else
	{
		std::uniform_int_distribution<> pos(0, unkn_reg.size() - 1);
		auto ppos = pos(gen);
		py = unkn_reg[ppos].first;
		px = unkn_reg[ppos].second;
	}

	while (px == 0 && 0)
	{
		px = dx(gen);
		py = dy(gen);

		if (trimap.at<float>(py, px) != 0.5)
		{
			px = 0;
			py = 0;
			counter++;
			if (counter >= 1000000)
			{
				return false;
				cout << "Searching for too long" << endl;
				px = dx(gen);
				py = dy(gen);
			}
		}
	}
	//std::cout << px << " " << py << " "<< counter << " " << int(trimap.at<uchar>(py, px)) << std::endl;

	Rect r(px - crop_size / 2, py - crop_size / 2, crop_size, crop_size);
	image = image(r);
	grd_tr = grd_tr(r);
	trimap = trimap(r);

	float ratio = 320.0f / static_cast<float>(crop_size);

	resize(image, image, Size(), ratio, ratio);
	resize(grd_tr, grd_tr, Size(), ratio, ratio);
	resize(trimap, trimap, Size(), ratio, ratio);

	//image.convertTo(image, CV_8UC3);

	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.
	//
	//namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window2", grd_tr); // Show our image inside it.
	//
	//namedWindow("Display window3", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window3", trimap); // Show our image inside it.
	//waitKey(0); // Wait for a keystroke in the window
	//destroyWindow("Display window");
	//destroyWindow("Display window2");
	//destroyWindow("Display window3");
	return true;
}

bool Image_preparator::get_next_dim_train_image(const char * img_name, const char * gt_name, std::vector<float>& img, std::vector<float>& gt)
{
	//std::cout << img_name << std::endl << gt_name << std::endl;
	using namespace cv;
	using namespace std;

	Mat image, grd_tr;
	image = imread(img_name, IMREAD_COLOR);
	grd_tr = imread(gt_name, IMREAD_GRAYSCALE);

	if (image.empty() || grd_tr.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return false;
	}

	// BGR color mode - since weights pre-trained on BGR
	
	//image.convertTo(image, CV_32FC3, 1.0f / 255);
	//grd_tr.convertTo(grd_tr, CV_32FC1, 1.0f / 255);
	//trimap.convertTo(trimap, CV_32FC1);

	// create trimap from ground truth
	Mat trimap = create_trimap(grd_tr); // and fix grd_tr
	//imshow("test", trimap);
	//waitKey(0);
			
	// crop, centered on pixels in the unkrown area
	if (!crop(image, grd_tr, trimap)) // and resize to 320x320
		return false;
	//cout << image.size << " " << grd_tr.size << " " << trimap.size << endl;
	
	// covert to float
	image.convertTo(image, CV_32FC3);
	//grd_tr.convertTo(grd_tr, CV_32FC1);
	//trimap.convertTo(trimap, CV_32FC1);
	
	Scalar mean, stddev;	
	meanStdDev(image, mean, stddev);
	//outIm.convertTo(outIm, CV_32FC1);
	//image = image - mean; // subtract mean
	// or
	image = image - cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2]);

	std::random_device rd;
	std::mt19937 gen(rd());		
	std::uniform_int_distribution<> fl(0, 1);
	int flip = fl(gen);	

	if (flip)
	{
		cv::flip(image, image, 1);
		cv::flip(grd_tr, grd_tr, 1);
		cv::flip(trimap, trimap, 1);
	}

	auto size = image.size();
	cv::Size newsize(size.width, size.height * 3);
	cv::Mat destination(newsize, CV_32FC1);
	// copy the channels from the source image to the destination
	for (int i = 0; i < image.channels(); ++i)
	{
		cv::extractChannel(
			image,
			cv::Mat(
				size.height,
				size.width,
				CV_32FC1,
				&(destination.at<float>(size.height*size.width*i))),
			i);
	}
	Mat res;
	vconcat(destination, trimap, res);

	destination = res;
	//resize(destination, destination, Size(), 1.0f / 3.0f, 1.0f / 3.0f);

	img.clear();
	if (destination.isContinuous())
		img.assign((float*)destination.datastart, (float*)destination.dataend);
	else
		for (int i = 0; i < destination.rows; ++i)
			img.insert(img.end(), destination.ptr<float>(i), destination.ptr<float>(i) + destination.cols);

	gt.clear();
	if (grd_tr.isContinuous())
		gt.assign((float*)grd_tr.datastart, (float*)grd_tr.dataend);
	else
		for (int i = 0; i < grd_tr.rows; ++i)
			gt.insert(gt.end(), grd_tr.ptr<float>(i), grd_tr.ptr<float>(i) + grd_tr.cols);
	//cv::cvtColor(im_f, im_f, CV_RGB2BGR);

	//destination.convertTo(destination, CV_8UC1);

	//Mat r(320, 320, CV_32FC1);
	//memcpy(r.data, gt.data(), 320 * 320 * sizeof(float));
	//r.convertTo(r, CV_8UC1);
	
	//grd_tr.convertTo(grd_tr, CV_8UC1);
	//trimap.convertTo(trimap, CV_8UC1);
	//destination.convertTo(destination, CV_8UC1);
	//std::cout << "Test: " << gt.size() << std::endl;
	//image = image + cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2]);
	//image.convertTo(image, CV_8UC3);

	////namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	////imshow("Display window", grd_tr); // Show our image inside it.
	//namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window2", image); // Show our image inside it.
	//waitKey(1); // Wait for a keystroke in the window
	//destroyWindow("Display window");
	//run = false;
	return true;
}

void Image_preparator::get_next_dim_eval_image(const char * img_name, const char * gt_name, std::vector<float>& img, std::vector<float>& gt)
{
	//std::cout << img_name << std::endl << gt_name << std::endl;
	using namespace cv;
	using namespace std;

	Mat image, grd_tr;
	image = imread(img_name, IMREAD_COLOR);
	grd_tr = imread(gt_name, IMREAD_GRAYSCALE);

	if (image.empty() || grd_tr.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}

	// BGR color mode - since weights pre-trained on BGR

	// create trimap from ground truth
	Mat trimap = create_trimap(grd_tr);
	
	// convert to float
	image.convertTo(image, CV_32FC3);
	//grd_tr.convertTo(grd_tr, CV_32FC1);
	//trimap.convertTo(trimap, CV_32FC1);

	Scalar mean, stddev;
	meanStdDev(image, mean, stddev);
	//outIm.convertTo(outIm, CV_32FC1);
	//image = image - mean; // subtract mean
						  // or
	image = image - cv::Scalar(VGG_MEAN[0], VGG_MEAN[1], VGG_MEAN[2]);

	auto size = image.size();

	cv::Size newsize(size.width, size.height * 3);
	cv::Mat destination(newsize, CV_32FC1);

	//std::uniform_int_distribution<> fl(0, 1);
	//int flip = fl(gen);

	//if (flip)
	//{
	//	cv::flip(image, image, 1);
	//	cv::flip(grd_tr, grd_tr, 1);
	//	cv::flip(trimap, trimap, 1);
	//}

	// copy the channels from the source image to the destination
	for (int i = 0; i < image.channels(); ++i)
	{
		cv::extractChannel(
			image,
			cv::Mat(
				size.height,
				size.width,
				CV_32FC1,
				&(destination.at<float>(size.height*size.width*i))),
			i);
	}
	Mat res;
	vconcat(destination, trimap, res);

	destination = res;
	img.clear();
	if (destination.isContinuous())
		img.assign((float*)destination.datastart, (float*)destination.dataend);
	else
		for (int i = 0; i < destination.rows; ++i)
			img.insert(img.end(), destination.ptr<float>(i), destination.ptr<float>(i) + destination.cols);

	gt.clear();
	if (grd_tr.isContinuous())
		gt.assign((float*)grd_tr.datastart, (float*)grd_tr.dataend);
	else
		for (int i = 0; i < grd_tr.rows; ++i)
			gt.insert(gt.end(), grd_tr.ptr<float>(i), grd_tr.ptr<float>(i) + grd_tr.cols);
}

void Image_preparator::prepare_filenames()
{
	//std::cout << train_labels_path << std::endl;
	// read a JSON file
	std::ifstream in(train_labels_path);
	using json = nlohmann::json;
	json j;
	in >> j;	
	std::map<std::string, int> tags_map;
	std::string str;
	for (json::iterator it = j.begin(); it != j.end(); ++it)
	{
		str = it.value()[0].dump();
		str.pop_back();
		str.erase(str.begin());
		tags_map[str] = atoi(it.key().c_str());
	}

	namespace fs = std::experimental::filesystem;
	int k = 0;
	
	for (auto & p : fs::directory_iterator(train_image_path))
	{		
		fs::path str = p;		
		if (str.has_filename())
		{	
			int label = tags_map[str.filename().generic_string()];
			//std::cout << label << std::endl;
			for (auto & sub : fs::directory_iterator(str))
			{
				//k++;
				fs::path name = sub;
				auto str1 = name.generic_string();
				file_map.push_back(std::make_pair(name.generic_string(), label));
				//if (train_size > 0 && k >= train_size)
				//	break;
			}
		}
		//if (train_size > 0 && k >= train_size)
		//	break;
	}

	if (train_size > file_map.size())
	{
		std::cout << "Stated directory contains less files than mentioned in train_size...\n Aborting...\n";
		system("pause");
		exit(3);
	}

	//size_t pic_per_th = file_map.size() / thread_num;

	size_t pic_per_th = train_size / thread_num; // it would be better if train_size would be multiply of thread_num
	//std::cout << "Trainig set contains: " << file_map.size() << " images and ppt= "<<pic_per_th<< std::endl;
	std::cout << "Trainig set contains: " << train_size << "out of " << file_map.size()<<" total images and ppt= " << pic_per_th << std::endl;
	
	for (int i = 0; i<thread_num; i++)
		th_file_map[i].resize(pic_per_th);
}

void Image_preparator::prepare_dim_filenames()
{
	//std::cout << train_labels_path << std::endl;

	namespace fs = std::experimental::filesystem;
	int k = 0;

	for (auto & p : fs::directory_iterator(train_image_path))
	{
		fs::path str = p;
		//std::cout << str << std::endl;
		if (str.has_filename())
		{
			std::vector<std::string> images;
			std::vector<std::string> gt;
			for (auto & sub : fs::directory_iterator(str))
			{				
				//k++;
				fs::path name = sub;
				//std::cout << name << std::endl;
				if (name.filename().generic_string() == "image")
				{
					for (auto &dst : fs::directory_iterator(name))
					{
						fs::path filename = dst;
						//std::cout << filename << std::endl;
						if (filename.extension().generic_string() == ".png")
							images.push_back(filename.generic_string());
					}
				}
				if (name.filename().generic_string() == "alpha")
				{
					for (auto &dst : fs::directory_iterator(name))
					{
						fs::path filename = dst;
						//std::cout << filename << std::endl;
						if (filename.extension().generic_string() == ".png")
							gt.push_back(filename.generic_string());
					}
				}							
			}

			if (images.size() != gt.size())
			{
				//std::cout << "something wrong!" << images.size() << " " << gt.size() << std::endl;
				std::wcout << "Missing images in " << str << std::endl;
				//system("pause");
				//exit(4);
			}
			else
				for (int i = 0; i < images.size(); i++)
					dim_files.push_back(std::make_pair(images[i], gt[i]));
		}
	}

	if (train_size > dim_files.size())
	{
		std::cout << "Stated directory contains less files than mentioned in train_size...\n Aborting...\n";
		system("pause");
		exit(3);
	}

	//size_t pic_per_th = file_map.size() / thread_num;
	if (train_size == 0)
		train_size = dim_files.size();

	size_t pic_per_th = train_size / thread_num; // it would be better if train_size would be multiply of thread_num										
	std::cout << "Trainig set contains: " << train_size << "out of " << dim_files.size() << " total images and ppt= " << pic_per_th << std::endl;

	//for (int i = 0; i<thread_num; i++)
	//	th_file_map[i].resize(pic_per_th);
}

void Image_preparator::rand_dim_files()
{	
	size_t pic_per_th = train_size / thread_num;
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(dim_files.begin(), dim_files.end(), g);

	if (train_size < dim_files.size())
		dim_files.erase(dim_files.begin() + train_size, dim_files.end());
		

	for (int i = 0; i < thread_num; i++)
	{
		auto start_itr = std::next(dim_files.cbegin(), i*pic_per_th);
		auto end_itr = std::next(dim_files.cbegin(), (i + 1)*pic_per_th);

		th_dim_files[i].resize(pic_per_th);
		std::copy(start_itr, end_itr, th_dim_files[i].begin());
	}

}

int Image_preparator::prepare_dim_seq()
{
	namespace fs = std::experimental::filesystem;
	int k = 0;


	std::vector<std::string> images;
	std::vector<std::string> gt;
	for (auto & sub : fs::directory_iterator(train_image_path))
	{
		fs::path name = sub;
		if (name.filename().generic_string() == "image")
		{
			for (auto &dst : fs::directory_iterator(name))
			{
				fs::path filename = dst;
				if (filename.extension().generic_string() == ".png")
					images.push_back(filename.generic_string());
			}
		}
		if (name.filename().generic_string() == "alpha")
		{
			for (auto &dst : fs::directory_iterator(name))
			{
				fs::path filename = dst;
				if (filename.extension().generic_string() == ".png")
					gt.push_back(filename.generic_string());
			}
		}
	}

	if (images.size() != gt.size())
	{

		std::cout << "Missing images in " << train_image_path << std::endl;
		system("pause");
		exit(4);
	}
	else
		for (int i = 0; i < images.size(); i++)
			dim_files.push_back(std::make_pair(images[i], gt[i]));



	if (train_size > dim_files.size())
	{
		std::cout << "Stated directory contains less files than mentioned in train_size...\n Aborting...\n";
		system("pause");
		exit(3);
	}

	//size_t pic_per_th = file_map.size() / thread_num;
	if (train_size == 0)
		train_size = dim_files.size();

	size_t pic_per_th = train_size / thread_num; // it would be better if train_size would be multiply of thread_num										
	std::cout << "Trainig set contains: " << train_size << "out of " << dim_files.size() << " total images and ppt= " << pic_per_th << std::endl;

	return train_size;
}

void Image_preparator::validation_preparation()
{
	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaFreeHost(batches[i]));
		checkCudaErrors(cudaFreeHost(labels[i]));
	}

	batchSize = 1;
	val_param.resize(thread_num);
	flipped.resize(thread_num);
	size_t max_bytes = sizeof(float) * 384 * 1600 * 3; // assume that no img will be bigger than 1600*384*3 after preparation
	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaMallocHost(&batches[i], max_bytes));
		checkCudaErrors(cudaMallocHost(&flipped[i], max_bytes));
		checkCudaErrors(cudaMallocHost(&labels[i], sizeof(float)));
	}

	stop();
	run = true;
	//map_and_rand();

	for (int i = 0; i < thread_num; i++)
	{
		threads.emplace_back(&Image_preparator::val_img_prep_th_2, this, i);
	}
}

void Image_preparator::cifar_preparation(std::string cifar_path_, int batchSize_, size_t byte_size_, bool train)
{
	cifar_path = cifar_path_;
	batchSize = batchSize_;
	byte_size = byte_size_;

	for (int i = 0; i < thread_num; i++)
	{
		checkCudaErrors(cudaMallocHost(&batches[i], byte_size));
		checkCudaErrors(cudaMallocHost(&labels[i], sizeof(float) * batchSize));
	}

	if (train)
		cifar_data.resize(50000);
	else
		cifar_data.resize(10000);

	namespace fs = std::experimental::filesystem;
	int k = 0;
	using namespace cv;

	for (auto & p : fs::directory_iterator(cifar_path))
	{
		fs::path str = p;
		std::ifstream is(str.generic_string().c_str(), std::ifstream::binary);
		if (!is)
		{
			std::cout << "Can't open Cifar file " << str << std::endl;
			system("pause");
			exit(2);
		}

		
		Scalar mean, stddev;
		float adj_stddev;

		//for (int i = 1e+4 * k; i < 1e+4 *(k + 1); i++)
		//{
		//	cifar_data[i].second.resize(3072);
		//	is.read((char*)&cifar_data[i].first, 1);
		//	is.read((char*)&cifar_data[i].second[0], 3072);
		//}

		std::vector<double> tmp_vec(1024);
		char tmp;

		for (int i = 1e+4 * k; i < 1e+4 * (k + 1); ++i)
		{
			is.read(&tmp, 1);
			cifar_data[i].first = float(tmp);
			//is.read((char*)&cifar_data[i].first, 1);
			cifar_data[i].second.resize(3072);
			
			for (int ch = 0; ch < 3; ch++)
			{
				Mat img(32, 32, CV_8UC1);
				is.read((char*)img.data, 1024);
				meanStdDev(img, mean, stddev);
				img.convertTo(img, CV_32FC1);
				img = img - mean; // subtract mean
				adj_stddev = std::max(stddev[0], 1.0 / 32.0); // TF suggest to use this to prevent division by 0

				divide(img, Scalar(adj_stddev), img); // divide by standard deviation
				// Mat -> vector
				if (img.isContinuous())
					tmp_vec.assign((float*)img.datastart, (float*)img.dataend);
				else
					std::cout << "cifar_preparation: data not continious!" << std::endl;

				std::copy(tmp_vec.begin(), tmp_vec.end(), &cifar_data[i].second[1024 * ch]); 
			}
		}

		if (!is)
			std::cout << "Something wrong!" << std::endl;
		is.close();
		k++;
	}

	if (!train)
	{
		run = true;
		threads.emplace_back(&Image_preparator::cifar_prep_th, this);
	}
}

void Image_preparator::start_next_cifar_epoch()
{
	stop();

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(cifar_data.begin(), cifar_data.end(), g);

	run = true;
	threads.emplace_back(&Image_preparator::cifar_prep_th, this);
}

