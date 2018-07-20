#pragma once
#include <string>
#include <vector>
#include <thread>
#include <filesystem>
#include <mutex>
#include <condition_variable>

class Image_preparator
{
	std::string validation_image_path;
	std::string validation_labels_path;

	std::string train_image_path;
	std::string train_labels_path;

	std::string cifar_path;

	int thread_num;
	int batchSize;
	std::vector<float*> batches;
	std::vector<float*> labels;

	std::vector<float*> flipped;
	std::vector<std::pair<int, int>> val_param; // height and width for validation

	size_t byte_size;
	size_t train_size;
	size_t validation_size;
	size_t test_size;

	std::vector<std::thread> threads;
	std::vector<float> vec_labels;
	std::vector<std::experimental::filesystem::path> filenames;
	std::vector<std::pair<std::string,std::string>> dim_files; // image_path, gt_path
	std::vector<std::vector<std::pair<std::string, std::string>>> th_dim_files;
	//std::vector<std::experimental::filesystem::path> foldernames;
	std::vector<std::pair<std::string, int>> file_map;
	std::vector<std::vector<std::pair<std::string, int>>> th_file_map;

	// Cifar data
	std::vector<std::pair<float, std::vector<float>>> cifar_data;

	std::vector<std::mutex> *mutices;

	std::vector<int> queue; // ready queue
	std::mutex q_mutex;
	std::condition_variable cv;
	bool run = true;

	std::vector<std::condition_variable> *load_cvs;
	std::vector<bool> ready_to_load;

	//// WARNING: MAGIC NUMBERS AHEAD
	float VGG_MEAN[3] = { 103.939f, 116.779f, 123.68f };  // ORDER: B_mean, G_mean, R_mean
	//float VGG_MEAN[3] = { 126.88f, 120.24f, 112.19f };  // ORDER: B_mean, G_mean, R_mean
	float CIFAR_MEAN[3] = { 125.307f, 122.95f, 113.865f }; // ORDER: RGB
	////

public:

	Image_preparator(std::string image_path_, std::string labels_path_, int thread_num_,
		int batchSize_, std::vector<float*> &batches_, std::vector<float*> &labels_, size_t byte_size_, size_t train_size_ = 0, size_t test_size_ = 0)
	{
		//image_path = image_path_;
		//labels_path = labels_path_;
		thread_num = thread_num_;
		batchSize = batchSize_;
		batches = batches_;
		labels = labels_;
		byte_size = byte_size_;
		train_size = train_size_;
		test_size = test_size_;
				
		mutices = new std::vector<std::mutex>(thread_num);
		ready_to_load.resize(thread_num, true);
		load_cvs = new std::vector<std::condition_variable>(thread_num);		
	}

	Image_preparator(int thread_num_);
	void set_train_param(std::string image_path_, std::string labels_path_, int batchSize_, size_t byte_size_, size_t train_size_ = 0);
	void set_train_param_test(std::string image_path_, std::string labels_path_, int batchSize_, size_t byte_size_, size_t train_size_ = 0);
	void set_validation_param(std::string image_path_, std::string labels_path_, size_t validation_size_ = 0);
		
	void set_dim_train_param(std::string image_path_, int batchSize_, size_t byte_size_, size_t train_size_ = 0);
	int set_dim_seq_param(std::string image_path_, int batchSize_, size_t byte_size_, size_t train_size_ = 0);
	void next_dim_epoch();
	void dim_eval();


	void start_validation();	
	void start_next_epoch();
	void map_and_rand();
	void load_next_batch(float *d_batch, float *d_label);
	bool load_next_batch_internal(float *d_batch, float *d_label);
	void load_next_val_pair(float *d_img, float *d_flipped, int &height, int &width, int & label);
	void stop();

	void validation_preparation();

	void cifar_preparation(std::string cifar_path_, int batchSize_, size_t byte_size_, bool train);
	void start_next_cifar_epoch();

	~Image_preparator();

	void test_func();
	void check_dim_image(float* res, float* label, int height, int width);
	void check_dim_and_stay(float *res, float *label, float *image, int height, int width);
	void save_dim_pic(const char* save_path, float *res, float* image, int height, int width, int num);

private:
	void val_img_prep_th(int num);
	void val_img_prep_th_2(int num);
	void train_img_prep_th(int num);
	void get_next_train_image(const char* filename, std::vector<float> &img);	
	void get_next_val_image(const char* filename, int &height, int &width, std::vector<float> &img, std::vector<float> &flipped);
	void next_cifar_train_img(int num, std::vector<float> &img);
	void next_cifar_test_img(int num, std::vector<float> &img);
	void cifar_prep_th();
	void cifar_prep_val_th();

	void train_dim_prep_th(int num);
	void train_dim_prep_th_no_white(int num);

	void eval_dim_th(int num);
	bool get_next_dim_train_image(const char* img_name, const char* gt_name, std::vector<float> &img, std::vector<float> &gt);
	void get_next_dim_eval_image(const char* img_name, const char* gt_name, std::vector<float> &img, std::vector<float> &gt);
	void prepare_dim_filenames();
	void rand_dim_files();
	int prepare_dim_seq();
	

	void prepare_filenames();
	
	
};