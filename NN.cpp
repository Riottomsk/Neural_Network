#include "NN.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <functional>
#include <array>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include "json.hpp"
#include <fstream>
#include <experimental\filesystem>

#include "direct_gpu_computation.cuh"

Neural_Network::Neural_Network(int gpuid, int batch_size, bool pretrained_, int update_type_, int loss_type_)
{
	m_gpuid = gpuid;
	m_batchSize = batch_size;
	pretrained = pretrained_;
	update_type = update_type_;
	loss_type = loss_type_;

	//checkCudaErrors(cudaSetDevice(gpuid));
	checkCudaErrors(cublasCreate(&cublasHandle));
	checkCUDNN(cudnnCreate(&cudnnHandle));

	// creating one_vec

	std::vector<float> one_v(batch_size, 1.0f);
	
	checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float) * batch_size));
	checkCudaErrors(cudaMemcpy(d_onevec, one_v.data(), sizeof(float) * batch_size, cudaMemcpyHostToDevice));
	
}

void Neural_Network::addInput(int channels, int height, int width, std::string name)
{
	if (!layers.empty())
		return; // only one input layer per nn
	Layer *l = new Input_layer(height, width, channels, this, name);
	layers.push_back(l);
}

void Neural_Network::addConv(int channels, int kernel_size, int pad, std::string name)
{
	if (layers.empty())
		return; // only input layer can be first
	Layer *l = new Conv_layer(channels, kernel_size, pad, name);	
	l->set_forward_propagation(layers.back(), this);
	layers.push_back(l);
}

void Neural_Network::addPool(int size,int stride, int pad, std::string name)
{
	if (layers.empty())
		return; // only input layer can be first
	Layer *l = new Pool_layer(size, stride, pad, name);
	l->set_forward_propagation(layers.back(), this);
	layers.push_back(l);
}

void Neural_Network::addReLU(std::string name)
{
	if (layers.empty())
		return; // only input layer can be first
	Layer *l = new RELU_layer(name);
	l->set_forward_propagation(layers.back(), this);
	layers.push_back(l);
}

void Neural_Network::addFC(int channels, std::string name)
{
	if (layers.empty())
		return; // only input layer can be first
	Layer *l = new FC_layer(channels, name);
	l->set_forward_propagation(layers.back(), this);
	layers.push_back(l);
}

void Neural_Network::addDropout(float dropout_ratio, std::string name)
{
	if (layers.empty())
		return; // only input layer can be first
	Layer *l = new Dropout_layer(dropout_ratio, name);
	l->set_forward_propagation(layers.back(), this);
	layers.push_back(l);
}

void Neural_Network::addSoftMax(std::string name)
{
	if (layers.empty())
		return; // only input layer can be first
	Layer *l = new SoftMax_layer(name);
	l->set_forward_propagation(layers.back(), this);
	layers.push_back(l);
}
void Neural_Network::addUnpool(std::string name)
{
	if (layers.empty())
		return; // only input layer can be first

	std::vector<int> pools;
	//int counter = 0;
	for (int i = 0; i < layers.size(); i++)
	{
		if (layers[i]->type == POOL)		
			pools.push_back(i);		
		if (layers[i]->type == UNPOOL)
			if (pools.size() > 0)
				pools.pop_back();
			else
				return; // there should be pool layers before unpool
	}

	Layer *l = new Unpool_layer((Pool_layer*)layers[pools.back()],layers[pools.back()-1],name);
	l->set_forward_propagation(layers.back(), this);
	layers.push_back(l);
}

// avoid using since Unified memory implemented
void show_some_data(float* d_output, int size)
{
	int bytes = sizeof(float) * size;
	float * h_output = new float[bytes];
	checkCudaErrors(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
	for (int i = 0; i < size; i++)
		printf("%f ", h_output[i]);
	std::cout << std::endl;
	delete[] h_output;
}

void debug_data(float *d_data, int size)
{
	checkCudaErrors(cudaDeviceSynchronize());
	for (int i = 0; i < size; ++i)
		printf("%f ", d_data[i]);
	printf("\n");
}

void write_point(size_t iter, double loss)
{
	std::ofstream out;
	out.open("tmp\\loss.csv", std::ios::app);
	std::string str = std::to_string(iter) + "," + std::to_string(loss) + "\n";
	out << str;
}

void Neural_Network::train(float * data, float * label, size_t iter, bool visualize, bool backup)
{
	//std::cout << "Train iter inside: " << iter << std::endl;
	if (layers.empty())
		return;
	if (update_type == ONLY_INFERENCE)
		return; // training can't be performed since nn is only for inference
	if (layers[0]->type == INPUT && layers.size() > 1)
	{		
		//std::cout << iter << std::endl;
		auto t1 = std::chrono::high_resolution_clock::now();
		forwardPropagation(data);
		auto t2 = std::chrono::high_resolution_clock::now();
		//if (visualize)
		//	printf("Forward time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);
		t1 = std::chrono::high_resolution_clock::now();

		//float * tmp;
		//tmp = new float[m_batchSize];
		//checkCudaErrors(cudaMemcpyAsync(tmp, label, sizeof(float) * m_batchSize, cudaMemcpyDeviceToHost));
		//std::cout << "Label in train: " << tmp[0] << std::endl;

		calculate_loss(label);
		t2 = std::chrono::high_resolution_clock::now();
		//if (visualize)
		//	printf("Calculation time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);
		t1 = std::chrono::high_resolution_clock::now();
		if (visualize)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			//show_some_data(layers[layers.size() - 2]->data, 10);	
			//show_some_data(layers.back()->data, 16);
			//show_some_data(label, 16);
			float *tmp_loss = layers.back()->diff_data;
			int byte_size = sizeof(float) * m_batchSize * layers.back()->out_channels * layers.back()->out_height * layers.back()->out_width;
			float * h_output = new float[byte_size];
			checkCudaErrors(cudaMemcpyAsync(h_output, tmp_loss, byte_size, cudaMemcpyDeviceToHost));
			int out_size = byte_size / sizeof(float);
			double loss = 0;
			if (loss_type == CROSS_ENTROPY)
			{
				//for (int i = 0; i < out_size; i++)
				//	loss += std::pow(h_output[i], 2);
				//loss = loss / (m_batchSize * out_size);
				//loss = std::sqrt(loss);

				for (int i = 0; i < out_size; ++i)
				{
					if (h_output[i] < 0)
						loss += -std::log(h_output[i] + 1); // Now this is true Cross-Entopy loss!
				}
			}
			int unk_reg = 0;
			if (loss_type == COMPOSITIONAL)
			{
				float * h_data;
				float * h_label;
				float * h_original;
				h_data = new float[out_size];
				h_label = new float[out_size];
				h_original = new float[out_size * 4];
				checkCudaErrors(cudaMemcpyAsync(h_data, layers.back()->data, byte_size, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpyAsync(h_label, label, byte_size, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpyAsync(h_original, layers[0]->data, byte_size * 4, cudaMemcpyDeviceToHost));

				int size = out_size / m_batchSize;
				for (int i = 0; i < out_size; ++i)
				{

					int batch_num = int(i / size);
					int pix_num = i % size;
					if (h_original[batch_num*size * 4 + pix_num * 4] == 0.5)
					{
						unk_reg++;
					}
						loss += std::abs(h_data[i] - h_label[i]);
					
				}
				delete[] h_data;
				delete[] h_label;
				delete[] h_original;
			}
			delete[] h_output;
			//show_some_data(tmp_loss, 10);
			
			//checkCudaErrors(cudaGetLastError());
			//checkCudaErrors(cudaDeviceSynchronize());
			//double loss = 0;
			//int out_size = m_batchSize * layers.back()->out_channels * layers.back()->out_height * layers.back()->out_width;
			//float *d_loss = layers[layers.size() - 1]->diff_data;
			//for (int i = 0; i < out_size; ++i)
			//	loss += std::pow(d_loss[i], 2);

			std::cout << "Res: " << iter <<": "<< loss << " avr loss: " << loss / m_batchSize << " unk_reg: "<< unk_reg << std::endl;
			write_point(iter,loss);
			//if ((iter) % 5000 == 0 || 1)
				//show_weights();
		}
		t2 = std::chrono::high_resolution_clock::now();
		//if (visualize)
		//	printf("Visualization time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);

		it++;

		if (backup)
		{
			save_weights("tmp");
		}
		if (update_type == SGD)
		{
			learning_rate = static_cast<float>(0.0001 * pow((1.0 + 0.0001 * iter), (-0.75)));		
			//learning_rate = 0.01f;
		}

		if (update_type == mSGD)
		{
			msgd.t++;
		}

		if (update_type == ADAM)
		{
			adam.t++;
			adam.alpha_t = -adam.alpha * std::sqrt(1 - std::pow(adam.beta2, adam.t)) / (1 - std::pow(adam.beta1, adam.t));
			adam.epsilon_t = adam.epsilon/* * std::sqrt(1 - std::pow(adam.beta2, adam.t))*/; // this line is not in Adam paper but it should be like this		
		}
		t1 = std::chrono::high_resolution_clock::now();
		backPropagation();		
		t2 = std::chrono::high_resolution_clock::now();
		//if (visualize)
		//	printf("Back time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f);
		//std::cout << "Train iter over\n";
	}
}

void Neural_Network::save_weights(const char * nn_name)
{
	namespace fs = std::experimental::filesystem;

	fs::path path = nn_name;

	if (!fs::exists(path))
		fs::create_directory(path);

	if (update_type == ADAM)
	{
		std::stringstream ss;
		ss << nn_name << "\\iter.bin"; // iter number

		FILE *f = fopen(ss.str().c_str(), "wb");
		if (!f)
		{
			printf("ERROR: Cannot open file %s\n", ss.str().c_str());
			exit(2);
		}

		fwrite(&adam.t, sizeof(size_t), 1, f);
		fclose(f);
	}
	
	{
		std::stringstream ss;
		ss << nn_name << "\\iter.bin"; // iter number

		FILE *f = fopen(ss.str().c_str(), "wb");
		if (!f)
		{
			printf("ERROR: Cannot open file %s\n", ss.str().c_str());
			exit(2);
		}

		fwrite(&it, sizeof(size_t), 1, f);
		fclose(f);
	}
	
	for (int i = 0; i < layers.size(); ++i)
	{
		if (layers[i]->type == CONV)
		{
			//((Conv_layer*)layers[i])->save_weights(nn_name);
			std::stringstream ssf;
			if (layers[i]->name != "")
				ssf << nn_name << "\\" << layers[i]->name << "_W.bin";
			else
				ssf << nn_name << "\\conv" << i << "_W.bin";

			// Write weights file
			FILE *fp = fopen(ssf.str().c_str(), "wb");
			if (!fp)
			{
				printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
				exit(2);
			}
			checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->kernel_value[0],
				((Conv_layer*)layers[i])->filter_data,
				sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(), cudaMemcpyDeviceToHost));
			fwrite(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
				((Conv_layer*)layers[i])->kernel_value.size(), fp);

			fclose(fp);

			// Write bias file

			std::stringstream ssb;
			if (layers[i]->name != "")
				ssb << nn_name << "\\" << layers[i]->name << "_b.bin";
			else
				ssb << nn_name << "\\conv" << i << "_b.bin";
			
			fp = fopen(ssb.str().c_str(), "wb");
			if (!fp)
			{
				printf("ERROR: Cannot open file %s\n", ssb.str().c_str());
				exit(2);
			}
			checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->bias_value[0],
				((Conv_layer*)layers[i])->d_bias,
				sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(), cudaMemcpyDeviceToHost));
			fwrite(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
				((Conv_layer*)layers[i])->bias_value.size(), fp);

			fclose(fp);

			// Write update files
			// Some update types requires previous values
			// So it might be helpful with continious learning

			if (update_type == ADAM)
			{
				std::stringstream ssu;
				if (layers[i]->name != "")
					ssu << nn_name << "\\" << layers[i]->name << "_adam_m_W.bin";
				else
					ssu << nn_name << "\\conv" << i << "_adam_m_W.bin";

				fp = fopen(ssu.str().c_str(), "wb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssu.str().c_str());
					exit(2);
				}

				checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->kernel_value[0],
					((Conv_layer*)layers[i])->adam.d_m_f,
					sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(), cudaMemcpyDeviceToHost));

				fwrite(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
					((Conv_layer*)layers[i])->kernel_value.size(), fp);
				
				fclose(fp);
				
				ssu.str("");

				if (layers[i]->name != "")
					ssu << nn_name << "\\" << layers[i]->name << "_adam_v_W.bin";
				else
					ssu << nn_name << "\\conv" << i << "_adam_v_W.bin";

				fp = fopen(ssu.str().c_str(), "wb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssu.str().c_str());
					exit(2);
				}

				checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->kernel_value[0],
					((Conv_layer*)layers[i])->adam.d_v_f,
					sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(), cudaMemcpyDeviceToHost));

				fwrite(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
					((Conv_layer*)layers[i])->kernel_value.size(), fp);

				fclose(fp);

				ssu.str("");

				if (layers[i]->name != "")
					ssu << nn_name << "\\" << layers[i]->name << "_adam_m_b.bin";
				else
					ssu << nn_name << "\\conv" << i << "_adam_m_b.bin";

				fp = fopen(ssu.str().c_str(), "wb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssu.str().c_str());
					exit(2);
				}

				checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->bias_value[0],
					((Conv_layer*)layers[i])->adam.d_m_b,
					sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(), cudaMemcpyDeviceToHost));
				fwrite(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
					((Conv_layer*)layers[i])->bias_value.size(), fp);

				fclose(fp);

				ssu.str("");

				if (layers[i]->name != "")
					ssu << nn_name << "\\" << layers[i]->name << "_adam_v_b.bin";
				else
					ssu << nn_name << "\\conv" << i << "_adam_v_b.bin";

				fp = fopen(ssu.str().c_str(), "wb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssu.str().c_str());
					exit(2);
				}

				checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->bias_value[0],
					((Conv_layer*)layers[i])->adam.d_v_b,
					sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(), cudaMemcpyDeviceToHost));
				fwrite(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
					((Conv_layer*)layers[i])->bias_value.size(), fp);

				fclose(fp);
			}

			if (update_type == mSGD)
			{
				std::stringstream ssu;
				if (layers[i]->name != "")
					ssu << nn_name << "\\" << layers[i]->name << "_msgd_v_W.bin";
				else
					ssu << nn_name << "\\conv" << i << "_msgd_v_W.bin";

				fp = fopen(ssu.str().c_str(), "wb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssu.str().c_str());
					exit(2);
				}

				checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->kernel_value[0],
					((Conv_layer*)layers[i])->msgd.d_v_f,
					sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(), cudaMemcpyDeviceToHost));

				fwrite(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
					((Conv_layer*)layers[i])->kernel_value.size(), fp);

				fclose(fp);

				ssu.str("");

				if (layers[i]->name != "")
					ssu << nn_name << "\\" << layers[i]->name << "_msgd_v_b.bin";
				else
					ssu << nn_name << "\\conv" << i << "_msgd_v_b.bin";

				fp = fopen(ssu.str().c_str(), "wb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssu.str().c_str());
					exit(2);
				}

				checkCudaErrors(cudaMemcpy(&((Conv_layer*)layers[i])->bias_value[0],
					((Conv_layer*)layers[i])->msgd.d_v_b,
					sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(), cudaMemcpyDeviceToHost));
				fwrite(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
					((Conv_layer*)layers[i])->bias_value.size(), fp);

				fclose(fp);
			}

		}
	}
}

void Neural_Network::load_weights(const char * nn_name)
{
	namespace fs = std::experimental::filesystem;
	fs::path path = nn_name;
	if (!fs::exists(path))
	{
		std::cerr << "Directory doesn't exists!" << std::endl;
		system("pause");
		exit(2);
	}

	if (update_type == ADAM)
	{
		std::stringstream ss;
		ss << nn_name << "\\iter.bin"; // iter number

		FILE *f = fopen(ss.str().c_str(), "rb");
		if (!f)
		{
			printf("ERROR: Cannot open file %s\n", ss.str().c_str());
			exit(2);
		}

		fread(&adam.t, sizeof(size_t), 1, f);
		fclose(f);
	}
	
	if (update_type != ONLY_INFERENCE)
	{
		std::stringstream ss;
		ss << nn_name << "\\iter.bin"; // iter number

		FILE *f = fopen(ss.str().c_str(), "rb");
		if (!f)
		{
			printf("ERROR: Cannot open file %s\n", ss.str().c_str());
			exit(2);
		}

		fread(&it, sizeof(size_t), 1, f);
		fclose(f);
	}

	for (int i = 0; i < layers.size(); ++i)
	{
		if (layers[i]->type == CONV)
		{
			std::stringstream ss;
			if (layers[i]->name != "")
				ss << nn_name << "\\" << layers[i]->name << "_W.bin";
			else
				ss << nn_name << "\\conv" << i << "_W.bin";

			// Write weights file
			FILE *fp = fopen(ss.str().c_str(), "rb");
			if (!fp)
			{
				printf("ERROR: Cannot open file %s\n", ss.str().c_str());
				system("pause");
				exit(2);
			}
			fread(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
				((Conv_layer*)layers[i])->kernel_value.size(), fp);
			fclose(fp);

			checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->filter_data,
				((Conv_layer*)layers[i])->kernel_value.data(),
				sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(),
				cudaMemcpyHostToDevice));

			ss.str("");

			if (layers[i]->name != "")
				ss << nn_name << "\\" << layers[i]->name << "_b.bin";
			else
				ss << nn_name << "\\conv" << i << "_b.bin";


			fp = fopen(ss.str().c_str(), "rb");
			if (!fp)
			{
				printf("ERROR: Cannot open file %s\n", ss.str().c_str());
				system("pause");
				exit(2);
			}
			fread(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
				((Conv_layer*)layers[i])->bias_value.size(), fp);
			fclose(fp);

			checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->d_bias,
				((Conv_layer*)layers[i])->bias_value.data(),
				sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(),
				cudaMemcpyHostToDevice));

			ss.str("");

			if (update_type == ADAM)
			{				
				if (layers[i]->name != "")
					ss << nn_name << "\\" << layers[i]->name << "_adam_m_W.bin";
				else
					ss << nn_name << "\\conv" << i << "_adam_m_W.bin";
								
				fp = fopen(ss.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ss.str().c_str());
					system("pause");
					exit(2);
				}
				fread(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
					((Conv_layer*)layers[i])->kernel_value.size(), fp);
				fclose(fp);

				checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->adam.d_m_f,
					((Conv_layer*)layers[i])->kernel_value.data(),
					sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(),
					cudaMemcpyHostToDevice));

				ss.str("");

				if (layers[i]->name != "")
					ss << nn_name << "\\" << layers[i]->name << "_adam_v_W.bin";
				else
					ss << nn_name << "\\conv" << i << "_adam_v_W.bin";
								
				fp = fopen(ss.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ss.str().c_str());
					system("pause");
					exit(2);
				}
				fread(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
					((Conv_layer*)layers[i])->kernel_value.size(), fp);
				fclose(fp);

				checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->adam.d_v_f,
					((Conv_layer*)layers[i])->kernel_value.data(),
					sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(),
					cudaMemcpyHostToDevice));

				ss.str("");

				if (layers[i]->name != "")
					ss << nn_name << "\\" << layers[i]->name << "_adam_m_b.bin";
				else
					ss << nn_name << "\\conv" << i << "_adam_m_b.bin";

				fp = fopen(ss.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ss.str().c_str());
					system("pause");
					exit(2);
				}
				fread(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
					((Conv_layer*)layers[i])->bias_value.size(), fp);
				fclose(fp);

				checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->adam.d_m_b,
					((Conv_layer*)layers[i])->bias_value.data(),
					sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(),
					cudaMemcpyHostToDevice));

				ss.str("");

				if (layers[i]->name != "")
					ss << nn_name << "\\" << layers[i]->name << "_adam_v_b.bin";
				else
					ss << nn_name << "\\conv" << i << "_adam_v_b.bin";

				fp = fopen(ss.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ss.str().c_str());
					system("pause");
					exit(2);
				}
				fread(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
					((Conv_layer*)layers[i])->bias_value.size(), fp);
				fclose(fp);

				checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->adam.d_v_b,
					((Conv_layer*)layers[i])->bias_value.data(),
					sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(),
					cudaMemcpyHostToDevice));
			}

			//((Conv_layer*)layers[i])->load_weights();			
		}
	}
}

void Neural_Network::load_only_weights(const char * nn_name)
{
	namespace fs = std::experimental::filesystem;
	fs::path path = nn_name;
	if (!fs::exists(path))
	{
		std::cerr << "Directory doesn't exists!" << std::endl;
		system("pause");
		exit(2);
	}

	for (int i = 0; i < layers.size(); ++i)
	{
		if (layers[i]->type == CONV)
		{
			std::stringstream ss;
			if (layers[i]->name != "")
				ss << nn_name << "\\" << layers[i]->name << "_W.bin";
			else
				ss << nn_name << "\\conv" << i << "_W.bin";


			FILE *fp = fopen(ss.str().c_str(), "rb");
			if (!fp)
			{
				printf("ERROR: Cannot open file %s\n", ss.str().c_str());
				system("pause");
				exit(2);
			}
			fread(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
				((Conv_layer*)layers[i])->kernel_value.size(), fp);
			fclose(fp);

			checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->filter_data,
				((Conv_layer*)layers[i])->kernel_value.data(),
				sizeof(float) * ((Conv_layer*)layers[i])->kernel_value.size(),
				cudaMemcpyHostToDevice));

			ss.str("");

			if (layers[i]->name != "")
				ss << nn_name << "\\" << layers[i]->name << "_b.bin";
			else
				ss << nn_name << "\\conv" << i << "_b.bin";


			fp = fopen(ss.str().c_str(), "rb");
			if (!fp)
			{
				printf("ERROR: Cannot open file %s\n", ss.str().c_str());
				system("pause");
				exit(2);
			}
			fread(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
				((Conv_layer*)layers[i])->bias_value.size(), fp);
			fclose(fp);

			checkCudaErrors(cudaMemcpyAsync(((Conv_layer*)layers[i])->d_bias,
				((Conv_layer*)layers[i])->bias_value.data(),
				sizeof(float) * ((Conv_layer*)layers[i])->bias_value.size(),
				cudaMemcpyHostToDevice));
		}
	}
}

void Neural_Network::forwardPropagation(float * data)
{
	if (layers.empty())
		return;
	if (layers[0]->type == INPUT && layers.size() > 1)
	{
		((Input_layer*)layers[0])->input_data(data, this);
		//show_some_data(layers[0]->data, 10);
		//show_some_data(((Conv_layer*)layers[1])->filter_data, 10);
		for (int i = 1; i < layers.size(); i++)
			layers[i]->forward_propagation(layers[i - 1], this);
	}
}

void Neural_Network::setWorkSpace()
{
	gpu_memory_allocated += m_workspaceSize;
	if (DEBUG_LEVEL > 0)
	{
		std::cout << "NN Workspace memory: " << m_workspaceSize / 1048576.0 << "MB" << std::endl;
		std::cout << "Total memory allocated: " << gpu_memory_allocated / 1048576.0 << "MB" << std::endl;
	}
	//checkCudaErrors(cudaMallocManaged(&cudnn_workspace, m_workspaceSize));
	checkCudaErrors(cudaMalloc(&cudnn_workspace, m_workspaceSize));
}

void Neural_Network::backPropagation()
{
	if (layers.empty())
		return;
	if (layers[0]->type == INPUT && layers.size() > 1)
	{
		for (int i = layers.size() - 1; i > 0; i--)
		{
			//show_some_data(layers[i]->diff_data, layers[i]->out_channels*layers[i]->out_height*layers[i]->out_width);
			layers[i]->back_propagation(layers[i - 1], this);
		}
	}
}


void sum_loss(float* d_output, int size)
{
	int bytes = sizeof(float) * size;
	float * h_output = new float[bytes];
	checkCudaErrors(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
	double loss = 0;
	for (int i = 0; i < size; i++)
		loss += std::pow(h_output[i], 2);
	loss = loss / (2 * size);
	std::cout << "Error: "<<loss<< std::endl;
}

double Neural_Network::calculate_loss(float * d_label)
{
	float *d_diff = layers[layers.size() - 1]->diff_data;
	
	int byte_size = sizeof(float) * m_batchSize * layers.back()->out_channels * layers.back()->out_height * layers.back()->out_width;
	
	
	//float *d_res = layers.back()->data;	
	//debug_data(d_diff, 10);
	int num_classes = layers.back()->out_channels * layers.back()->out_height * layers.back()->out_width; 
	if (loss_type == CROSS_ENTROPY)
	{
		checkCudaErrors(cudaMemcpyAsync(d_diff, layers.back()->data, byte_size, cudaMemcpyDeviceToDevice));
		calculate_loss_with_gpu(d_label, num_classes, m_batchSize, d_diff);
	}

	if (loss_type == COMPOSITIONAL)
	{
		float *d_data = layers.back()->data;
		int size = layers.back()->out_channels * layers.back()->out_height * layers.back()->out_width * m_batchSize;
		comp_loss_with_gpu(d_data, d_label, size, d_diff);
		//int size = layers.back()->out_channels * layers.back()->out_height * layers.back()->out_width;
		//comp_loss_from_unknown_area(layers[0]->data, d_data, d_label, m_batchSize, size, d_diff);
	}

	//debug_data(d_diff, 10);
	//calculate_class_loss_with_gpu(d_label, num_classes, m_batchSize, d_res, d_diff);
	return 0;
}

void Neural_Network::show_weights()
{
	for (auto &l : layers)
		if (l->type == CONV)
			static_cast<Conv_layer*>(l)->show_weights();
}

//void Neural_Network::load_weights()
//{
//	if (layers.back()->type == CONV)
//	{
//		// Mystery kernel
//		const float kernel_template[3][3] = {
//			{ 1,  1, 1 },
//			{ 1, -8, 1 },
//			{ 1,  1, 1 }
//		};
//		float h_kernel[3][3][3][3];
//		std::vector<float> vec_kernel;// (9, 1);
//		for (int kernel = 0; kernel < 3; ++kernel) {
//			for (int channel = 0; channel < 3; ++channel) {
//				for (int row = 0; row < 3; ++row) {
//					for (int column = 0; column < 3; ++column) 
//					{
//						h_kernel[kernel][channel][row][column] = kernel_template[row][column];
//						vec_kernel.push_back(kernel_template[row][column]);
//					}
//				}
//			}
//		}
//
//		((Conv_layer*)layers.back())->load_weights(vec_kernel);
//	}
//}

float * Neural_Network::getResult()
{
	return layers.back()->data;
}

void Neural_Network::clearTensors()
{
	for (auto l : layers)
		l->clear();	
}

void Neural_Network::redefineTensors(int height, int width, int channels)
{
	if (layers.empty())
		return;
	if (layers[0]->type != INPUT)
		return;
	if (update_type != ONLY_INFERENCE) // no idea why you might want to use this for training
		return;
	//std::cout << width << " " << height << std::endl;
	((Input_layer*)layers[0])->redefine(height, width, channels, this); // explicitly
	for (int i = 1; i < layers.size(); i++)
	{
		if (layers[i]->type == DROPOUT)
		{
			layers[i + 1]->redefine(layers[i - 1], this);
			i++;
		}
		else
			layers[i]->redefine(layers[i - 1], this);
	}
}

void Neural_Network::getOutputDimensions(int & height, int & width, int & classes)
{
	height = layers.back()->out_height;
	width = layers.back()->out_width;
	classes = layers.back()->out_channels;
}

std::vector<float> Neural_Network::interpret_softmax_result(const char* path, bool show)
{
	assert(layers.back()->type != SOFTMAX || m_batchSize != 1);
	checkCudaErrors(cudaDeviceSynchronize());
	auto data = getResult();
	int channels = layers.back()->out_channels;
	int height = layers.back()->out_height;
	int width = layers.back()->out_width;
	int size = channels*height*width;
	int bytes = sizeof(float) * size;
	float * h_output = new float[bytes];
	
	checkCudaErrors(cudaMemcpy(h_output, data, bytes, cudaMemcpyDeviceToHost));

	std::vector<float> res(channels);
	for (int c = 0; c < channels; ++c)
		for (int h = 0; h < height; ++h)
			for (int w = 0; w < width; ++w)
				res[c] += h_output[c*height*width + h*width + w] / (height*width);
	
	std::vector<std::pair<float, int>> r;
	for (int i = 0; i < channels; i++)
		r.push_back(std::make_pair(res[i], i));


	std::sort(r.begin(), r.end(), [](std::pair<float, int> a, std::pair<float, int> b) {
		return a > b;
	});
	if (show)
	{
		if (path != nullptr)
		{
			// read a JSON file
			std::ifstream in(path);
			using json = nlohmann::json;
			json j;
			in >> j;
			std::vector<std::string> object_names(channels);
			for (json::iterator it = j.begin(); it != j.end(); ++it)
				object_names[atoi(it.key().c_str())] = it.value()[1].dump();

			for (int i = 0; i < 5; i++)
				std::cout << r[i].first * 100 << " " << r[i].second << " " << object_names[r[i].second] << std::endl;
		}
		else
			for (int i = 0; i < 5; i++)
				std::cout << r[i].first * 100 << " " << r[i].second << std::endl;
	}
	delete[] h_output;
	
	//show_some_data(layers.back()->data, 10);
	//show_some_data(layers[layers.size() - 3]->data, 10);
	return res;
}

void Neural_Network::inference(float * d_data)
{
	if (layers.empty())
		return;
	if (layers[0]->type == INPUT && layers.size() > 1)
	{
		((Input_layer*)layers[0])->input_data(d_data, this);
		for (int i = 1; i < layers.size(); i++)
			if (layers[i]->type!=DROPOUT)
				layers[i]->forward_propagation(layers[i - 1], this);
			else
			{
				assert(layers[i + 1]->type != DROPOUT);
				layers[i + 1]->forward_propagation(layers[i - 1], this);
				i++;
			}
		//show_some_data(layers[layers.size() - 2]->data, 10);
	}
}

void Neural_Network::convert_VGG16_weights()
{
	const char* nn_name = "VGG16_pretrained";
	for (int i = 0; i < layers.size(); ++i)
	{
		if (layers[i]->type == CONV)
		{
			{
				((Conv_layer*)layers[i])->convert_and_load();
				continue;
				std::stringstream ssf;
				ssf << nn_name << "\\conv" << i << "_W.bin";

				// load weights file
				FILE *fp = fopen(ssf.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
					exit(2);
				}
				fread(&((Conv_layer*)layers[i])->kernel_value[0], sizeof(float),
					((Conv_layer*)layers[i])->kernel_value.size(), fp);
				fclose(fp);

				std::stringstream ssfb;
				ssfb << nn_name << "\\conv" << i << "_b.bin";

				// load bias file
				fp = fopen(ssfb.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssfb.str().c_str());
					exit(2);
				}
				fread(&((Conv_layer*)layers[i])->bias_value[0], sizeof(float),
					((Conv_layer*)layers[i])->bias_value.size(), fp);
				fclose(fp);
			}

			((Conv_layer*)layers[i])->convert_and_load();
		}
		if (layers[i]->type == FC)
		{
			{
				std::stringstream ssf;
				ssf << nn_name << "\\fc" << i << "_f.bin";

				// load weights file
				FILE *fp = fopen(ssf.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
					exit(2);
				}
				fread(&((FC_layer*)layers[i])->neurons[0], sizeof(float),
					((FC_layer*)layers[i])->neurons.size(), fp);
				fclose(fp);

				std::stringstream ssfb;
				ssfb << nn_name << "\\fc" << i << "_b.bin";

				// load bias file
				fp = fopen(ssfb.str().c_str(), "rb");
				if (!fp)
				{
					printf("ERROR: Cannot open file %s\n", ssfb.str().c_str());
					exit(2);
				}
				fread(&((FC_layer*)layers[i])->bias[0], sizeof(float),
					((FC_layer*)layers[i])->bias.size(), fp);
				fclose(fp);
			}
		
			((FC_layer*)layers[i])->convert_and_load();
		}
	}
}

void Neural_Network::load_specified_weights(float * data, size_t size, int layer_num)
{
	((Conv_layer*)layers[layer_num])->load_specified_weights(data, size);
}

Neural_Network::~Neural_Network()
{
	
	for (auto l : layers)
		delete l;
	checkCUDNN(cudnnDestroy(cudnnHandle));
	checkCudaErrors(cublasDestroy(cublasHandle));
	checkCudaErrors(cudaFree(cudnn_workspace));
}

Neural_Network::Layer::Layer(std::string name_)
{
	name = name_;
	checkCUDNN(cudnnCreateTensorDescriptor(&tensor));
}

void Neural_Network::Layer::define_tensor(Neural_Network *nn)
{
	//std::cout << out_height << " " << out_width << std::endl;
	//auto t1 = std::chrono::high_resolution_clock::now();
	checkCUDNN(cudnnSetTensor4dDescriptor(tensor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/nn->m_batchSize,
		/*channels=*/out_channels,
		/*image_height=*/out_height,
		/*image_width=*/out_width));
	size_t memory_needed = sizeof(float) * nn->m_batchSize * out_channels * out_height * out_width;

	if (type == INPUT)
	{
		checkCudaErrors(cudaMalloc(&data, memory_needed));
		nn->gpu_memory_allocated += memory_needed;
	}
	else
	{
		//checkCudaErrors(cudaMallocManaged(&data, memory_needed));
		checkCudaErrors(cudaMalloc(&data, memory_needed));
		//checkCudaErrors(cudaMemset(data, 0, sizeof(float) * nn->m_batchSize * out_channels * out_height * out_width));
		if (nn->update_type != ONLY_INFERENCE)
		{
			//checkCudaErrors(cudaMallocManaged(&diff_data, memory_needed));
			checkCudaErrors(cudaMalloc(&diff_data, memory_needed));
			memory_needed *= 2;
		}		
		nn->gpu_memory_allocated += memory_needed;
	}
	if (nn->DEBUG_LEVEL > 1)
		std::cout << "Memory allocated: " << memory_needed / 1048576.0 << "MB" << std::endl;
}

void Neural_Network::Layer::redefine_tensor(Neural_Network * nn)
{
	//std::cout << type << ": " << out_channels << " " << out_width << " " << out_height << std::endl;
	checkCUDNN(cudnnSetTensor4dDescriptor(tensor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/nn->m_batchSize,
		/*channels=*/out_channels,
		/*image_height=*/out_height,
		/*image_width=*/out_width));	
	//checkCudaErrors(cudaMallocManaged(&data, sizeof(float) * nn->m_batchSize * out_channels * out_height * out_width));
	checkCudaErrors(cudaMalloc(&data, sizeof(float) * nn->m_batchSize * out_channels * out_height * out_width));
}

void Neural_Network::Layer::link_tensors(Layer *src, Neural_Network * nn)
{
	tensor = src->tensor;
	data = src->data;
	diff_data = src->diff_data;
}

void Neural_Network::Layer::clear()
{	
	if (type != RELU)
	{
		if (data != nullptr)
		{
			checkCudaErrors(cudaFree(data));
			data = nullptr;			
		}
		if (diff_data != nullptr)
		{
			checkCudaErrors(cudaFree(diff_data));
			diff_data = nullptr;			
		}		
	}	
}

Neural_Network::Layer::~Layer()
{
	clear();
	if (type!= RELU)
		checkCUDNN(cudnnDestroyTensorDescriptor(tensor));
}



Neural_Network::Pool_layer::Pool_layer(int size_, int stride_, int pad_, std::string name) : Layer(name)
{
	size = size_;
	stride = stride_;

	type = POOL;
	checkCUDNN(cudnnCreatePoolingDescriptor(&desc));
	checkCUDNN(cudnnSetPooling2dDescriptor(desc,
		CUDNN_POOLING_MAX,
		CUDNN_PROPAGATE_NAN,
		size, size,
		pad_, pad_,
		stride, stride));
}

void Neural_Network::Pool_layer::set_forward_propagation(Layer * src, Neural_Network * nn)
{	
	out_channels = src->out_channels;
	int batchSize, channels;
	cudnnGetPooling2dForwardOutputDim(desc, src->tensor, &batchSize, &channels, &out_height, &out_width);
	//std::cout << out_height << " " << out_width << std::endl;
	assert(batchSize == nn->m_batchSize);
	assert(channels == out_channels);

	define_tensor(nn);
}

void Neural_Network::Pool_layer::forward_propagation(Layer *src, Neural_Network *nn)
{
	float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnPoolingForward(nn->cudnnHandle, desc, &alpha, src->tensor,
		src->data, &beta, tensor, data));
}

void Neural_Network::Pool_layer::back_propagation(Layer * dst, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnPoolingBackward(nn->cudnnHandle, desc, &alpha,
		tensor, data, tensor, diff_data,
		dst->tensor, dst->data, &beta, dst->tensor, dst->diff_data));

}

void Neural_Network::Pool_layer::redefine(Layer * src, Neural_Network * nn)
{
	int batchSize, channels;
	cudnnGetPooling2dForwardOutputDim(desc, src->tensor, &batchSize, &channels, &out_height, &out_width);
	assert(batchSize == nn->m_batchSize);
	assert(channels == out_channels);

	redefine_tensor(nn);
}

Neural_Network::Pool_layer::~Pool_layer()
{
	checkCUDNN(cudnnDestroyPoolingDescriptor(desc));
}


Neural_Network::Conv_layer::Conv_layer(int out_channels_, int kernel_size_, int pad_, std::string name) : Layer(name)
{
	out_channels = out_channels_;
	kernel_size = kernel_size_;
	type = CONV;
	pad = pad_;
}

void Neural_Network::Conv_layer::set_forward_propagation(Layer * src, Neural_Network * nn) // rename - cos not only forward prop now
{	
	in_channels = src->out_channels;
	in_height = src->out_height;
	in_width = src->out_width;

	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/out_channels,
		/*in_channels=*/in_channels,
		/*kernel_height=*/kernel_size,
		/*kernel_width=*/kernel_size)); // most of convolution requires square kernels - change if needed
	kernel_value.resize(in_channels * kernel_size * kernel_size * out_channels,0);
	
	size_t filter_memory = sizeof(float) * kernel_value.size();
	//checkCudaErrors(cudaMallocManaged(&filter_data, sizeof(float) * kernel_value.size()));
	checkCudaErrors(cudaMalloc(&filter_data, sizeof(float) * kernel_value.size()));
	if (nn->update_type != ONLY_INFERENCE)
	{
		//checkCudaErrors(cudaMallocManaged(&gfdata, sizeof(float) * kernel_value.size()));
		checkCudaErrors(cudaMalloc(&gfdata, sizeof(float) * kernel_value.size()));
		filter_memory *= 2;
	}
	if (nn->DEBUG_LEVEL > 2)
		std::cout << "Conv memory allocated, filters: " << filter_memory / 1048576.0 << "MB" << std::endl;
	nn->gpu_memory_allocated += filter_memory;

	size_t update_data = 0;
	if (nn->update_type == ADAM)
	{		
		// Adam parameters
		//checkCudaErrors(cudaMallocManaged(&adam.d_m_f, sizeof(float) * kernel_value.size()));
		checkCudaErrors(cudaMalloc(&adam.d_m_f, sizeof(float) * kernel_value.size()));
		checkCudaErrors(cudaMemset(adam.d_m_f, 0, sizeof(float) * kernel_value.size()));

		//checkCudaErrors(cudaMallocManaged(&adam.d_v_f, sizeof(float) * kernel_value.size()));
		checkCudaErrors(cudaMalloc(&adam.d_v_f, sizeof(float) * kernel_value.size()));
		checkCudaErrors(cudaMemset(adam.d_v_f, 0, sizeof(float) * kernel_value.size()));
		//for (size_t i = 0; i < kernel_value.size(); ++i)
		//{
		//	adam.d_m_f[i] = 0;
		//	adam.d_v_f[i] = 0;
		//}
		//checkCudaErrors(cudaMallocManaged(&adam.tmp_f, sizeof(float) * kernel_value.size()));
		checkCudaErrors(cudaMalloc(&adam.tmp_f, sizeof(float) * kernel_value.size()));
		update_data = sizeof(float) * kernel_value.size() * 3;
		//
	}

	if (nn->update_type == mSGD)
	{
		//checkCudaErrors(cudaMallocManaged(&msgd.d_v_f, sizeof(float) * kernel_value.size()));
		//checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMalloc(&msgd.d_v_f, sizeof(float) * kernel_value.size()));
		
		//checkCudaErrors(cudaGetLastError());
		//std::cout << kernel_value.size() << std::endl;
		//for (size_t i = 0; i < kernel_value.size(); ++i)
		//{
		//	msgd.d_v_f[i] = 0;
		//}
		//checkCudaErrors(cudaMemset(msgd.d_v_f, 0, sizeof(float) * kernel_value.size()));
		update_data = sizeof(float) * kernel_value.size();
	}

	//// bias
	checkCUDNN(cudnnCreateTensorDescriptor(&bias_tensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(bias_tensor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1, out_channels,
		1, 1));
	bias_value.resize(out_channels);
	size_t bias_memory = sizeof(float) * out_channels;
	//checkCudaErrors(cudaMallocManaged(&d_bias, sizeof(float) * out_channels));
	checkCudaErrors(cudaMalloc(&d_bias, sizeof(float) * out_channels));
	if (nn->update_type != ONLY_INFERENCE)
	{
		//checkCudaErrors(cudaMallocManaged(&d_gbias, sizeof(float) * out_channels));		
		checkCudaErrors(cudaMalloc(&d_gbias, sizeof(float) * out_channels));
		bias_memory *= 2;
	}
	if (nn->DEBUG_LEVEL > 2)
		std::cout << "Conv memory allocated, bias: " << bias_memory / 1048576.0 << "MB" << std::endl;
	nn->gpu_memory_allocated += bias_memory;

	if (nn->update_type == ADAM)
	{
		// Adam parameters
		//checkCudaErrors(cudaMallocManaged(&adam.d_m_b, sizeof(float) * out_channels));
		checkCudaErrors(cudaMalloc(&adam.d_m_b, sizeof(float) * out_channels));
		checkCudaErrors(cudaMemset(adam.d_m_b, 0, sizeof(float) * out_channels));

		//checkCudaErrors(cudaMallocManaged(&adam.d_v_b, sizeof(float) * out_channels));
		checkCudaErrors(cudaMalloc(&adam.d_v_b, sizeof(float) * out_channels));
		checkCudaErrors(cudaMemset(adam.d_v_b, 0, sizeof(float) * out_channels));
		//for (size_t i = 0; i < bias_value.size(); ++i)
		//{
		//	adam.d_m_b[i] = 0;
		//	adam.d_v_b[i] = 0;
		//}
		//checkCudaErrors(cudaMallocManaged(&adam.tmp_b, sizeof(float) * out_channels));
		checkCudaErrors(cudaMalloc(&adam.tmp_b, sizeof(float) * out_channels));
		update_data += sizeof(float) * out_channels * 3;		
		//
	}

	if (nn->update_type == mSGD)
	{
		//checkCudaErrors(cudaMallocManaged(&msgd.d_v_b, sizeof(float) * out_channels));
		checkCudaErrors(cudaMalloc(&msgd.d_v_b, sizeof(float) * out_channels));
		//checkCudaErrors(cudaMemset(msgd.d_v_b, 0, sizeof(float) * out_channels));
		//for (size_t i = 0; i < bias_value.size(); ++i)
		//	msgd.d_v_b[i] = 0;
		update_data += sizeof(float) * out_channels;
	}
	if (nn->DEBUG_LEVEL > 2)
		std::cout << "Conv memory allocated, update: " << update_data / 1048576.0 << "MB" << std::endl;
	nn->gpu_memory_allocated += update_data;

	////

	if (!nn->pretrained)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		// Xavier weight filling
		//float w = sqrt(3.0f / (kernel_size * kernel_size * in_channels));
		float w = 1.0f / (kernel_size * kernel_size * in_channels);
		std::uniform_real_distribution<> d(-w, w);

		//// TODO: remove vectors since Unified memory implemented, or not
		// Randomize network
		for (auto&& iter : kernel_value)		
			iter = static_cast<float>(d(gen));		
		for (auto&& iter : bias_value)
			iter = 0.0f;
			//iter = static_cast<float>(d(gen));

		checkCudaErrors(cudaMemcpyAsync(filter_data, kernel_value.data(), sizeof(float) * kernel_value.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(d_bias, bias_value.data(), sizeof(float) * bias_value.size(), cudaMemcpyHostToDevice));

		//checkCudaErrors(cudaDeviceSynchronize());
		//for (size_t i = 0; i < kernel_value.size(); ++i)
		//	filter_data[i] = kernel_value[i];
		//for (size_t i = 0; i < bias_value.size(); ++i)
		//	d_bias[i] = bias_value[i];
	}
	
	checkCUDNN(cudnnCreateConvolutionDescriptor(&desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(desc,
		/*pad_height=*/pad,
		/*pad_width=*/pad,
		/*vertical_stride=*/1,
		/*horizontal_stride=*/1,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION,
		/*computeType=*/CUDNN_DATA_FLOAT));

	int obatch_size, ochannels, oheight, owidth;
	checkCUDNN(
		cudnnGetConvolution2dForwardOutputDim(desc,
			src->tensor,
			filterDesc,
			&obatch_size, &ochannels,
			&oheight, &owidth));

	size_t image_bytes = obatch_size * ochannels * oheight * owidth * sizeof(float);

	if (nn->DEBUG_LEVEL > 2)
		std::cout << obatch_size << " " << ochannels << " " << oheight << " " << owidth << " " << image_bytes << std::endl;
	assert(nn->m_batchSize == obatch_size);
	assert(out_channels == ochannels);
	out_height = oheight;
	out_width = owidth;
	
	define_tensor(nn);

	bool with_limit = true;
	cudnnConvolutionFwdPreference_t fwdPref;
	cudnnConvolutionBwdDataPreference_t bwdDPref;
	cudnnConvolutionBwdFilterPreference_t bwdFPref;
	size_t memory_limit;

	if (with_limit)
	{
		memory_limit = 1073741824 * 1.5; // == 1.5 GB
		fwdPref = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
		bwdDPref = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
		bwdFPref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
	}
	else
	{
		memory_limit = 0;
		fwdPref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
		bwdDPref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
		bwdFPref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
	}
	
	checkCUDNN(
		cudnnGetConvolutionForwardAlgorithm(nn->cudnnHandle,
			src->tensor, // input desc
			filterDesc, // kernel desc
			desc, // convolution desc
			tensor, //output desc
			fwdPref,
			/*memoryLimitInBytes=*/memory_limit,
			&fw_algo));

	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(nn->cudnnHandle,
		src->tensor, // input desc
		filterDesc, // kernel desc
		desc, // convolution desc
		tensor, //output desc
		fw_algo,
		&workspace_bytes));
	if (nn->DEBUG_LEVEL > 1)
	{
		if (name=="")
			std::cerr << "Conv_" << out_channels << " Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
		else
			std::cerr << name << " Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
	}

	if (workspace_bytes > nn->m_workspaceSize)
		nn->m_workspaceSize = workspace_bytes;

	if (nn->update_type != ONLY_INFERENCE)
	{
		// set backpropagation

		checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
			nn->cudnnHandle, src->tensor, tensor, desc, filterDesc,
			bwdFPref,
			/*memoryLimitInBytes=*/memory_limit,
			&bwF_algo));

		checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			nn->cudnnHandle, src->tensor, tensor, desc, filterDesc,
			bwF_algo, &workspace_bytes));

		nn->m_workspaceSize = std::max(nn->m_workspaceSize, workspace_bytes);

		checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
			nn->cudnnHandle, filterDesc, tensor, desc, src->tensor,
			bwdDPref,
			/*memoryLimitInBytes=*/memory_limit,
			&bwD_algo));

		checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
			nn->cudnnHandle, filterDesc, tensor, desc, src->tensor,
			bwD_algo, &workspace_bytes));

		nn->m_workspaceSize = std::max(nn->m_workspaceSize, workspace_bytes);
	}
}

void Neural_Network::Conv_layer::forward_propagation(Layer * src, Neural_Network * nn)
{
	//std::cout << out_channels << std::endl;

	float alpha = 1.0f, beta = 0.0f; // not sure, maybe it can be a nn member
	checkCUDNN(cudnnConvolutionForward(nn->cudnnHandle, &alpha, 
		src->tensor, src->data,		
		filterDesc, filter_data,
		desc,
		fw_algo,
		nn->cudnn_workspace, nn->m_workspaceSize,
		&beta,		
		tensor, data));
	checkCUDNN(cudnnAddTensor(nn->cudnnHandle, &alpha, bias_tensor,
		d_bias, &alpha, tensor, data));
}

void Neural_Network::Conv_layer::back_propagation(Layer * dst, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f; // not sure, maybe it can be a nn member

	checkCUDNN(cudnnConvolutionBackwardBias(nn->cudnnHandle, &alpha, tensor,
		diff_data, &beta, bias_tensor, d_gbias));

	checkCUDNN(cudnnConvolutionBackwardFilter(nn->cudnnHandle, &alpha, dst->tensor,
		dst->data, tensor, diff_data, desc,
		bwF_algo, nn->cudnn_workspace, nn->m_workspaceSize,
		&beta, filterDesc, gfdata));
	if (dst->type != INPUT)
	{
		checkCUDNN(cudnnConvolutionBackwardData(nn->cudnnHandle, &alpha, filterDesc,
			filter_data, tensor, diff_data, desc,
			bwD_algo, nn->cudnn_workspace, nn->m_workspaceSize,
			&beta, dst->tensor, dst->diff_data));
	}
	// Update weights
	if (nn->update_type == SGD)
	{
		alpha = -nn->learning_rate;
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&alpha, gfdata, 1, filter_data, 1));
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&alpha, d_gbias, 1, d_bias, 1));
	}

	if (nn->update_type == mSGD)
	{
		//std::cout << out_channels << " backprop mSGD\n";
		//show_some_data(filter_data, 10);
		checkCudaErrors(cublasSscal(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&nn->msgd.momentum, msgd.d_v_f, 1)); // v = momentum * v
		alpha = -nn->msgd.L2*nn->msgd.lr; // alpha = -L2*epsilon
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&alpha, filter_data, 1, msgd.d_v_f, 1)); // v = -L2*epsilon*w + v
		alpha = -nn->msgd.lr; // alpha = -epsilon
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&alpha, gfdata, 1, msgd.d_v_f, 1)); // v = -epsilon*grad + v
		alpha = 1;
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&alpha, msgd.d_v_f, 1, filter_data, 1)); // w = v + w
		//show_some_data(filter_data, 10);

		// bias
		checkCudaErrors(cublasSscal(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&nn->msgd.momentum, msgd.d_v_b, 1)); // v = momentum * v
		alpha = -nn->msgd.L2*nn->msgd.lr; // alpha = -L2*epsilon
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&alpha, d_bias, 1, msgd.d_v_b, 1)); // v = -L2*epsilon*w + v
		alpha = -nn->msgd.lr; // alpha = -epsilon
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&alpha, d_gbias, 1, msgd.d_v_b, 1)); // v = -epsilon*grad + v
		alpha = 1;
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&alpha, msgd.d_v_b, 1, d_bias, 1)); // w = v + w

	}

	if (nn->update_type == ADAM)
	{		
		checkCudaErrors(cublasSscal(nn->cublasHandle,static_cast<int>(kernel_value.size()),
			&nn->adam.beta1, adam.d_m_f, 1)); // m = b1 * m
		float tmp = (1 - nn->adam.beta1);
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&tmp, gfdata, 1, adam.d_m_f, 1)); // m = (1 - b1) * g + m
		checkCudaErrors(cublasSscal(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&nn->adam.beta2, adam.d_v_f, 1)); // v = b2 * v
		tmp = (1 - nn->adam.beta2);

		elementwise_multiplication(gfdata, gfdata, adam.tmp_f, kernel_value.size()); // tmp_f = g * g elementwise
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&tmp, adam.tmp_f, 1, adam.d_v_f, 1)); // v = (1 - b2) * g^2 + v
		//tmp = 1 / (1 - std::pow(nn->adam.beta1,nn->adam.t));
		//checkCudaErrors(cublasSscal(nn->cublasHandle, static_cast<int>(kernel_value.size()),
		//	&tmp, adam.d_m_f, 1)); // m = m / (1 - b1^t)
		//tmp = 1 / (1 - std::pow(nn->adam.beta2, nn->adam.t));
		//checkCudaErrors(cublasSscal(nn->cublasHandle, static_cast<int>(kernel_value.size()),
		//	&tmp, adam.d_v_f, 1)); // v = v / (1 - b2^t)

		elementwise_sqrt(adam.d_v_f, adam.tmp_f, kernel_value.size()); // tmp_f = sqrt(v)
		elementwise_add(adam.tmp_f, adam.tmp_f, nn->adam.epsilon_t, kernel_value.size()); // tmp_f = (sqrt(v) + epsilon)
		elementwise_devision(adam.d_m_f, adam.tmp_f, adam.tmp_f, kernel_value.size()); //tmp_f =  m / (sqrt(v) + epsilon)
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(kernel_value.size()),
			&nn->adam.alpha_t, adam.tmp_f, 1, filter_data, 1)); // theta = theta - alpha * m / (sqrt(v) + epsilon) // alpha = - alpha already

		//bias
		checkCudaErrors(cublasSscal(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&nn->adam.beta1, adam.d_m_b, 1)); // m = b1 * m
		tmp = (1 - nn->adam.beta1);
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&tmp, d_gbias, 1, adam.d_m_b, 1)); // m = (1 - b1) * g + m
		checkCudaErrors(cublasSscal(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&nn->adam.beta2, adam.d_v_b, 1)); // v = b2 * v
		tmp = (1 - nn->adam.beta2);		
		elementwise_multiplication(d_gbias, d_gbias, adam.tmp_b, bias_value.size()); // tmp_m = g * g elementwise
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&tmp, adam.tmp_b, 1, adam.d_v_b, 1)); // v = (1 - b2) * g^2 + v

		elementwise_sqrt(adam.d_v_b, adam.tmp_b, bias_value.size()); // tmp_f = sqrt(v)
		elementwise_add(adam.tmp_b, adam.tmp_b, nn->adam.epsilon_t, bias_value.size()); // tmp_f = (sqrt(v) + epsilon)
		elementwise_devision(adam.d_m_b, adam.tmp_b, adam.tmp_b, bias_value.size()); //tmp_f =  m / (sqrt(v) + epsilon)
		checkCudaErrors(cublasSaxpy(nn->cublasHandle, static_cast<int>(bias_value.size()),
			&nn->adam.alpha_t, adam.tmp_b, 1, d_bias, 1)); // theta = theta - alpha * m / (sqrt(v) + epsilon) // alpha = - alpha already			
	}
}

void Neural_Network::Conv_layer::load_weights()
{
	// Deprecated, but works
	checkCudaErrors(cudaMemcpyAsync(filter_data, 
									kernel_value.data(), 
									sizeof(float) * kernel_value.size(), 
									cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_bias,
									bias_value.data(),
									sizeof(float) * bias_value.size(),
									cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaDeviceSynchronize());

	//for (size_t i = 0; i < kernel_value.size(); ++i)
	//	filter_data[i] = kernel_value[i];
	//for (size_t i = 0; i < bias_value.size(); ++i)
	//	d_bias[i] = bias_value[i];
}

void Neural_Network::Conv_layer::convert_and_load()
{
	using namespace std;
	vector<float> n_w;

	if (pad != 0 || true)
	{
		for (int o = 0; o < out_channels; ++o)		
			for (int i = 0; i < in_channels; ++i)
				for (int h = 0; h < kernel_size; ++h)
					for (int w = 0; w < kernel_size; ++w)
						n_w.push_back(
							kernel_value[h*kernel_size*in_channels*out_channels + w*in_channels*out_channels + i*out_channels + o]);		
	}
	else
	{
		//this part is incorrect
		for (int o = 0; o < out_channels; ++o)			
			for (int i = 0; i < in_channels*kernel_size*kernel_size; ++i)
						n_w.push_back(
							kernel_value[i*out_channels + o]);		
	}

	printf("Layer %i complete!\n",out_channels);
	//std::cout << n_w.size() << " "<<kernel_value.size()<< endl;
	//for (int i = 0; i < 12; i++)
	//	cout << kernel_value[i] << " ";
	//cout << endl;

	kernel_value = n_w;

	//for (int i = 0; i < 12; i++)
	//	cout << kernel_value[i] << " ";
	//cout << endl;
	load_weights();
}

void Neural_Network::Conv_layer::save_weights(const char * nn_name)
{
	std::stringstream ssf;
	ssf << nn_name << "\\test.bin";

	// Write weights file
	FILE *fp = fopen(ssf.str().c_str(), "wb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
		exit(2);
	}
	//fwrite(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
	fwrite("Hello, world!", sizeof(char), 14, fp);
	fclose(fp);
}

void Neural_Network::Conv_layer::redefine(Layer * src, Neural_Network * nn)
{
	int obatch_size, ochannels, oheight, owidth;
	checkCUDNN(
		cudnnGetConvolution2dForwardOutputDim(desc,
			src->tensor,
			filterDesc,
			&obatch_size, &ochannels,
			&oheight, &owidth));

	//size_t image_bytes = obatch_size * ochannels * oheight * owidth * sizeof(float);
	//std::cout << src->out_height<<" "<<oheight << " " << src->out_width<<" "<<owidth << std::endl;
		
	assert(nn->m_batchSize == obatch_size);
	assert(out_channels == ochannels);
	out_height = oheight;
	out_width = owidth;

	redefine_tensor(nn);
}

void Neural_Network::Conv_layer::load_specified_weights(float * data, size_t size)
{
	assert(size == kernel_value.size() * sizeof(float));
	checkCudaErrors(cudaMemcpyAsync(filter_data, data, size, cudaMemcpyDeviceToDevice));
}

void Neural_Network::Conv_layer::show_weights()
{
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(kernel_value.data(), filter_data, kernel_size*kernel_size*out_channels * sizeof(float), cudaMemcpyDeviceToHost));

	for (int c = 0; c < out_channels; ++c)
	{
		for (int h = 0; h < kernel_size; ++h)
		{
			for (int w = 0; w < kernel_size; ++w)
				std::cout << kernel_value[c*kernel_size*kernel_size + h*kernel_size + w] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

Neural_Network::Conv_layer::~Conv_layer()
{
	//std::cout << "Conv destructor!\n";
	// TODO: check if something missing
	checkCUDNN(cudnnDestroyConvolutionDescriptor(desc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(bias_tensor));
	if (filter_data != nullptr)
		checkCudaErrors(cudaFree(filter_data));
	if (gfdata != nullptr)
		checkCudaErrors(cudaFree(gfdata));
	if (d_bias != nullptr)
		checkCudaErrors(cudaFree(d_bias));
	if (d_gbias != nullptr)
		checkCudaErrors(cudaFree(d_gbias));
	if (adam.d_m_f != nullptr)
		checkCudaErrors(cudaFree(adam.d_m_f));
	if (adam.d_m_f != nullptr)
		checkCudaErrors(cudaFree(adam.d_v_f));
	if (adam.d_m_f != nullptr)
		checkCudaErrors(cudaFree(adam.d_m_b));
	if (adam.d_m_f != nullptr)
		checkCudaErrors(cudaFree(adam.d_v_b));
	if (adam.tmp_f != nullptr)
		checkCudaErrors(cudaFree(adam.tmp_f));
	if (adam.tmp_b != nullptr)
		checkCudaErrors(cudaFree(adam.tmp_b));
	kernel_value.clear();
	bias_value.clear();
}

Neural_Network::RELU_layer::RELU_layer(std::string name) : Layer(name)
{
	type = RELU;
	checkCUDNN(cudnnCreateActivationDescriptor(&relu_desc));
	checkCUDNN(cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU,
		CUDNN_PROPAGATE_NAN, 0.0));
}

void Neural_Network::RELU_layer::set_forward_propagation(Layer * src, Neural_Network * nn)
{
	out_channels = src->out_channels;
	out_height = src->out_height;
	out_width = src->out_width;
	link_tensors(src, nn);
	//define_tensor(nn);
}

void Neural_Network::RELU_layer::forward_propagation(Layer * src, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f; // not sure, maybe it can a nn member
	checkCUDNN(cudnnActivationForward(nn->cudnnHandle, relu_desc, &alpha,
		src->tensor, src->data, &beta, tensor, data));
}

void Neural_Network::RELU_layer::back_propagation(Layer * dst, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f; // not sure ...
	checkCUDNN(cudnnActivationBackward(nn->cudnnHandle, relu_desc, &alpha,
		tensor, data, tensor, diff_data,
		dst->tensor, dst->data, &beta, dst->tensor, dst->diff_data));
}

void Neural_Network::RELU_layer::redefine(Layer * src, Neural_Network * nn)
{
	out_channels = src->out_channels;
	out_height = src->out_height;
	out_width = src->out_width;
	link_tensors(src, nn);
}

Neural_Network::RELU_layer::~RELU_layer()
{
	checkCUDNN(cudnnDestroyActivationDescriptor(relu_desc));
}

Neural_Network::Input_layer::Input_layer(int height_, int width_, int channels_, Neural_Network *nn, std::string name) : Layer(name)
{
	type = INPUT;
	out_channels = channels_;
	out_height = height_;
	out_width = width_;
	define_tensor(nn);
}

void Neural_Network::Input_layer::input_data(float * data_, Neural_Network *nn)
{
	//auto t3 = std::chrono::high_resolution_clock::now();
	//checkCudaErrors(cudaMemcpy(data, data_, nn->m_batchSize * out_channels * out_height * out_width * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(data, data_, nn->m_batchSize * out_channels * out_height * out_width * sizeof(float), cudaMemcpyDeviceToDevice));
	//auto t4 = std::chrono::high_resolution_clock::now();
	//printf("Copy time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0f);
	//size_t size = nn->m_batchSize * out_channels * out_height * out_width;
	//for (int i = 0; i < size; ++i)
	//	data[i] = data_[i];
}

void Neural_Network::Input_layer::redefine(int height_, int width_, int channels_, Neural_Network * nn)
{
	out_channels = channels_;
	out_height = height_;
	out_width = width_;
	redefine_tensor(nn);
}

void Neural_Network::FC_layer::set_forward_propagation(Layer * src, Neural_Network * nn)
{
	in_channels = src->out_channels;
	in_height = src->out_height;
	in_width = src->out_width;

	out_height = 1;
	out_width = 1;
	define_tensor(nn);

	neurons.resize(src->out_channels * src->out_height * src->out_width * out_channels);	
	checkCudaErrors(cudaMalloc(&neurons_data, sizeof(float) * neurons.size()));	
	if (!nn->pretrained)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		// Xavier weight filling
		float w = sqrt(3.0f / (neurons.size()));
		std::uniform_real_distribution<> d(-w, w);

		// Randomize network
		for (auto&& iter : neurons)
			iter = static_cast<float>(d(gen));

		checkCudaErrors(cudaMemcpyAsync(neurons_data, neurons.data(), sizeof(float) * neurons.size(), cudaMemcpyHostToDevice));
	}	

	bias.resize(out_channels);
	checkCudaErrors(cudaMalloc(&d_bias, sizeof(float) * bias.size()));
	if (!nn->pretrained)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		// Xavier weight filling
		float w = sqrt(3.0f / (bias.size()));
		std::uniform_real_distribution<> d(-w, w);

		// Randomize network
		for (auto&& iter : bias)
			iter = static_cast<float>(d(gen));

		checkCudaErrors(cudaMemcpyAsync(d_bias, bias.data(), sizeof(float) * bias.size(), cudaMemcpyHostToDevice));
	}
}

void Neural_Network::FC_layer::forward_propagation(Layer * src, Neural_Network * nn)
{
	//not tested - need to be checked
	float alpha = 1.0f, beta = 0.0f; // not sure, maybe it can a nn member
	// FC1 layer
	// Forward propagate neurons using weights (fc1 = pfc1'*pool2)
	checkCudaErrors(cublasSgemm(nn->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		out_channels, nn->m_batchSize, src->out_channels * src->out_height * src->out_width,
		&alpha,
		neurons_data, src->out_channels * src->out_height * src->out_width,
		src->data, src->out_channels * src->out_height * src->out_width,
		&beta,
		data, out_channels));

	// Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
	checkCudaErrors(cublasSgemm(nn->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		out_channels, nn->m_batchSize, 1,
		&alpha,
		d_bias, out_channels,
		nn->d_onevec, 1,
		&alpha,
		data, out_channels));
}

void Neural_Network::FC_layer::convert_and_load()
{
	using namespace std;
	vector<float> n_w;

	for (int o = 0; o < out_channels; ++o)
		for (int i = 0; i < in_channels; ++i)
			for (int h = 0; h < in_height; ++h)
				for (int w = 0; w < in_width; ++w)
					n_w.push_back(
						neurons[h*in_width*in_channels*out_channels + w*in_channels*out_channels + i*out_channels + o]);

	//for (int o = 0; o < out_channels; ++o)
	//	for (int i = 0; i < neurons.size()/out_channels; ++i)
	//		n_w.push_back(
	//			neurons[i*out_channels + o]);


	printf("Layer %i complete!\n", out_channels);
	//std::cout << n_w.size() << " "<<kernel_value.size()<< endl;
	//for (int i = 0; i < 12; i++)
	//	cout << kernel_value[i] << " ";
	//cout << endl;

	neurons = n_w;

	//for (int i = 0; i < 12; i++)
	//	cout << kernel_value[i] << " ";
	//cout << endl;
	load_weights();
}

void Neural_Network::FC_layer::load_weights()
{
	checkCudaErrors(cudaMemcpyAsync(neurons_data,
		neurons.data(),
		sizeof(float) * neurons.size(),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_bias,
		bias.data(),
		sizeof(float) * bias.size(),
		cudaMemcpyHostToDevice));
}

Neural_Network::FC_layer::~FC_layer()
{
	neurons.clear();
	if (neurons_data != nullptr)
		delete[] neurons_data;
}

void Neural_Network::SoftMax_layer::set_forward_propagation(Layer * src, Neural_Network * nn)
{
	out_channels = src->out_channels;
	out_height = src->out_height;
	out_width = src->out_width;
	define_tensor(nn);
}

void Neural_Network::SoftMax_layer::forward_propagation(Layer * src, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnSoftmaxForward(nn->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha, src->tensor, src->data, &beta, tensor, data));
}

void Neural_Network::SoftMax_layer::back_propagation(Layer * dst, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f;	
	checkCUDNN(cudnnSoftmaxBackward(nn->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha, tensor, data, tensor, diff_data, &beta, dst->tensor, dst->diff_data));
}

void Neural_Network::SoftMax_layer::redefine(Layer * src, Neural_Network * nn)
{
	out_channels = src->out_channels;
	out_height = src->out_height;
	out_width = src->out_width;
	define_tensor(nn);
}

Neural_Network::Dropout_layer::Dropout_layer(float dropout_ratio, std::string name) : Layer(name)
{
	type = DROPOUT;
	dropout = dropout_ratio;
	checkCUDNN(cudnnCreateDropoutDescriptor(&desc));	
}

void Neural_Network::Dropout_layer::set_forward_propagation(Layer * src, Neural_Network * nn)
{
	out_channels = src->out_channels;
	out_height = src->out_height;
	out_width = src->out_width;
	define_tensor(nn);

	
	checkCUDNN(cudnnDropoutGetStatesSize(
		nn->cudnnHandle,
		&statesSpace_size));
	//checkCudaErrors(cudaMallocManaged(&d_statesSpace, statesSpace_size));
	checkCudaErrors(cudaMalloc(&d_statesSpace, statesSpace_size));
	std::cout << "Dropout memory allocated, statesSpace: " << statesSpace_size / 1048576.0 << "MB" << std::endl;
	nn->gpu_memory_allocated += statesSpace_size;

	std::random_device rd;
	std::mt19937 gen(rd());
		
	std::uniform_int_distribution<> d(0, INTMAX_MAX);

	checkCUDNN(cudnnSetDropoutDescriptor(
		desc,
		nn->cudnnHandle,
		dropout,
		d_statesSpace, statesSpace_size, 
		d(gen)));
		
	checkCUDNN(cudnnDropoutGetReserveSpaceSize(
		tensor,
		&reservedSpace_size));

	//checkCudaErrors(cudaMallocManaged(&d_reservedSpace, reservedSpace_size));
	checkCudaErrors(cudaMalloc(&d_reservedSpace, reservedSpace_size));
	std::cout << "Dropout memory allocated, reservedSpace: " << reservedSpace_size / 1048576.0 << "MB" << std::endl;
	nn->gpu_memory_allocated += reservedSpace_size;
}

void Neural_Network::Dropout_layer::forward_propagation(Layer * src, Neural_Network * nn)
{
	checkCUDNN(cudnnDropoutForward(nn->cudnnHandle,
		desc,
		src->tensor, src->data,
		tensor, data,
		d_reservedSpace, reservedSpace_size));
}

void Neural_Network::Dropout_layer::back_propagation(Layer * dst, Neural_Network * nn)
{
	checkCUDNN(cudnnDropoutBackward(nn->cudnnHandle,
		desc,
		tensor, diff_data,
		dst->tensor, dst->diff_data,
		d_reservedSpace, reservedSpace_size));
}

Neural_Network::Dropout_layer::~Dropout_layer()
{
	if (d_reservedSpace != nullptr)
	{
		checkCudaErrors(cudaFree(d_reservedSpace));
		d_reservedSpace = nullptr;
	}

	if (d_statesSpace != nullptr)
	{
		checkCudaErrors(cudaFree(d_statesSpace));
		d_statesSpace = nullptr;
	}
	checkCUDNN(cudnnDestroyDropoutDescriptor(desc));
}

Neural_Network::Unpool_layer::Unpool_layer(Pool_layer * src, Layer * pre_pool_ref_, std::string name) : Layer(name)
{
	type = UNPOOL;
	pool_ref = src;
	pre_pool_ref = pre_pool_ref_;
	size = src->size;
	stride = src->stride;
	desc = src->desc;
}

void Neural_Network::Unpool_layer::set_forward_propagation(Layer * src, Neural_Network * nn)
{
	out_channels = src->out_channels;
	out_height = pre_pool_ref->out_height;
	out_width = pre_pool_ref->out_width;
	//out_height = src->out_height * 2; // not universal unpooling 
	//out_width = src->out_width * 2;   // will work only for size = 2 and stride = 2
	//int batchSize, channels;
	//cudnnGetPooling2dForwardOutputDim(desc, src->tensor, &batchSize, &channels, &out_height, &out_width);
	//assert(batchSize == nn->m_batchSize);
	//assert(channels == out_channels);

	define_tensor(nn);
}

void Neural_Network::Unpool_layer::forward_propagation(Layer * src, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f;
	// not sure - maybe will have to do it manually
	checkCUDNN(cudnnPoolingBackward(nn->cudnnHandle, desc, &alpha,
		pool_ref->tensor, pool_ref->data, src->tensor, src->data, //ref->tensor and src->tensor should be equal...
		pre_pool_ref->tensor, pre_pool_ref->data, &beta, tensor, data));

}

void Neural_Network::Unpool_layer::back_propagation(Layer * dst, Neural_Network * nn)
{
	float alpha = 1.0f, beta = 0.0f;
	// not sure - have to check papers and math behind it..
	checkCUDNN(cudnnPoolingForward(nn->cudnnHandle, desc, &alpha, tensor,
		diff_data, &beta, dst->tensor, dst->diff_data));
}
