This is just a raw C++ code, linking not included. To Work properly you need to link it against CUDA(with cuDNN) and OpenCV (used in CIFAR-10, VGG and DIM).

Neural Network (NN) implementation is located in NN.h and NN.cpp. This class becomed too big, so I probably should use PIMP or separate layers structure from Neural_Network class. Loss calculation happens in direct_gpu_computation.cuh so as mSGD and ADAM optimization.

On input NN expects data stored already in GPU. This is not optimal but allows to increase speed by using threads.
