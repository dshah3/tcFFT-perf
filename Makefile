# OBJ = tcfft_half_accuracy tcfft_half_speed tcfft_half_2d_accuracy tcfft_half_2d_speed \
# 	cufft_half_accuracy cufft_half_speed cufft_half_2d_accuracy cufft_half_2d_speed

OBJ =   tcfft_half_2d_accuracy tcfft_half_2d_speed \
	  cufft_half_2d_accuracy cufft_half_2d_speed

FLAGS = -std=c++11 -lcublas -gencode arch=compute_90,code=sm_90 -res-usage -lcudart -lfftw3 -lcufft -lineinfo -Xcompiler -fopenmp
NCU_PATH := $(shell which ncu)

ifdef DEBUG
FLAGS += -g -G
endif

all : $(OBJ)

tcfft_half_accuracy : accuracy.cpp tcfft_doit_half.cpp tcfft_half.cu
	nvcc $^ -o $@ $(FLAGS)

tcfft_half_speed : speed.cpp tcfft_doit_half.cpp tcfft_half.cu
	nvcc $^ -o $@ $(FLAGS)

tcfft_half_2d_accuracy : accuracy_2d.cpp tcfft_doit_half_2d.cpp tcfft_half_2d.cu
	nvcc $^ -o $@ $(FLAGS)

tcfft_half_2d_speed : speed_2d.cpp tcfft_doit_half_2d.cpp tcfft_half_2d.cu
	nvcc $^ -o $@ $(FLAGS)

cufft_half_accuracy : accuracy.cpp cufft_doit_half.cpp
	nvcc $^ -o $@ $(FLAGS)

cufft_half_speed : speed.cpp cufft_doit_half.cpp
	nvcc $^ -o $@ $(FLAGS)

cufft_half_2d_accuracy : accuracy_2d.cpp cufft_doit_half_2d.cpp
	nvcc $^ -o $@ $(FLAGS)

cufft_half_2d_speed : speed_2d.cpp cufft_doit_half_2d.cpp
	nvcc $^ -o $@ $(FLAGS)

.PHONY : clean

clean :
	rm -f $(OBJ)

profile2d : tcfft_half_2d_speed
	sudo $(NCU_PATH) --set full --import-source yes -o profile2d -f ./tcfft_half_2d_speed
