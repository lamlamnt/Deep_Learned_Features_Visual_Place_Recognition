# Compiler settings
CC = g++ -I/usr/local/cuda/include/ -I/home/lamlam/downloads/TensorRT-8.6.1.6/include/ -I/home/lamlam/miniconda3/envs/opencv-cuda-env/include/opencv4/
CFLAGS = -std=c++11 -Wall

# Source file and executable name
SRC = main.cpp
EXEC = main

all: $(EXEC)

$(EXEC): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXEC)

clean:
	rm -f $(EXEC)
