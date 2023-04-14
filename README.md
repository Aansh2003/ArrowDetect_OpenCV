# ArrowDetect_OpenCV
A purely OpenCV(C++) approach to detect arrows without use of machine learning.

## Dependencies
Requires OpenCV(>=4.6.0), install guide available [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html).

## Compiling and Executing

To compile the code with g++ type the following command
```
g++ detect.cpp -o detect `pkg-config --cflags --libs opencv`
```
It is recommended to compile using CMake for easier dependency management. Click [here](https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html) for a guide on how to do so.

Running the code:
```
./detect
```
