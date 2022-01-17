# SYCL_easy_tutorial

## Overview
This is an easy SYCL tutorial written by jhson.  
Feel free to read and use it. (but it is a little bit inefficient.)  
SYCL device codes are inplemented as C++ template function in the directory "include/"
The codes in the directory "test/" are host codes for using these device kernels.

## 0. How to Run
Environemnt
- clang 14.0.0 (https://github.com/intel/llvm.git 8c5b7017a925701ef4034056b5ed8e0fac2a0011)
- cmake 3.20.2
- cuda 10.2
- Intel CPU Runtime for OpenCL 18.1
  
Build steps
1. mkdir build && cd build
2. cmake ..
3. make

## 1. Device Querying Example
Very easy SYCL device querying example   
  
  
## 2. Vector Addition Example
Vesy easy vector addition example implemented with SYCL DPC++.    
No parallel optimization technique implemented.    
A work item only performs a single element-wise addition.  
    
## 3. Matrix Multiplication Example
Simple matrix multiplication example implemented with SYCL DPC++.  
&nbsp;C[M,N] = A[M,K] * B[K,N];  
No parallel optimization technique implemented.    
The work item with ID[m, n] only does dot product between the m-th row of A and the n-th colomn of B.   
  
## 4. Matrix Stencil Operation Example
Simple stencil operation on matrix example implemented with SYCL DPC++.  
This example performs correlation operation.  
No parallel optimization technique implemented.    

## 5. Parallel Reduction Example
Simple parallel reduction example implemented with SYCL DPC++.  
Use the local memory and the sequential addressing optimization [1] technique

## 6. Memory Management Example
Sycl provide two memory management techniques: Unified shared memory (USM) and Buffer.  
USM can use three different memory spaces, which are host, device, and shared.
This example shows the performance of each technique.

## 7. Execution Model Example


#### References
>> [1] Harris, M. (2007). Optimizing parallel reduction in CUDA. NVIDIA Developer Technology.