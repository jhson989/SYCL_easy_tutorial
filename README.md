# SYCL_easy_tutorial

## 0. How to Run
Environemnt
- clang 14.0.0 (https://github.com/intel/llvm.git 8c5b7017a925701ef4034056b5ed8e0fac2a0011)
- cmake 3.20.2
- cuda 10.2
  
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
  