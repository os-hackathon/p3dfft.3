// Copyright 2021 Fluid Numerics LLC
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation 
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the 
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//
#include "p3dfft.h"


// ************** Float Real ************** //

__global__ void transpose_021_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}

__global__ void transpose_201_fr(float *dIn, float *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}

// ************** Double Real ************** //

__global__ void transpose_021_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}

__global__ void transpose_201_dr(double *dIn, double *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}

// ************** Float Complex ************** //

__global__ void transpose_021_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}

__global__ void transpose_201_fc(cuFloatComplex *dIn, cuFloatComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}

// ************** Double Complex ************** //

__global__ void transpose_021_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(i,k,j,Nx,Nz,Ny)];

  }

}

__global__ void transpose_102_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,i,k,Ny,Nx,Nz)];

  }

}

__global__ void transpose_120_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(j,k,i,Ny,Nz,Nx)];

  }

}

__global__ void transpose_210_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,j,i,Nz,Ny,Nx)];

  }

}

__global__ void transpose_201_dc(cuDoubleComplex *dIn, cuDoubleComplex *dOut, int Nx, int Ny, int Nz){

  size_t i = threadIdx.x + blockDim.x*blockIdx.x;
  size_t j = threadIdx.y + blockDim.y*blockIdx.y;
  size_t k = threadIdx.z + blockDim.z*blockIdx.z;

  if( i < Nx && j < Ny && k < Nz ) {

    dOut[T3D_INDEX(i,j,k,Nx,Ny,Nz)] = dIn[T3D_INDEX(k,i,j,Nz,Nx,Ny)];

  }

}
