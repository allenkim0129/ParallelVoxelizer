
#include <stdio.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <float.h>

dim3 gridSize = 256;
dim3 blockSize = 256;

//////////////////////////////////////////////////////////////////////////////////
//** General vector helper functions **//
//////////////////////////////////////////////////////////////////////////////////
__device__
void crossProduct(const glm::fvec3& A, const glm::fvec3& B, glm::fvec3& C)
{
    C.x = A.y*B.z - A.z*B.y;
    C.y = A.z*B.x - A.x*B.z;
    C.z = A.x*B.y - A.y*B.x;
    return;
}

__device__
double dotProduct(const glm::fvec3& A, const glm::fvec3& B)
{
    return A.x*B.x + A.y*B.y + A.z*B.z;
}

__device__
double vecLength(const glm::fvec3& vec)
{
    return std::sqrt(dotProduct(vec, vec));
}

__device__
float fSaturate(float f)
{
	return max(0.0f, min(f, 1.0f));
}

///////////////////////////////////////////////////////////////////////////////////
//** Bounding Box Kernel **//
///////////////////////////////////////////////////////////////////////////////////
__global__
void findRecudedBoundingBox(const glm::fvec3 *vertices, glm::fmat2x3 *box, int *mutex, uint n)
{
	unsigned int index  = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	//shard variable couldn't be defined with glm vectors
	//Since fmat2x3 is a float array with 2 x 3 dimension, 
	//multiplied 6 with the number of thread
	__shared__ float s_cache[256*6];
	glm::fmat2x3* cache = (glm::fmat2x3*)&s_cache;

	//Init temp box with float max values.
	glm::fmat2x3 temp(FLT_MAX, FLT_MAX, FLT_MAX,
                     -FLT_MAX, -FLT_MAX,  -FLT_MAX);

	//loop over all blocks to get min/max of the same thread index.
	//This enable this knernal to handle n greater than number of threads in each block
	while(index + offset < n){
		temp[0] = glm::min(temp[0], vertices[index + offset]);
		temp[1] = glm::max(temp[1], vertices[index + offset]);

		offset += stride;
	}

	//store the min/max to shared memory
	cache[threadIdx.x] = temp;

	//Sycronize before reducing, because some threads still might be working.
	__syncthreads();
	
	// general reduction loop
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x][0] = glm::min(cache[threadIdx.x][0], cache[threadIdx.x + i][0]);
			cache[threadIdx.x][1] = glm::max(cache[threadIdx.x][1], cache[threadIdx.x + i][1]);
		}

		__syncthreads();
		i /= 2;
	}

	//Since there is no atomic min/max, use mutex value to lock and unlock threads.
	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		(*box)[0] = glm::min((*box)[0], cache[0][0]);
		(*box)[1] = glm::max((*box)[1], cache[0][1]);
		atomicExch(mutex, 0);  //unlock
	}
}

///////////////////////////////////////////////////////////////////////////////////
//** Triangle intersection Kernel **//
///////////////////////////////////////////////////////////////////////////////////

//TBD

///////////////////////////////////////////////////////////////////////////////////
//** signed_distance Kernels **/
///////////////////////////////////////////////////////////////////////////////////
__device__
void findClosestPointOnTriangle(const glm::fmat3x3& triangle, const glm::fvec3* center, glm::fvec3& closestPoint)
{
	glm::fvec3 edge0 = triangle[1] - triangle[0];
	glm::fvec3 edge1 = triangle[2] - triangle[0];
	glm::fvec3 v0    = triangle[0] - *center;

	float a = dotProduct(edge0, edge0);
	float b = dotProduct(edge0, edge1);
	float c = dotProduct(edge1, edge1);
	float d = dotProduct(edge0, v0);
	float e = dotProduct(edge1, v0);

	float det = a*c - b*b;
	float s   = b*e - c*d;
	float t   = b*d - a*e;

	if (s + t < det)
	{
		if (s < 0.0f)
		{
			if (t < 0.0f)
			{
				if (d < 0.0f)
				{
					s = fSaturate(-d / a);
					t = 0.0f;
				}
				else
				{
					s = 0.0f;
					t = fSaturate(-e / c);
				}
			}
			else
			{
				s = 0.0f;
				t = fSaturate(-e / c);
			}
		}
		else if (t < 0.0f)
		{
			s = fSaturate(-d / a);
			t = 0.f;
		}
		else
		{
			float invDet = 1.0f / det;
			s *= invDet;
			t *= invDet;
		}
	}
	else
	{
		if (s < 0.0f)
		{
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0)
			{
				float numer = tmp1 - tmp0;
				float denom = a - 2.0f*b + c;
				s = fSaturate(numer / denom);
				t = 1.0f - s;
			}
			else
			{
				t = fSaturate(-e / c);
				s = 0.0f;
			}
		}
		else if (t < 0.0f)
		{
			if (a + d > b + e)
			{
				float numer = c + e - b - d;
				float denom = a - 2.0f*b + c;
				s = fSaturate(numer / denom);
				t = 1.0f - s;
			}
			else
			{
				s = fSaturate(-e / c);
				t = 0.0f;
			}
		}
		else
		{
			float numer = c + e - b - d;
			float denom = a - 2.0f*b + c;
			s = fSaturate(numer / denom);
			t = 1.0f - s;
		}
	}

	closestPoint = triangle[0] + (s * edge0) + (t * edge1);
}

__global__
void calcSignedDistance(const glm::fmat3x3 *triangles, const glm::fvec3 *center, float* dists, float* signs, uint n)
{
	glm::fvec3 norm, closestPoint, delta;

	unsigned int index  = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	while ((index + offset) < n)
	{
		const glm::fmat3x3& triangle = triangles[index + offset];

		//vector cross to get normal
		crossProduct(triangle[2]-triangle[0], triangle[1]-triangle[0], norm);

		//calc closestPoint
		findClosestPointOnTriangle(triangle, center, closestPoint);

		//calc delta
		delta = closestPoint - *center;

		//calc sign
		signs[index + offset] = dotProduct(delta, norm) < 0 ? -1.0f : 1.0f;

		//calc dist
		dists[index + offset] = vecLength(delta);	

		offset += stride;
	}	
}

__global__
void findRecudedMin(float *dists, float *signs, float *min, float *sign, int *mutex, uint n)
{
	uint index  = threadIdx.x + blockIdx.x*blockDim.x;
	uint stride = gridDim.x*blockDim.x;
	uint offset = 0;

	__shared__ float cache1[256];
	__shared__ float cache2[256];

	float temp_dist = FLT_MAX;
	float temp_sign = 0.0;
	while (index + offset < n)
	{
		if (dists[index + offset] < temp_dist )
		{
			temp_dist = dists[index + offset];
			temp_sign = signs[index + offset];
		}
		// temp = std::min(temp, dists[index + offset]);

		offset += stride;
	}

	cache1[threadIdx.x] = temp_dist;
	cache2[threadIdx.x] = temp_sign;
	__syncthreads();


	// reduction
	uint i = blockDim.x/2;
	while (i != 0)
	{
		if(threadIdx.x < i){
			if (cache1[threadIdx.x + i] < cache1[threadIdx.x])
			{
				cache1[threadIdx.x] = cache1[threadIdx.x + i];
				cache2[threadIdx.x] = cache2[threadIdx.x + i];
			}
			// cache[threadIdx.x] = std::min(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0)
	{
		while (atomicCAS(mutex,0,1) != 0);  //lock
		if (cache1[0] < *min)
		{
			*min = cache1[0];
			*sign = cache2[0];
		}
		// *min = std::min(*min, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}






///////////////////////////////////////////////////////////////////////////////////
//** extern functions **/
///////////////////////////////////////////////////////////////////////////////////
//Return AABB of triangle meshes
void getBoundingBox(uint n, const glm::fmat3x3* d_triangles, 
	                       glm::fmat2x3& h_box)
{
	glm::fmat2x3 *d_box;
	int        *d_mutex;

	//allocate
	cudaMalloc((void**)&d_box, sizeof(glm::fmat2x3));
	
	cudaMalloc((void**)&d_mutex, sizeof(int));
	cudaMemset(d_mutex, 0, sizeof(int));

	//copy to device
	cudaMemcpy(d_box, &h_box, sizeof(glm::fmat2x3), cudaMemcpyHostToDevice);


	//call kernel
	findRecudedBoundingBox<<< gridSize, blockSize >>>(
				              (const glm::fvec3*)d_triangles, 
				              d_box, 
				              d_mutex, 
				              n*3);
	cudaDeviceSynchronize();

	//copy back to host
	cudaMemcpy(&h_box, d_box, sizeof(glm::fmat2x3), cudaMemcpyDeviceToHost);

	//free
	cudaFree(d_box);
	cudaFree(d_mutex);
}

//Return true if there is a intersection between a node and triangle meshes
bool isBoxleIntersectsTriangle(uint n, const glm::fmat3x3* d_tri_meshes,
							   const glm::fvec3& min,const glm::fvec3& max)
{

}

//Return the shortest dist between a node and triangle meshes.
float calcLeafNodeSignedDistance(uint n, const glm::fmat3x3* d_triangles,
							     const glm::fvec3& h_center)
{
	//allocate
	float signed_dist;
	float *h_dist, *d_dist, *h_sign, *d_sign, 
	      *d_dists, *d_signs;
  	int *d_mutex;
	glm::fvec3 *d_center;

	h_dist = (float*)malloc(sizeof(float));
	h_sign = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&d_dist, sizeof(float));
	cudaMalloc((void**)&d_sign, sizeof(float));
	cudaMalloc((void**)&d_dists, n*sizeof(float));
	cudaMalloc((void**)&d_signs, n*sizeof(float));
	cudaMalloc((void**)&d_mutex, sizeof(int));
	cudaMalloc((void**)&d_center, sizeof(glm::fvec3));
	cudaMemset(d_mutex, 0, sizeof(float));

	//copy center from host to device
	cudaMemcpy(d_center, &h_center, sizeof(glm::fvec3), cudaMemcpyHostToDevice);

	//get distance to every triangle along with its sign
	calcSignedDistance<<< gridSize, blockSize >>>(d_triangles, 
		                                          d_center,
		                                          d_dists,
		                                          d_signs,
		                                          n);
	cudaDeviceSynchronize();
	//reduce get min
	findRecudedMin<<< gridSize, blockSize >>>(d_dists, d_signs, 
		                                      d_dist, d_sign, 
		                                      d_mutex, n);
	cudaDeviceSynchronize();
	//copy dist and sign from devicce to host
	cudaMemcpy(h_dist, d_dists, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sign, d_signs, sizeof(float), cudaMemcpyDeviceToHost);

	//Save co
	signed_dist = (*h_dist) * (*h_sign);

	//free
	free(h_dist);
	free(h_sign);
	cudaFree(d_dist);
	cudaFree(d_sign);
	cudaFree(d_dists);
	cudaFree(d_signs);
	cudaFree(d_mutex);
	cudaFree(d_center);

	return signed_dist;
}


///////////////////////////////////////////////////////////////////////////////////
//** testing main **/
///////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <iostream>
#include <cstdlib>
#include <time.h>

void print3DBox(const glm::fmat2x3& box)
{
	printf("min(%f, %f, %f), max(%f, %f, %f)\n", box[0].x, box[0].y, box[0].z,
		                                       box[1].x, box[1].y, box[1].z);
}

void printfvec3(const glm::fvec3& vec)
{
	printf("(%f, %f, %f)\n", vec.x, vec.y, vec.z);
}

void printTriangle(const glm::fmat3x3& tri)
{
	printf("v0:");
	printfvec3(tri[0]);
	printf("v1:");
	printfvec3(tri[1]);
	printf("v2:");
	printfvec3(tri[2]);	
}

//main for bounding box test
int main()
{
	srand( (unsigned)time( NULL ) );

	//** Testing initialization **//
	//1 Million triangles
	uint N = 100000;
	std::vector<glm::fmat3x3> triangles(N);	
	glm::fmat3x3* d_triangles;	

	//Init box
	glm::fmat2x3 box(FLT_MAX, FLT_MAX, FLT_MAX,
                     -FLT_MAX, -FLT_MAX,  -FLT_MAX);
	// print3DBox(box);
	
	//alloc device triangle
	cudaMalloc(&d_triangles, N*sizeof(glm::fmat3x3));

	//init host triangle
	for (int i = 0; i < N; ++i)
	{

		glm::fvec3 v1(float(rand()) / RAND_MAX,
					  float(rand()) / RAND_MAX,
					  float(rand()) / RAND_MAX);
		glm::fvec3 v2(float(rand()) / RAND_MAX,
					  float(rand()) / RAND_MAX,
					  float(rand()) / RAND_MAX);
		glm::fvec3 v3(float(rand()) / RAND_MAX,
					  float(rand()) / RAND_MAX,
					  float(rand()) / RAND_MAX);

		triangles[i] = glm::fmat3x3(v1, v2, v3);
	}

	//copy triangle from host to device
    cudaMemcpy(d_triangles, triangles.data(), 
    	N*sizeof(glm::fmat3x3), cudaMemcpyHostToDevice);
    //////////////////////////////////////////////



    std::cout << "start GPU" << std::endl;
    //** actual GPU boundingbox call here **//
	getBoundingBox(N, d_triangles, box);	
	//check output
	print3DBox(box);
	//////////////////////////////////////


    std::cout << "start CPU" << std::endl;
	//** target CPU boundingbox call here **//
	glm::fmat2x3 t_box(FLT_MAX, FLT_MAX, FLT_MAX,
                      -FLT_MAX, -FLT_MAX,  -FLT_MAX);

	glm::fvec3* vertices = (glm::fvec3*)triangles.data();
	for (int i = 0; i < N*3; ++i)
	{
		t_box[0] = glm::min(t_box[0], vertices[i]);
		t_box[1] = glm::max(t_box[1], vertices[i]);
	}
	print3DBox(t_box);
	//////////////////////////////////////////

	//free
	cudaFree(d_triangles);
}