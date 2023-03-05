#define REGISTER_A_SIZE 512
#define SHARED_B_SIZE 512 * 512
#define TS 4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm_v0(__global const float *A, __global float *B, __global float *C, unsigned int heightA, unsigned int widthA, unsigned int widthB) {
 	unsigned int i = get_global_id(0);  // M
  	unsigned int j = get_global_id(1);  // N
  	float tmp = 0;
  	if ((i < heightA) && (j < widthB))
  	{
      	for (int k = 0; k < widthA; k++) 
		{
        	tmp += A[i * widthA + k] * B[k * widthB + j];
      	}
      	C[i * widthB + j] = tmp;
    }
}

__kernel void gemm_v1(__global const float *A, __global float *B, __global float *C, unsigned int heightA, unsigned int widthA, unsigned int widthB)
{
	unsigned int i = get_global_id(0);
    float tmp = 0;
    if (i < heightA) 
	{
    	for (int j = 0; j < widthB; j++) 
		{
      		tmp = 0;
      		for (int k = 0; k < widthA; k++) 
			{
        		tmp += A[i * widthA + k] * B[k * widthB + j];
      		}
      		C[i * widthB + j] = tmp;
    	}
  	}
}

__kernel void gemm_v2(__global const float *A, __global float *B,
                      __global float *C, int heightA, int widthA, int widthB) {
  	unsigned int i = get_global_id(0);

  	float tmp = 0;
  	float tmpA[REGISTER_A_SIZE];
  	if (i < heightA)
	{
    	for (int k = 0; k < widthA; k++) 
		{
      		tmpA[k] = A[i * widthA + k];
    	}
    	for (int j = 0; j < widthB; j++) 
		{
      		tmp = 0;
      		for (int k = 0; k < widthA; k++) 
			{
        		tmp += tmpA[k] * B[k * widthB + j];
      		}
      		C[i * widthB + j] = tmp;
   		}
  	}
}

__kernel void gemm_v3(__global const float *A, __global float *B, __global float *C, int heightA, int widthA, int widthB) {
  	int i = get_global_id(0);
	int tid = get_local_id(0);
	float tmp = 0;

  	float tmpA[REGISTER_A_SIZE];
  	__local float sharedB[SHARED_B_SIZE];
  	
  	if (i < heightA) 
	{
    	for (int k = 0; k < widthA; k++) 
		{
      		tmpA[k] = A[i * widthA + k];
    	}
		for (int k = 0; k < widthA; k++) 
		{
      		sharedB[k * widthB + tid] = B[k * widthB + tid];
    	}
    	barrier(CLK_LOCAL_MEM_FENCE);
    	for (int j = 0; j < widthB; j++)
		{
			tmp = 0;
      		for (int k = 0; k < widthA; k++) 
			{
        		tmp += tmpA[k] * sharedB[k * widthB + j];
      		}
      		C[i * widthB + j] = tmp;
    	}
  	}
}

__kernel void gemm_v4(__global const float *A, __global float *B, __global float *C, int heightA, int widthA, int widthB) {
  	int i = get_global_id(0);
	int tid = get_local_id(0);
	float tmp = 0;

  	float tmpA[REGISTER_A_SIZE];
  	__local float sharedB[SHARED_B_SIZE];
  	
  	if (i < heightA) 
	{
    	for (int k = 0; k < widthA; k++) 
		{
      		tmpA[k] = A[i * widthA + k];
    	}
		for (int k = 0; k < widthA; k++) 
		{
      		sharedB[k * widthB + tid] = B[k * widthB + tid];
    	}
    	barrier(CLK_LOCAL_MEM_FENCE);
    	for (int j = 0; j < widthB; j+=4)
		{
			half4 c0 = 0;
      		for (int k = 0; k < widthA; k+=4) 
			{
				half4 a0 = convert_half4(vload4(0, tmpA + k));
				half4 b0 = convert_half4(vload4(0, sharedB + k * widthB + j));
           		half4 b1 = convert_half4(vload4(0, sharedB + (k + 1) * widthB + j));
            	half4 b2 = convert_half4(vload4(0, sharedB + (k + 2) * widthB + j));
           		half4 b3 = convert_half4(vload4(0, sharedB + (k + 3) * widthB + j));

				c0 += a0.x * b0;
            	c0 += a0.y * b1;
            	c0 += a0.z * b2;
            	c0 += a0.w * b3;
        		// tmp += tmpA[k] * sharedB[k * widthB + j];
				vstore4(convert_float4(c0), i, C);
			}
      		// C[i * widthB + j] = tmp;
    	}
  	}
}

__kernel void gemm_v5(__global const float *A, __global float *B, __global float *C, unsigned int heightA, unsigned int widthA, unsigned int widthB) {
 	// calculate row * col  per block
	unsigned int row = get_local_id(0);  // sub row of C, max TS
  	unsigned int col = get_local_id(1);  // sub col of C, max TS
	unsigned int globalRow = TS * get_group_id(0) + row;
	unsigned int globalCol = TS * get_group_id(1) + col;

  	float acc = 0;
	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];
	for (int t = 0; t < widthA / TS; t++) 
	{
		unsigned subRow = t * TS + row;
		unsigned subCol = t * TS + col;
		Asub[row][col] = A[globalRow * widthA + subCol];
		Bsub[row][col] = B[subRow * widthB + globalCol];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < TS; k++)
		{	
			acc+= Asub[row][k] * Bsub[k][col];
		}
		/*
			for (int k = 0; k < widthA; k++) 
			{
        		tmp += tmpA[k] * sharedB[k * widthB + j];
      		}
		 */
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[globalRow * widthB + globalCol] = acc;
}