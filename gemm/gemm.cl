#define REGISTER_A_SIZE 512
#define SHARED_B_SIZE 512 * 512
#define TS 8
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
				vstore4(convert_float4(c0), 0, C + k * widthB + j);
			}
      		// C[i * widthB + j] = tmp;
    	}
  	}
}

__kernel void gemm_v5(__global const float *A, __global float *B, __global float *C, unsigned int heightA, unsigned int widthA, unsigned int widthB) {
	unsigned int row = get_local_id(0);  // sub row of C, max TS
  	unsigned int col = get_local_id(1);  // sub col of C, max TS
	unsigned int globalRow = TS * get_group_id(0) + row;
	unsigned int globalCol = TS * get_group_id(1) + col;

  	float acc = 0;
	
	for (int k = 0; k < widthA; k++)
	{	
		acc+= A[globalRow * widthA + k] * B[k * widthB + globalCol];
	}
	C[globalRow * widthB + globalCol] = acc;
}

__kernel void gemm_v6(__global const float *A, __global float *B, __global float *C, unsigned int heightA, unsigned int widthA, unsigned int widthB) {
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

// 1X4 C[0, 0], C[0, 1], C[0, 2], C[0, 3]

// C[i, j] 	   += A[i, k] * B[k, j];
// C[i, j + 1] += A[i, k] * B[k, j + 1];
// C[i, j + 2] += A[i, k] * B[k, j + 2];
// C[i, j + 3] += A[i, k] * B[k, j + 3];

// C[i, j] 	   += A[i, k + 1] * B[k + 1, j];
// C[i, j + 1] += A[i, k + 1] * B[k + 1, j + 1];
// C[i, j + 2] += A[i, k + 1] * B[k + 1, j + 2];
// C[i, j + 3] += A[i, k + 1] * B[k + 1, j + 3];
// 
// C[i, j] 	   += A[i, k + 2] * B[k + 2, j];
// C[i, j + 1] += A[i, k + 2] * B[k + 2, j + 1];
// C[i, j + 2] += A[i, k + 2] * B[k + 2, j + 2];
// C[i, j + 3] += A[i, k + 2] * B[k + 2, j + 3];
// 
// C[i, j] 	   += A[i, k + 3] * B[k + 3, j];
// C[i, j + 1] += A[i, k + 3] * B[k + 3, j + 1];
// C[i, j + 2] += A[i, k + 3] * B[k + 3, j + 2];
// C[i, j + 3] += A[i, k + 3] * B[k + 3, j + 3];

/**
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
 */

__kernel void gemm_v7(__global const float *A, __global float *B, __global float *C, int heightA, int widthA, int widthB) 
{
  	int i = get_global_id(0); // max:M
	int j = get_global_id(1) * 4; // max:N/4

	float4 c0 = 0;
    for (int k = 0; k < widthA; k+=4) 
	{
		float4 a0 = vload4(0, A + i * widthA + k);
		float4 b0 = vload4(0, B + k * widthB + j);
		float4 b1 = vload4(0, B + (k + 1) * widthB + j);
		float4 b2 = vload4(0, B + (k + 2) * widthB + j);
		float4 b3 = vload4(0, B + (k + 3) * widthB + j);

		c0 += a0.x * b0;
		c0 += a0.y * b1;
		c0 += a0.z * b2;
		c0 += a0.w * b3;
    }
	vstore4(c0, 0, C + i * widthB + j);
}

// gemm 4*4
__kernel void gemm_v8(__global const float *A, __global float *B, __global float *C, int heightA, int widthA, int widthB) 
{
  	int i = get_global_id(0) * 4; // max:M/4
	int j = get_global_id(1) * 4; // max:N/4
	float4 c0 = 0;
	float4 c1 = 0;
	float4 c2 = 0;
	float4 c3 = 0;
	// 4*4 
    for (int k = 0; k < widthA; k+=4) 
	{
		float4 a0 = vload4(0, A + i * widthA + k);
		float4 a1 = vload4(0, A + (i + 1) * widthA + k);
		float4 a2 = vload4(0, A + (i + 2) * widthA + k);
		float4 a3 = vload4(0, A + (i + 3) * widthA + k);

		float4 b0 = vload4(0, B + k * widthB + j);
		float4 b1 = vload4(0, B + (k + 1) * widthB + j);
		float4 b2 = vload4(0, B + (k + 2) * widthB + j);
		float4 b3 = vload4(0, B + (k + 3) * widthB + j);
		
		c0 += a0.x * b0;
		c0 += a0.y * b1;
		c0 += a0.z * b2;
		c0 += a0.w * b3;
	
		c1 += a1.x * b0;
		c1 += a1.y * b1;
		c1 += a1.z * b2;
		c1 += a1.w * b3;
	
		c2 += a2.x * b0;
		c2 += a2.y * b1;
		c2 += a2.z * b2;
		c2 += a2.w * b3;

		c3 += a3.x * b0;
		c3 += a3.y * b1;
		c3 += a3.z * b2;
		c3 += a3.w * b3;
    }
	vstore4(c0, 0, C + i * widthB + j);
	vstore4(c1, 0, C + (i + 1) * widthB + j);
	vstore4(c2, 0, C + (i + 2) * widthB + j);
	vstore4(c3, 0, C + (i + 3) * widthB + j);
}

__kernel void gemm_v9(__global const float *A, __global float *B, __global float *C, int heightA, int widthA, int widthB) 
{
  	int i = get_global_id(0) * 8; // max:M/8
	int j = get_global_id(1) * 8; // max:N/8
	float8 c0 = 0;
	float8 c1 = 0;
	float8 c2 = 0;
	float8 c3 = 0;
	float8 c4 = 0;
	float8 c5 = 0;
	float8 c6 = 0;
	float8 c7 = 0;
	// 8*8 
    for (int k = 0; k < widthA; k+=8) 
	{
		float8 a0 = vload8(0, A + i * widthA + k);
		float8 a1 = vload8(0, A + (i + 1) * widthA + k);
		float8 a2 = vload8(0, A + (i + 2) * widthA + k);
		float8 a3 = vload8(0, A + (i + 3) * widthA + k);
		float8 a4 = vload8(0, A + (i + 4) * widthA + k);
		float8 a5 = vload8(0, A + (i + 5) * widthA + k);
		float8 a6 = vload8(0, A + (i + 6) * widthA + k);
		float8 a7 = vload8(0, A + (i + 7) * widthA + k);

		float8 b0 = vload8(0, B + k * widthB + j);
		float8 b1 = vload8(0, B + (k + 1) * widthB + j);
		float8 b2 = vload8(0, B + (k + 2) * widthB + j);
		float8 b3 = vload8(0, B + (k + 3) * widthB + j);
		float8 b4 = vload8(0, B + (k + 4) * widthB + j);
		float8 b5 = vload8(0, B + (k + 5) * widthB + j);
		float8 b6 = vload8(0, B + (k + 6) * widthB + j);
		float8 b7 = vload8(0, B + (k + 7) * widthB + j);
		
		c0 += a0.s0 * b0;
		c0 += a0.s1 * b1;
		c0 += a0.s2 * b2;
		c0 += a0.s3 * b3;
		c0 += a0.s4 * b4;
		c0 += a0.s5 * b5;
		c0 += a0.s6 * b6;
		c0 += a0.s7 * b7;

		c1 += a1.s0 * b0;
		c1 += a1.s1 * b1;
		c1 += a1.s2 * b2;
		c1 += a1.s3 * b3;
		c1 += a1.s4 * b4;
		c1 += a1.s5 * b5;
		c1 += a1.s6 * b6;
		c1 += a1.s7 * b7;
	
		c2 += a2.s0 * b0;
		c2 += a2.s1 * b1;
		c2 += a2.s2 * b2;
		c2 += a2.s3 * b3;
		c2 += a2.s4 * b4;
		c2 += a2.s5 * b5;
		c2 += a2.s6 * b6;
		c2 += a2.s7 * b7;

		c3 += a3.s0 * b0;
		c3 += a3.s1 * b1;
		c3 += a3.s2 * b2;
		c3 += a3.s3 * b3;
		c3 += a3.s4 * b4;
		c3 += a3.s5 * b5;
		c3 += a3.s6 * b6;
		c3 += a3.s7 * b7;

		c4 += a4.s0 * b0;
		c4 += a4.s1 * b1;
		c4 += a4.s2 * b2;
		c4 += a4.s3 * b3;
		c4 += a4.s4 * b4;
		c4 += a4.s5 * b5;
		c4 += a4.s6 * b6;
		c4 += a4.s7 * b7;

		c5 += a5.s0 * b0;
		c5 += a5.s1 * b1;
		c5 += a5.s2 * b2;
		c5 += a5.s3 * b3;
		c5 += a5.s4 * b4;
		c5 += a5.s5 * b5;
		c5 += a5.s6 * b6;
		c5 += a5.s7 * b7;

		c6 += a6.s0 * b0;
		c6 += a6.s1 * b1;
		c6 += a6.s2 * b2;
		c6 += a6.s3 * b3;
		c6 += a6.s4 * b4;
		c6 += a6.s5 * b5;
		c6 += a6.s6 * b6;
		c6 += a6.s7 * b7;

		c7 += a7.s0 * b0;
		c7 += a7.s1 * b1;
		c7 += a7.s2 * b2;
		c7 += a7.s3 * b3;
		c7 += a7.s4 * b4;
		c7 += a7.s5 * b5;
		c7 += a7.s6 * b6;
		c7 += a7.s7 * b7;
    }
	vstore8(c0, 0, C + i * widthB + j);
	vstore8(c1, 0, C + (i + 1) * widthB + j);
	vstore8(c2, 0, C + (i + 2) * widthB + j);
	vstore8(c3, 0, C + (i + 3) * widthB + j);
	vstore8(c4, 0, C + (i + 4) * widthB + j);
	vstore8(c5, 0, C + (i + 5) * widthB + j);
	vstore8(c6, 0, C + (i + 6) * widthB + j);
	vstore8(c7, 0, C + (i + 7) * widthB + j);
}

// loop k, step = 4, cal 1X4 of c
// __kernel void gemm_v10(__global const float *A, __global float *tB, __global float *C, int heightA, int widthA, int widthB) 
// {
//   	int i = get_global_id(0); // max:M
// 	int j = get_global_id(1) * 4; // max:N/4

// 	float4 c0 = 0;
// 	float4 c1 = 0;
// 	float4 c2 = 0;
// 	float4 c3 = 0;
	
//     float4 c_vec4;
//     for (int k = 0; k < widthA; k+=4) 
// 	{
// 		float4 a0 = vload4(0, A +  i * widthA + k);
// 		float4 b0 = vload4(0, tB + j * widthA + k);
// 		float4 b1 = vload4(0, tB + (j + 1) * widthA + k);
// 		float4 b2 = vload4(0, tB + (j + 2) * widthA + k);
// 		float4 b3 = vload4(0, tB + (j + 3) * widthA + k);

// 		c0 += a0 * b0;
// 		c1 += a0 * b1;
// 		c2 += a0 * b2;
// 		c3 += a0 * b3;
//     }
// 	c_vec4.x = c0.x + c0.y + c0.z + c0.w;
// 	c_vec4.y = c1.x + c1.y + c1.z + c1.w;
// 	c_vec4.z = c2.x + c2.y + c2.z + c2.w;
// 	c_vec4.w = c3.x + c3.y + c3.z + c3.w;

// 	/*C[i*widthB + j] = c00;
// 	C[i*widthB + j + 1] = c01;
// 	C[i*widthB + j + 2] = c02;
// 	C[i*widthB + j + 3] = c03;*/
// 	vstore4(c_vec4, 0, C + i * widthB + j);
// }

// loop k, step = 8, cal 1X4 of c
__kernel void gemm_v10(__global const float *A, __global float *tB, __global float *C, int heightA, int widthA, int widthB) 
{
  	int i = get_global_id(0); // max:M
	int j = get_global_id(1) * 4; // max:N/4

	float8 c0 = 0;
	float8 c1 = 0;
	float8 c2 = 0;
	float8 c3 = 0;

	// float4 c2 = 0;
	// float4 c3 = 0;
    float4 c_vec4;
    for (int k = 0; k < widthA; k+=8) 
	{
		float8 a0 = vload8(0, A +  i * widthA + k);
		float8 b0 = vload8(0, tB + j * widthA + k);
		float8 b1 = vload8(0, tB + (j + 1) * widthA + k);
		float8 b2 = vload8(0, tB + (j + 2) * widthA + k);
		float8 b3 = vload8(0, tB + (j + 3) * widthA + k);

		c0 += a0 * b0;
		c1 += a0 * b1;
		c2 += a0 * b2;
		c3 += a0 * b3;
    }
	c_vec4.s0 = c0.s0 + c0.s1 + c0.s2 + c0.s3 + c0.s4 + c0.s5 + c0.s6 + c0.s7;
	c_vec4.s1 = c1.s0 + c1.s1 + c1.s2 + c1.s3 + c1.s4 + c1.s5 + c1.s6 + c1.s7;
	c_vec4.s2 = c2.s0 + c2.s1 + c2.s2 + c2.s3 + c2.s4 + c2.s5 + c2.s6 + c2.s7;
	c_vec4.s3 = c3.s0 + c3.s1 + c3.s2 + c3.s3 + c3.s4 + c3.s5 + c3.s6 + c3.s7;

	// float c02 = c2.x + c2.y + c2.z + c2.w;
	// float c03 = c3.x + c3.y + c3.z + c3.w;

	// C[i*widthB + j] = c00;
	// C[i*widthB + j + 1] = c01;
	// C[i*widthB + j + 2] = c02;
	// C[i*widthB + j + 3] = c03;
	vstore4(c_vec4, 0, C + i * widthB + j);
}

// loop k, step = 8, cal 2X4 of c
__kernel void gemm_v11(__global const float *A, __global float *tB, __global float *C, int heightA, int widthA, int widthB) 
{
  	int i = get_global_id(0) * 2; // max:M
	int j = get_global_id(1) * 4; // max:N/4

	float8 c0 = 0;
	float8 c1 = 0;
	float8 c2 = 0;
	float8 c3 = 0;
	
	float8 c4 = 0;
	float8 c5 = 0;
	float8 c6 = 0;
	float8 c7 = 0;

	// float4 c2 = 0;
	// float4 c3 = 0;
    float4 c_vec4_0;
	float4 c_vec4_1;

    for (int k = 0; k < widthA; k+=8) 
	{
		float8 a0 = vload8(0, A +  i * widthA + k);
		float8 a1 = vload8(0, A +  (i + 1) * widthA + k);

		float8 b0 = vload8(0, tB + j * widthA + k);
		float8 b1 = vload8(0, tB + (j + 1) * widthA + k);
		float8 b2 = vload8(0, tB + (j + 2) * widthA + k);
		float8 b3 = vload8(0, tB + (j + 3) * widthA + k);

		c0 += a0 * b0;
		c1 += a0 * b1;
		c2 += a0 * b2;
		c3 += a0 * b3;

		c4 += a1 * b0;
		c5 += a1 * b1;
		c6 += a1 * b2;
		c7 += a1 * b3;
    }
	c_vec4_0.s0 = c0.s0 + c0.s1 + c0.s2 + c0.s3 + c0.s4 + c0.s5 + c0.s6 + c0.s7;
	c_vec4_0.s1 = c1.s0 + c1.s1 + c1.s2 + c1.s3 + c1.s4 + c1.s5 + c1.s6 + c1.s7;
	c_vec4_0.s2 = c2.s0 + c2.s1 + c2.s2 + c2.s3 + c2.s4 + c2.s5 + c2.s6 + c2.s7;
	c_vec4_0.s3 = c3.s0 + c3.s1 + c3.s2 + c3.s3 + c3.s4 + c3.s5 + c3.s6 + c3.s7;

	c_vec4_1.s0 = c4.s0 + c4.s1 + c4.s2 + c4.s3 + c4.s4 + c4.s5 + c4.s6 + c4.s7;
	c_vec4_1.s1 = c5.s0 + c5.s1 + c5.s2 + c5.s3 + c5.s4 + c5.s5 + c5.s6 + c5.s7;
	c_vec4_1.s2 = c6.s0 + c6.s1 + c6.s2 + c6.s3 + c6.s4 + c6.s5 + c6.s6 + c6.s7;
	c_vec4_1.s3 = c7.s0 + c7.s1 + c7.s2 + c7.s3 + c7.s4 + c7.s5 + c7.s6 + c7.s7;

	// float c02 = c2.x + c2.y + c2.z + c2.w;
	// float c03 = c3.x + c3.y + c3.z + c3.w;

	// C[i*widthB + j] = c00;
	// C[i*widthB + j + 1] = c01;
	// C[i*widthB + j + 2] = c02;
	// C[i*widthB + j + 3] = c03;
	vstore4(c_vec4_0, 0, C + i * widthB + j);
	vstore4(c_vec4_1, 0, C + (i + 1) * widthB + j);
}