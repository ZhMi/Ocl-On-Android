#define REGISTER_A_SIZE 4
#define SHARED_B_SIZE 4
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
    if (i < heightA) {
    for (int j = 0; j < widthB; j++) {
      tmp = 0;
      for (int k = 0; k < widthA; k++) {
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
	int id = get_local_id(0);
  	int size = get_local_size(0);
	float tmp = 0;

  	float tmpA[REGISTER_A_SIZE];
  	__local float sharedB[SHARED_B_SIZE];
  	
  	if (i < heightA) 
	{
    	for (int k = 0; k < widthA; k++) 
		{
      		tmpA[k] = A[i * widthA + k];
    	}
		
		for (int k = 0; k < widthB; k++) 
		{
      		sharedB[k] = B[k * widthB + k];
    	}
    	barrier(CLK_LOCAL_MEM_FENCE);
    	for (int j = 0; j < widthB; j++)
		{
			tmp = 0;
      		for (int k = 0; k < widthA; k++) 
			{
        		tmp += tmpA[k] * sharedB[k];
      		}
      		C[i * widthB + j] = tmp;
    	}
  	}
}

__kernel void gemm_v4(__global const float *A, __global float *B, __global float *C, int heightA, int widthA, int widthB) {
  
    unsigned int i = get_global_id(0);
  	float tmp = 0;
  	float tmpA[REGISTER_A_SIZE];
  	if (i < heightA)
	{
    	for (int k = 0; k < widthA; k++) 
		{
      		tmpA[k] = A[i * widthA + k];
    	}
    	for (int j = 0; j < widthB; j+= 4) 
		{
			half4 c0 = 0.0f;
      		tmp = 0;
      		for (int k = 0; k < widthA; k += 4) 
			{
				half4 a0 = convert_half4(vload4(0, a + k));
				half4 b0 = convert_half4(vload4(0, b + k * widthB + j));
           		half4 b1 = convert_half4(vload4(0, b + (k + 1) * widthB + j));
            	half4 b2 = convert_half4(vload4(0, b + (k + 2) * widthB + j));
           		half4 b3 = convert_half4(vload4(0, b + (k + 3) * widthB + j));

				c0 += a0.x * b0;
            	c0 += a0.y * b1;
            	c0 += a0.z * b2;
            	c0 += a0.w * b3;
      		}
      		C[i * widthB + j] = c0.x;
			C[i * widthB + j + 1] = c0.y;
			C[i * widthB + j + 2] = c0.z;
			C[i * widthB + j + 3] = c0.w;
   		}
  	}
}