#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

_kernel void reduce(__global const int *input, __global int *output, __local int* sdata) {
	unsigned int tid  = get_local_id(0);
    unsigned int bid  = get_group_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int blockSize = get_local_size(0);
    sdata[tid] = input[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned int s = 1; s < blockSize; s <<= 1) {
		if (tid %(2*s) == 0){
            sdata[tid] += sdata[tid+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

	}
	if(tid == 0) {	
		output[bid] = sdata[0];
	}
}
