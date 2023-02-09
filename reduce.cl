#define THREAD_NUM_PER_BLOCK 256

__kernel void reduce(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int blockSize = get_local_size(0);
    __local int sdata[THREAD_NUM_PER_BLOCK];   
	sdata[tid] = input[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = 1; s < blockSize; s *= 2)
	{
		if (tid % (2 * s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (tid == 0)
	{
		output[bid] = sdata[0];
	}
}

__kernel void reduce_v2(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int blockSize = get_local_size(0);
    __local int sdata[THREAD_NUM_PER_BLOCK];   
	sdata[tid] = input[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = 1; s < blockSize; s *= 2)
	{
		int idx = 2 * s * tid;
		if (idx < blockSize)
		{
			sdata[idx] += sdata[idx + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (tid == 0)
	{
		output[bid] = sdata[0];
	}
}

__kernel void reduce_v3(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int blockSize = get_local_size(0);
    __local int sdata[THREAD_NUM_PER_BLOCK];   
	sdata[tid] = input[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = blockSize / 2; s > 0; s /= 2)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (tid == 0)
	{
		output[bid] = sdata[0];
	}
}

__kernel void reduce_v4(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int blockSize = get_local_size(0);
	unsigned int group_num = get_num_groups(0);
    __local int sdata[THREAD_NUM_PER_BLOCK];   
	sdata[tid] = input[gid] + input[gid + blockSize * group_num];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = blockSize / 2; s > 0; s /= 2)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (tid == 0)
	{
		output[bid] = sdata[0];
	}
}

__kernel void reduce_v5(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int blockSize = get_local_size(0);
	unsigned int group_num = get_num_groups(0);
    __local volatile int sdata[THREAD_NUM_PER_BLOCK];   
	sdata[tid] = input[gid] + input[gid + blockSize * group_num];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = blockSize / 2; s > 8; s /= 2)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (tid < 8)
	{
		sdata[tid] += sdata[tid + 8];
	}
	if (tid < 4)
	{
		sdata[tid] += sdata[tid + 4];
	}
	if (tid < 2)
	{
		sdata[tid] += sdata[tid + 2];
	}
	if (tid < 1)
	{
		sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0)
	{
		output[bid] = sdata[0];
	}
}

__kernel void reduce_v6(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int block_size = get_local_size(0);
	unsigned int group_num = get_num_groups(0);
    __local volatile int sdata[THREAD_NUM_PER_BLOCK];   
	sdata[tid] = input[gid] + input[gid + block_size * group_num];
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in shared mem
    if(block_size >= 512){
        if(tid < 256){
            sdata[tid] += sdata[tid + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(block_size >= 256){
        if(tid < 128){
            sdata[tid] += sdata[tid + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(block_size >= 128){
        if(tid < 64){
            sdata[tid] += sdata[tid + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
	if(block_size >= 64){
        if(tid < 32){
            sdata[tid] += sdata[tid + 32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
	if(block_size >= 32){
        if(tid < 16){
            sdata[tid] += sdata[tid + 16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
	if (tid < 8)
	{
		sdata[tid] += sdata[tid + 8];	
	}
	if (tid < 4)
	{
		sdata[tid] += sdata[tid + 4];
	}
	if (tid < 2)
	{
		sdata[tid] += sdata[tid + 2];
	}
	if (tid < 1)
	{
		sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0)
	{
		output[bid] = sdata[0];
	}
}

/*
__kernel void reduce_v7(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int block_size = get_local_size(0);
	unsigned int group_num = get_num_groups(0);
    __local int sdata[64];   
	sdata[tid] = input[gid] + input[gid + block_size * group_num];
	barrier(CLK_LOCAL_MEM_FENCE);

	int sum = 0;
	
	__local int warpLevelSums[8];
    const int lane_id = tid % 8;
    const int warp_id = tid / 8;

	sum = sub_group_reduce_add(sum);
	if(lane_id == 0)
	{
		warpLevelSums[warp_id]=sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	sum = (tid < block_size / 8)? warpLevelSums[lane_id]:0;
	if(warp_id == 0)
	{
		sum = sub_group_reduce_add(sum);
	}
	if (tid == 0)
	{
		output[bid] = sum;
	}
}*/