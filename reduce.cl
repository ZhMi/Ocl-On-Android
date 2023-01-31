__kernel void reduce(__global const int *input, __global int *output)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int blockSize = get_local_size(0);
    __local int sdata[64];   
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
    __local int sdata[64];   
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
    __local int sdata[64];   
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
    __local int sdata[64];   
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
    __local volatile int sdata[64];   
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