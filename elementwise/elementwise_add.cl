__kernel void elementwise_add_v0(__global const float *a,  __global const float *b, __global float *c)
{
    unsigned int i = get_global_id(0);
    c[i] = a[i] + b[i];
} 


__kernel void elementwise_add_v1(__global const float *a,  __global const float *b, __global float *c)
{
    unsigned int i = get_global_id(0);
    float4 a0 = vload4(i, a);
    float4 b0 = vload4(i, b);
    float4 c0 = a0 + b0;
    vstore4(c0, i, c);
}

__kernel void elementwise_add_v2(__global const float *a,  __global const float *b, __global float *c)
{
    unsigned int i = get_global_id(0);
    float8 a0 = vload8(i, a);
    float8 b0 = vload8(i, b);
    float8 c0 = a0 + b0;
    vstore8(c0, i, c);
}