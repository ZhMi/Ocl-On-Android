#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <CL/cl_gl_ext.h>
#include <math.h>
#include <iomanip>

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

#define DEBUG_VERSION_V7

// inputA, inputB, outputC, heightA, widthA, widthB
int isVerify(float *inputA, float *inputB, int heightA, int widthA, int widthB, float *actOutputC)
{
	int res = 1;
	for (int i = 0; i < heightA; i++)
	{
		for (int j = 0; j < widthB; j++)
		{
			float tmp = 0;
			for (int k = 0; k < widthA; k++)
			{
				tmp += inputA[i * widthA + k] * inputB[k * widthB + j];
			}
			// cout << "expect[" << i <<  "]" << "[" << j << "]" << ":" << tmp << "----" << "act:" <<   actOutputC[i * widthB + j] << "\n"; 
			if (tmp != actOutputC[i * widthB + j])
			{
				cout << "expect[" << i <<  "]" << "[" << j << "]" << ":" << tmp << "----" << "act:" <<   actOutputC[i * widthB + j] << "\n"; 
				res = 0;
			}
		}
	}
	return res;
}

const char *opencl_error_to_str(cl_int error)
{
#define CASE_CL_CONSTANT(NAME) \
	case NAME:                 \
		return #NAME;
	// Suppose that no combinations are possible.
	switch (error)
	{
		CASE_CL_CONSTANT(CL_SUCCESS)
		CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
		CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
		CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
		CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
		CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
		CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
		CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
		CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
		CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
		CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
		CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
		CASE_CL_CONSTANT(CL_MAP_FAILURE)
		CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
		CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
		CASE_CL_CONSTANT(CL_INVALID_VALUE)
		CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
		CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
		CASE_CL_CONSTANT(CL_INVALID_DEVICE)
		CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
		CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
		CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
		CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
		CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
		CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
		CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
		CASE_CL_CONSTANT(CL_INVALID_BINARY)
		CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
		CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
		CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL)
		CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
		CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
		CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
		CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
		CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
		CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
		CASE_CL_CONSTANT(CL_INVALID_EVENT)
		CASE_CL_CONSTANT(CL_INVALID_OPERATION)
		CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
		CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
		CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
		CASE_CL_CONSTANT(CL_INVALID_PROPERTY)

	default:
		return "UNKNOWN ERROR CODE";
	}
#undef CASE_CL_CONSTANT
}

void isStatusOK(cl_int status)
{
	cout << "STATUS NUM:" << status << ", " << opencl_error_to_str(status) << endl;
	if (status == CL_SUCCESS)
	{
		return;
	}
}

int main(int argc, char *argv[])
{
	cl_int status;
	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);
	printf("num_platforms: %d\n", (int)num_platforms);

	cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));

	clGetPlatformIDs(num_platforms, platforms, NULL);

	cl_uint num_devices = 0;
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

	cl_device_id *devices;
	devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));

	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

	cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, NULL);
	cl_command_queue cmd_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

	fflush(stdout);
	FILE *fil = fopen("/data/local/tmp/gemm.cl", "r");
	if (fil == NULL)
	{
		printf("Error, could not open the kernel.");
		fclose(fil);
		return -1;
	}
	char *src;
	cl_int error;
	src = (char *)malloc(MAX_SOURCE_SIZE);
	size_t srcsize = fread(src, 1, MAX_SOURCE_SIZE, fil);
	fclose(fil);
	const char *srcptr[] = {src};

	cl_program program = clCreateProgramWithSource(context, 1, srcptr, &srcsize, &error);
	isStatusOK(error);

	status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	isStatusOK(error);
	if (status != CL_SUCCESS)
	{
		char error_buffer[1000];
		clGetProgramBuildInfo(program, *devices, CL_PROGRAM_BUILD_LOG, sizeof(error_buffer), error_buffer, NULL);
		printf("%s\n", error_buffer);
	}

#ifdef DEBUG_VERSION_V0
	int heightA = 368;
	int widthA = 368;
	int heightB = 368;
	int widthB = 368;
	size_t global_work_size[2] = {368, 368};
	size_t local_work_size[1] = {1}; //
#endif

#ifdef DEBUG_VERSION_V1
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[1] = {256};
	size_t local_work_size[1] = {1};
#endif

#ifdef DEBUG_VERSION_V2
	int heightA = 368;
	int widthA = 368;
	int heightB = 368;
	int widthB = 368;
	size_t global_work_size[1] = {368};
	size_t local_work_size[1] = {1};
#endif

#ifdef DEBUG_VERSION_V3
    int heightA = 512;
	int widthA = 512;
	int heightB = 512;
	int widthB = 512;
	size_t global_work_size[1] = {512};
	size_t local_work_size[1] = {size_t(32)};
#endif

#ifdef DEBUG_VERSION_V4
	int heightA = 512;
	int widthA = 512;
	int heightB = 512;
	int widthB = 512;
	size_t global_work_size[1] = {512};
	size_t local_work_size[1] = {size_t(32)};
#endif

#ifdef DEBUG_VERSION_V5
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[2] = {256, 256};
	size_t local_work_size[2] = {8, 8};
#endif

#ifdef DEBUG_VERSION_V6
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[2] = {256, 256};
	size_t local_work_size[2] = {8, 8};
#endif

#ifdef DEBUG_VERSION_V7
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[2] = {256, 256/4};
	size_t local_work_size[2] = {1, 1};
#endif

#ifdef DEBUG_VERSION_V8
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[2] = {256/4, 256/4};
	size_t local_work_size[2] = {1, 1};
#endif

#ifdef DEBUG_VERSION_V9
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[2] = {256/8, 256/8};
	size_t local_work_size[2] = {1, 1};
#endif

#ifdef DEBUG_VERSION_V10
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[2] = {256, 256/4};
	size_t local_work_size[2] = {1, 1};
#endif

#ifdef DEBUG_VERSION_V11
	int heightA = 256;
	int widthA = 256;
	int heightB = 256;
	int widthB = 256;
	size_t global_work_size[2] = {256/2, 256/4};
	size_t local_work_size[2] = {1, 1};
#endif

	int numsA = heightA * widthA;
	float *inputA = new float[numsA];
	for (int i = 0; i < numsA; i++)
	{
		inputA[i] = i + 1;
	}

	int numsB = heightB * widthB;
	float *inputB = new float[numsB];
	for (int i = 0; i < numsB; i++)
	{
		inputB[i] = i + 1;
	}
	int numsC = heightA * widthB;
	float *outputC = new float[numsC];
	for (int i = 0; i < numsC; i++)
	{
		outputC[i] = 0;
	}

// transposeB
#if defined(DEBUG_VERSION_V10) or defined(DEBUG_VERSION_V11)
	float *originB = new float[numsB];
	for (int i = 0; i < numsA; i++)
	{
		inputA[i] = (i + 1) % 10 + 1;
	}
	for (int i = 0; i < numsB; i++)
	{
		originB[i] = (i + 1) % 10 + 1;
	}
	for (int i = 0; i < widthB; i++)
	{
		for (int j = 0; j < heightB; j++)
		{
			inputB[i * heightB + j] = originB[j * widthB + i];
		}
	}
#endif

	cl_mem inputABuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (numsA * sizeof(float)), (void *)inputA, NULL);
	cl_mem inputBBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (numsB * sizeof(float)), (void *)inputB, NULL);
	cl_mem outputCBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, (numsC * sizeof(float)), NULL, NULL);

#ifdef DEBUG_VERSION_V0
	// navie version
	cout << "v0 => navie gemm:" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v0", NULL);
#endif

#ifdef DEBUG_VERSION_V1
	cout << "v1 => cal one row of C in per block:" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v1", NULL);
#endif

#ifdef DEBUG_VERSION_V2
	cout << "v2 => store one row of A in register:" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v2", NULL);
#endif

#ifdef DEBUG_VERSION_V3
	cout << "v3 => store B in local memory:" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v3", NULL);
#endif

#ifdef DEBUG_VERSION_V4
	cout << "v4 => use vector 1*4:" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v4", NULL);
#endif

#ifdef DEBUG_VERSION_V5
	cout << "v5 => cal 4*4 of C per block " << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v5", NULL);
#endif

#ifdef DEBUG_VERSION_V6
	cout << "v6 => pack A[8][8] and B[8][8], cal 1*1 of C per block" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v6", NULL);
#endif

#ifdef DEBUG_VERSION_V7
	cout << "v7 => cal 1X4 of C per block" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v7", NULL);
#endif

#ifdef DEBUG_VERSION_V8
	cout << "v8 => cal 4X4 of C per block" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v8", NULL);
#endif

#ifdef DEBUG_VERSION_V9
	cout << "v9 => cal 8X8 of C per block" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v9", NULL);
#endif

#ifdef DEBUG_VERSION_V10
	cout << "v10 => transposeB, cal 1X4 of C per block" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v10", NULL);
#endif

#ifdef DEBUG_VERSION_V11
	cout << "v10 => transposeB, cal 2X4 of C per block" << endl;
	cl_kernel kernel = clCreateKernel(program, "gemm_v11", NULL);
#endif
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputABuf);
	isStatusOK(status);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputBBuf);
	isStatusOK(status);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputCBuf);
	isStatusOK(status);
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&heightA);
	isStatusOK(status);
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&widthA);
	isStatusOK(status);
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&widthB);
	isStatusOK(status);

	cl_event enentPoint;
#ifdef DEBUG_VERSION_V0	
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V1	
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V2	
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V3
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V4
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V5
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V6
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V7
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V8
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V9
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V9
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V10
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
#ifdef DEBUG_VERSION_V11
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
#endif
	clWaitForEvents(1, &enentPoint); /// wait

	isStatusOK(status);
	status = clEnqueueReadBuffer(cmd_queue, outputCBuf, CL_TRUE, 0, numsC * sizeof(float), outputC, 0, NULL, NULL);
	isStatusOK(status);
	clFinish(cmd_queue);
	// cal kernel execution time
	cl_ulong time_start;
	cl_ulong time_end;

	clGetEventProfilingInfo(enentPoint, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(enentPoint, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	// clReleaseEvent(enentPoint);

	double nanoSeconds = time_end - time_start;
	cout << "OpenCl Exec end time is(op):" << time_end << endl;
	cout << "OpenCl Exec start time is(nanoSeconds):" << time_start << endl;
	cout << "OpenCl Exec time is(nanoSeconds):" << nanoSeconds << endl;
	cout << "OpenCl Exec time is(millisecond):" << (nanoSeconds / 1e6) << endl;

#if defined(DEBUG_VERSION_V10) or defined(DEBUG_VERSION_V11)
	if (isVerify(inputA, originB, heightA, widthA, widthB, outputC) == 1)
		cout << "The result is right!!!" << endl;
	else
		cout << "The result is wrong!!!" << endl;
#else
	if (isVerify(inputA, inputB, heightA, widthA, widthB, outputC) == 1)
		cout << "The result is right!!!" << endl;
	else
		cout << "The result is wrong!!!" << endl;
#endif	
	

	// bandwidth
	double rw_bytes = (heightA * widthA * widthB) * 2 * sizeof(float) + sizeof(float) * (heightA * widthB); // 2 read and 1 write
	cout << "rw_bytes:" << rw_bytes << endl;
	double band_wid = rw_bytes / (nanoSeconds * 1e-9) / 1024 / 1024 / 1024; // gB/s
	cout << "band_wid:" << setprecision(5) <<  band_wid << "GB/s" << endl;

	// gfloops
	double gflops = (heightA * widthA * widthB) * 2;
	cout << "gflops:" << gflops << endl;
	double gflops_per_sec = gflops / (nanoSeconds * 1e-9) * 1e-9; // g/s
	cout << "gflops_per_sec:" << setprecision(5) << gflops_per_sec << " G/s"<< endl;


	status = clReleaseKernel(kernel);
	status = clReleaseProgram(program);
	status = clReleaseMemObject(inputABuf);
	status = clReleaseMemObject(inputBBuf);
	status = clReleaseMemObject(outputCBuf);
	status = clReleaseCommandQueue(cmd_queue);
	status = clReleaseContext(context);

	free(inputA);
	free(inputB);
	free(outputC);
	free(devices);
	return 0;
}