#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

int isVerify(int NUM, int groupNUM, int *res)
{
	int sum1 = (NUM + 1) * NUM / 2;
	int sum2 = 0;
	for (int i = 0; i < groupNUM; i++)
		sum2 += res[i];
	if (sum1 == sum2)
		return 0;
	return -1;
}

void isStatusOK(cl_int status)
{
	cout << "STATUS NUM:" << status << endl;
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

	cl_command_queue cmd_queue = clCreateCommandQueue(context, devices[0], 0, NULL);

	fflush(stdout);
	FILE *fil = fopen("/data/local/tmp/reduce.cl", "r");
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
	if (error != CL_SUCCESS)
	{
		printf("\n Error number %d", error);
	}

	status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

	int NUM = 640;												// 6400*4
	size_t global_work_size[1] = {640};							/// x
	size_t local_work_size[1] = {64};							/// 256 PE
	size_t groupNUM = global_work_size[0] / local_work_size[0]; // 400 个工作组
	cout << "groupNUM:" << groupNUM << endl;
	int *input = new int[NUM];
	for (int i = 0; i < NUM; i++)
		input[i] = i + 1;
	int *output = new int[groupNUM]; // 400

	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (NUM) * sizeof(int), (void *)input, NULL);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, (groupNUM * sizeof(int)), NULL, NULL);

	cl_kernel kernel = clCreateKernel(program, "reduce", NULL);

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
	isStatusOK(status);

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
	isStatusOK(status);

	cl_event enentPoint;
	status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &enentPoint);
	clWaitForEvents(1, &enentPoint); /// wait
	clReleaseEvent(enentPoint);
	isStatusOK(status);

	status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_TRUE, 0, groupNUM * sizeof(int), output, 0, NULL, NULL);
	isStatusOK(status);

	for (size_t j = 0; j < groupNUM; j++)
	{
		cout << "output[" << j << "]:" << output[j] << endl;
	}

	if (isVerify(NUM, groupNUM, output) == 0)
		cout << "The result is right!!!" << endl;
	else
		cout << "The result is wrong!!!" << endl;

	status = clReleaseKernel(kernel);		  //*Release kernel.
	status = clReleaseProgram(program);		  // Release the program object.
	status = clReleaseMemObject(inputBuffer); // Release mem object.
	status = clReleaseMemObject(outputBuffer);
	status = clReleaseCommandQueue(cmd_queue); // Release  Command queue.
	status = clReleaseContext(context);		   // Release context.

	free(input);
	free(output);
	free(devices);
	return 0;
}