#include <stdio.h>
#include "opcl.h"

#define MAXSTR 128
#define MATRIXSIZE 4096
#define NANO 1e-6f

/* Objetos do Open CL */
cl_platform_id platform;
cl_context context;
cl_device_id* devices;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;
cl_event event;
cl_mem opclMatrixA, opclMatrixB, opclMatrixC;
cl_ulong start, finish;
double total = 0;
size_t sizeOfMatrix = sizeof(float)*MATRIXSIZE*MATRIXSIZE;


/* Informações sobre os devices */
unsigned int devices_found;
unsigned int device_used = 0;

unsigned int opencl_create_platform(unsigned int num_platforms) {
  char name[MAXSTR];
  unsigned int num_platforms_found;
  
  if ( clGetPlatformIDs( num_platforms, &platform, &num_platforms_found ) == CL_SUCCESS ) {
    /* As duas linhas abaixo são usadas para teste.
    clGetPlatformInfo( platform, CL_PLATFORM_NAME, MAXSTR, &name, NULL );
    printf("Nome da plataforma %s\n",name); */ 
    return num_platforms_found;
  }
  else return -1;
}

unsigned int opencl_get_devices_id(cl_device_type device_type) {
  unsigned int vendor_id;
  
  /* Achando o número de devices na máquina */
  clGetDeviceIDs(platform, device_type, 0, NULL, &devices_found);
  devices = malloc(devices_found*(sizeof(cl_device_id)));
  
  if ( clGetDeviceIDs(platform, device_type, devices_found, devices, NULL) 
      == CL_SUCCESS ) 
  {
    /* As duas linhas abaixo são usadas para teste. 
    clGetDeviceInfo( devices, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(unsigned int), &vendor_id, NULL );
    printf("Vendor ID do Device %d\n",vendor_id); */
    return devices_found;
  }
  else return -1;
}

int opencl_create_context() {
  if ( ( context = clCreateContext( 0, 1, devices, NULL, NULL, NULL ) ) != NULL ) {
      return 1;
  }
  else return -1;
}

int opencl_create_queue() {
  if ( ( queue = clCreateCommandQueue(context, devices[device_used], CL_QUEUE_PROFILING_ENABLE, NULL ) ) != NULL ) {
    return 1;
  }
  else return -1;
}

/* Funções auxiliares para a criação do program */
char* loadProgramFromSource(char* program_path, int *size) {
  char* program_string;
  FILE* prog;

  prog = fopen(program_path, "r");
  fseek(prog, 0, SEEK_END);
  *size = ftell(prog);
  fseek(prog, 0, SEEK_SET);

  program_string = malloc((*size+1)*sizeof(char));
  *size = fread(program_string, 1, *size, prog);
  fclose(prog);
  program_string[*size] = '\0';

  return program_string;
}

int buildProgram() {
  int err;
  char *build_log, **program_binary;
  size_t ret_val_size, *binary_size, count;
  FILE *pFile;

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if ( err != CL_SUCCESS ) {
    clGetProgramBuildInfo(program, devices[device_used], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

    build_log = malloc((ret_val_size+1)*sizeof(char));
    clGetProgramBuildInfo(program, devices[device_used], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    build_log[ret_val_size] = '\0';

    printf("BUILD LOG: \n %s", build_log);
    printf("program built\n");
    return -1;
  }
  else {
    pFile = fopen("opcl.ptx", "w");
    if (pFile == NULL) {
      printf("Erro na criação do .ptx\n");
      exit(-1);
    }

    binary_size = malloc(devices_found*sizeof(size_t));
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, devices_found*sizeof(size_t), (void*)binary_size, NULL); 
    program_binary = malloc(devices_found*sizeof(char*));
    for (count = 0; count < devices_found; count++) 
      program_binary[count] = malloc(binary_size[0]*sizeof(char));
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, count*binary_size[0]*sizeof(char), program_binary, NULL);
      
    fputs(program_binary[0], pFile);

    return 1;
  }
}
/* Fim das funções auxiliares para a criação do program */

void profile_event (cl_event* profiler) {
  clWaitForEvents(1, &event);
  if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, (size_t)sizeof(cl_ulong), &start, NULL) != CL_SUCCESS) printf("Erro!\n");
  if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, (size_t)sizeof(cl_ulong), &finish, NULL) != CL_SUCCESS) printf("Erro!\n");

  total += (double)(finish-start);
}


int opencl_create_program(char* program_path) {
  char* program_source;
  int size;
  size_t prog_size;
  cl_int err;
  
  program_source = loadProgramFromSource(program_path, &size);
  prog_size = (size_t)size;
  program = clCreateProgramWithSource(context, 1, (const char**)&program_source, &prog_size, &err);
  if ( err != CL_SUCCESS ) printf("Erro = %d\n",err);
  return buildProgram();
}

int opencl_create_kernel(char* kernel_name) {
  cl_int err;
  kernel = clCreateKernel( program, (const char*) kernel_name, &err);
  if ( err == CL_SUCCESS ) return 1;
  else return -1;
}

void prepare_kernel() {
  float* MatrixA;
  int i, j, sizeC, sizeR;
  cl_int error;
  cl_mem rowSize, columnSize;

  sizeC = sizeR = MATRIXSIZE;
  MatrixA = malloc(sizeOfMatrix);

  for ( i = 0; i < MATRIXSIZE; i++ ) {
    for ( j = 0; j < MATRIXSIZE; j++ ) {
      MatrixA[i*MATRIXSIZE+j] = i+j;
    }
  }

  /* Criação dos buffers que o OpenCL vai usar. */
  opclMatrixA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeOfMatrix, NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  opclMatrixB = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeOfMatrix, NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  rowSize = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  columnSize = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");

  clEnqueueWriteBuffer(queue, opclMatrixA, CL_TRUE, 0, sizeOfMatrix, MatrixA, 0, NULL, &event);
  profile_event(&event);
  clEnqueueWriteBuffer(queue, rowSize, CL_TRUE, 0, sizeof(int), &sizeR, 0, NULL, &event);
  profile_event(&event);
  clEnqueueWriteBuffer(queue, columnSize, CL_TRUE, 0, sizeof(int), &sizeC, 0, NULL, &event);
  profile_event(&event);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&opclMatrixA);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&opclMatrixB);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&rowSize);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&columnSize);

  clFinish(queue);
  free(MatrixA);
}

int opencl_run_kernel() {
  size_t work_dim[2] = { MATRIXSIZE, MATRIXSIZE };
  size_t local_dim[2] = { 16, 16 };
  float *MatrixB;
  int i, j;

  MatrixB = malloc(sizeOfMatrix);

  prepare_kernel();
  clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_dim, local_dim, 0, NULL, &event);
  profile_event(&event);
  clReleaseEvent(event);
  clFinish(queue);

  if( clEnqueueReadBuffer(queue, opclMatrixB, CL_TRUE, 0, sizeOfMatrix, MatrixB, 0, NULL, &event) 
      != CL_SUCCESS ) printf("ERRROROOO\n");
  profile_event(&event);
  clReleaseEvent(event);

  /*for( i = 0; i < MATRIXSIZE; i++ ) {
    for( j = 0; j< MATRIXSIZE; j++ ) {
      printf("B[%d][%d] = %f\n", i, j, MatrixB[i*MATRIXSIZE+j]);
    }
  }*/

  printf("%lf\n", total*NANO);
  free(MatrixB);
  return 1;
}

