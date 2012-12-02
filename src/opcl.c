#include <stdio.h>
#include <string.h>
#include "opcl.h"

#define MAXSTR 128
#define MATRIXSIZE 4096
#define NANO 1e-6f

/* Objetos do Open CL */
cl_platform_id* platform;
cl_platform_id platform_of_choice;
cl_context context;
cl_device_id* devices;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;
cl_event event;
cl_mem opclMatrizInicial, opclMatriz, shifts, axisShift;
cl_ulong start, finish;
double total = 0;
size_t sizeOfMatrix;
int sizeR, sizeC, axis = 1;

/* Informações sobre os devices */
unsigned int devices_found;
unsigned int device_used = 0;

unsigned int opencl_create_platform(unsigned int num_platforms) {
  char name[MAXSTR];
  unsigned int num_platforms_found;
  int i;

  if ( clGetPlatformIDs( num_platforms, NULL, &num_platforms_found ) == CL_SUCCESS ) {
    platform = malloc(num_platforms_found*sizeof(cl_platform_id));
    clGetPlatformIDs( num_platforms, platform, NULL );
    /*As duas linhas abaixo são usadas para teste.*/
    for ( i = 0; i < num_platforms; i++) {
      clGetPlatformInfo( platform[i], CL_PLATFORM_NAME, MAXSTR, &name, NULL );
      if (strcmp(name, "AMD Accelerated Parallel Processing") == 0) {
        platform_of_choice = platform[i];
      }
    }
    return num_platforms_found;
  }
  else return -1;
}

unsigned int opencl_get_devices_id(cl_device_type device_type) {
  char name[MAXSTR];
  
  /* Achando o número de devices na máquina */
  clGetDeviceIDs(platform_of_choice, device_type, 0, NULL, &devices_found);
  devices = malloc(devices_found*(sizeof(cl_device_id)));
  
  if ( clGetDeviceIDs(platform_of_choice, device_type, devices_found, devices, NULL) 
      == CL_SUCCESS ) 
  {
    /* As duas linhas abaixo são usadas para teste. */
    /*clGetDeviceInfo( devices[device_used], CL_DEVICE_VENDOR, sizeof(unsigned int), &name, NULL );
    printf("Vendor do Device %s\n",name);*/
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
  return 1;
}
/* Fim das funções auxiliares para a criação do program */

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

void printMatrix(float* Matrix, char* message) {
  int i, j;
  puts(message);
  for( i = 0; i < sizeR; i++ ) {
    for( j = 0; j< sizeC; j++ ) {
      printf("[%.2f]\t", Matrix[i*sizeC+j]);
    }
    printf("\n");
  }

}

float* loadMatrix(char* fileName) {
  FILE *pFile;
  float* MatrixA;
  char line[256];
  int i, j;

  pFile = fopen(fileName, "r");
  if (pFile == NULL) {
    printf("Erro na leitura do arquivo com a matriz\n");
    exit(-1);
  }

  fgets(line, 256, pFile);
  sizeR = atoi(line);
  fgets(line, 256, pFile);
  sizeC = atoi(line);
  printf("Linha = %d\nColuna = %d\n", sizeR, sizeC);
  sizeOfMatrix = sizeof(float)*sizeR*sizeC;
  MatrixA = malloc(sizeOfMatrix);
  for ( i = 0; i < sizeR; i++ ) {
    for ( j = 0; j < sizeC; j++ ) {
      fgets(line, 256, pFile);
      MatrixA[i*sizeC+j] = atof(line);
    }
  }

  printMatrix(MatrixA, "Matriz Inicial:");

  return MatrixA;
}

void prepare_kernel(char* fileName) {
  float* MatrixA;
  int i, j, numberShift;
  cl_int error;
  cl_mem rowSize, columnSize;

  MatrixA = loadMatrix(fileName);
  numberShift = 1;
  /* Criação dos buffers que o OpenCL vai usar. */
  opclMatrizInicial = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeOfMatrix, NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  opclMatriz = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeOfMatrix, NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  rowSize = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  columnSize = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  shifts = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");
  axisShift = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error);
  if (error != CL_SUCCESS) printf("Erro na memoria\n");

  clEnqueueWriteBuffer(queue, opclMatrizInicial, CL_TRUE, 0, sizeOfMatrix, MatrixA, 0, NULL, &event);
  clEnqueueWriteBuffer(queue, rowSize, CL_TRUE, 0, sizeof(int), &sizeR, 0, NULL, &event);
  clEnqueueWriteBuffer(queue, columnSize, CL_TRUE, 0, sizeof(int), &sizeC, 0, NULL, &event);
  clEnqueueWriteBuffer(queue, shifts, CL_TRUE, 0, sizeof(int), &numberShift, 0, NULL, &event);
  clEnqueueWriteBuffer(queue, axisShift, CL_TRUE, 0, sizeof(int), &axis, 0, NULL, &event);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&opclMatrizInicial);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&opclMatriz);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&rowSize);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&columnSize);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&shifts);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&axisShift);

  clFinish(queue);
  free(MatrixA);
}

int opencl_run_kernel(char* fileName) {
  size_t work_dim[2];
  float *Matriz;
  int i, j;

  prepare_kernel(fileName);
  
  Matriz = malloc(sizeOfMatrix);
  work_dim[0] = sizeR;
  work_dim[1] = sizeC;
  clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_dim, NULL, 0, NULL, &event);
  clReleaseEvent(event);
  clFinish(queue);

  if( clEnqueueReadBuffer(queue, opclMatriz, CL_TRUE, 0, sizeOfMatrix, Matriz, 0, NULL, &event) 
      != CL_SUCCESS ) printf("ERRROROOO\n");
  clReleaseEvent(event);

  printMatrix(Matriz, "Matriz Shift-Row:");

  free(Matriz);
  return 1;
}

