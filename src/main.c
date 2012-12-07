#include <stdio.h>
#include "opcl.h"
#include <time.h>

int iniciar_opencl() {
  unsigned int num_platforms, num_devices;

  if ( ( num_platforms =  opencl_create_platform(2)  ) <= 0 ) {
      printf("Erro na criação da camada de plataforma!\n");
      return -1; 
  }
  
  if ( ( num_devices = opencl_get_devices_id(CL_DEVICE_TYPE_CPU) ) <= 0 ) {
    printf("Erro na busca por devices!\n");
    return -1;
  }
  
  if ( opencl_create_context() <= 0 ) {
    printf("Erro na criação do contexto!\n");
    return -1;
  }

  if ( opencl_create_queue() <= 0 ) {
    printf("Erro na criação da fila de comandos\n");
    return -1;
  }

  if ( opencl_create_program("shift.cl") <= 0 ) {
    printf("Erro na criação do programa\n");
    return -1;
  }

  if ( opencl_create_kernel("shift") <= 0 ) {
    printf("Erro na criação do kernel\n");
    return -1;
  }

  return 1;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Uso:\n\t ep4 <arquivo-com-matriz>\n");
    return 0;
  }
  if ( iniciar_opencl() == 1 ) {
    opencl_run_kernel(argv[1]);
  }
  return 0;
}
