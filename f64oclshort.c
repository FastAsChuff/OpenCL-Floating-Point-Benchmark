#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

//gcc f64oclshort.c -o f64oclshort.bin -lOpenCL -O3 -march=native -Wall

#define F64TEST2_PIXELDIM 122880
#define KERNEL_COUNT 1
#define MAX_PLATFORMS 10
#define MAX_DEVICES 25
#define NAMES_LENGTH 255
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

char* oclkernel_names[] = {"getfgcount"};
char* oclkernels[] = {"\
__kernel void getfgcount(__global unsigned long* counts) { \
  __private FLOATTYPE zr, zi, zi0, temp;\
  __private unsigned long x,y,i,count = 0;\
  y = get_global_id(0) + DIM0*(get_global_id(1) + DIM1*get_global_id(2));\
  zi0 = IMAGSTART + (IMAGEND - IMAGSTART)*y/(PIXELDIM-1);\
  for (x=0; x<PIXELDIM; x++) {\
    zr = REALSTART + (REALEND - REALSTART)*x/(PIXELDIM-1);\
    zi = zi0;\
    for (i=0; i<MAXITERATIONS; i++) {\
      temp = zr*zr - zi*zi + REALCONST;\
      zi = 2*zr*zi + IMAGCONST;\
      zr = temp;\
      temp = zr*zr - zi*zi + REALCONST;\
      zi = 2*zr*zi + IMAGCONST;\
      zr = temp;\
      temp = zr*zr - zi*zi + REALCONST;\
      zi = 2*zr*zi + IMAGCONST;\
      zr = temp;\
      temp = zr*zr - zi*zi + REALCONST;\
      zi = 2*zr*zi + IMAGCONST;\
      zr = temp;\
    }\
    count += ((zi*zi + zr*zr) < THRESHOLD);\
  }\
  counts[y] = count;\
    }"};

void printf_cl_error(cl_int res) {
  if (res == CL_INVALID_MEM_OBJECT) printf("CL_INVALID_MEM_OBJECT\n");
  if (res == CL_INVALID_SAMPLER) printf("CL_INVALID_SAMPLER\n");
  if (res == CL_INVALID_KERNEL) printf("CL_INVALID_KERNEL\n");
  if (res == CL_INVALID_ARG_INDEX) printf("CL_INVALID_ARG_INDEX\n");
  if (res == CL_INVALID_ARG_VALUE) printf("CL_INVALID_ARG_VALUE\n");
  if (res == CL_INVALID_ARG_SIZE) printf("CL_INVALID_ARG_SIZE\n");
  if (res == CL_INVALID_COMMAND_QUEUE) printf("CL_INVALID_COMMAND_QUEUE\n");
  if (res == CL_INVALID_CONTEXT) printf("CL_INVALID_CONTEXT\n");
  if (res == CL_INVALID_MEM_OBJECT) printf("CL_INVALID_MEM_OBJECT\n");
  if (res == CL_INVALID_VALUE) printf("CL_INVALID_VALUE\n");
  if (res == CL_INVALID_EVENT_WAIT_LIST) printf("CL_INVALID_EVENT_WAIT_LIST\n");
  if (res == CL_MEM_OBJECT_ALLOCATION_FAILURE) printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
  if (res == CL_OUT_OF_HOST_MEMORY) printf("CL_OUT_OF_HOST_MEMORY\n");
  if (res == CL_INVALID_PROGRAM_EXECUTABLE) printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
  if (res == CL_INVALID_KERNEL_ARGS) printf("CL_INVALID_KERNEL_ARGS\n");
  if (res == CL_INVALID_WORK_DIMENSION) printf("CL_INVALID_WORK_DIMENSION\n");
  if (res == CL_INVALID_GLOBAL_WORK_SIZE) printf("CL_INVALID_GLOBAL_WORK_SIZE\n");
  if (res == CL_INVALID_WORK_GROUP_SIZE) printf("CL_INVALID_WORK_GROUP_SIZE\n");
  if (res == CL_INVALID_WORK_ITEM_SIZE) printf("CL_INVALID_WORK_ITEM_SIZE\n");
  if (res == CL_INVALID_GLOBAL_OFFSET) printf("CL_INVALID_GLOBAL_OFFSET\n");
  if (res == CL_OUT_OF_RESOURCES) printf("CL_OUT_OF_RESOURCES\n");
  if (res == CL_INVALID_OPERATION) printf("CL_INVALID_OPERATION\n");
  if (res == CL_BUILD_PROGRAM_FAILURE) printf("CL_BUILD_PROGRAM_FAILURE\n");
  if (res == CL_COMPILER_NOT_AVAILABLE) printf("CL_COMPILER_NOT_AVAILABLE\n");
  if (res == CL_INVALID_BUILD_OPTIONS) printf("CL_INVALID_BUILD_OPTIONS\n");
  if (res == CL_INVALID_BINARY) printf("CL_INVALID_BUILD_OPTIONS\n");
  if (res == CL_INVALID_DEVICE) printf("CL_INVALID_DEVICE\n");
  if (res != CL_SUCCESS) {
    if (res == -1) {
      printf("OpenCL Error Code %i\n", res);
    } else {
      printf("OpenCL Failed With Error Code %i\n", res);
      exit(1);
    }
  }
}

typedef struct {
               time_t      tv_sec;     /* seconds */
               suseconds_t tv_usec;    /* microseconds */
           } timeval_t;

int64_t tstampmsec() {
  timeval_t timeval;
  gettimeofday((struct timeval * restrict)&timeval, 0);
  return timeval.tv_sec*1000LL + timeval.tv_usec/1000;
} 	

int main(int argc, char* argv[]) {
  printf("This program counts the number of foreground pixels in a large Julia set fractal image, without actually creating the image. It is to benchmark floating point arithmetic performance of an OpenCL device.\nUsage:- %s platformno deviceno floattype\nfloattype = h for half, f for float, blank for double\nAuthor: Simon Goater August 2024\n\n", argv[0]);
  unsigned int platformno = 0; // Choose Default Platform No.
  unsigned int deviceno = 0;  // Choose Default Device No.
  int floattypeno = 2; // half=0 float=1 double=2
  char floattypesuffixes[] = {'h', 'f', ' '};
  char *floattypes[] = {"half", "float", "double"};
  char *fpextensions[] = {"cl_khr_fp16", " ", "cl_khr_fp64"};
  int argv1, argv2;
  if (argc > 1) { 
    argv1 = atoi(argv[1]);
    if ((argv1 >= 0) && (argv1 < MAX_PLATFORMS)) platformno = argv1;
  }
  if (argc > 2) { 
    argv2 = atoi(argv[2]);
    if ((argv2 >= 0) && (argv2 < MAX_DEVICES)) deviceno = argv2;
  }
  if (argc > 3) { 
    if (argv[3][0] == 'h') floattypeno = 0;
    if (argv[3][0] == 'f') floattypeno = 1;
  }
  int64_t progstart, progend;
  int32_t i,j;
  uint64_t maxiterations = 50;
  uint64_t dim[3], dimlocal[3];
  char text[NAMES_LENGTH];
  dim[0] = 192;
  dim[2] = 64;
  dim[1] = 1 + (F64TEST2_PIXELDIM/(dim[2]*dim[0]));
  dimlocal[0] = dim[0];
  dimlocal[1] = 1;
  dimlocal[2] = 1;
  char ocloptions[512];
  sprintf(ocloptions, "-D FLOATTYPE=%s -D DIM0=%lu -D DIM1=%lu -D DIM2=%lu -D PIXELDIM=%u -D MAXITERATIONS=%lu -D REALSTART=-2.0%c -D REALEND=2.0%c -D IMAGSTART=-2.0%c -D IMAGEND=2.0%c -D REALCONST=-0.003%c -D IMAGCONST=0.647%c -D THRESHOLD=1000.0%c", floattypes[floattypeno], dim[0], dim[1], dim[2], F64TEST2_PIXELDIM, maxiterations, floattypesuffixes[floattypeno], floattypesuffixes[floattypeno], floattypesuffixes[floattypeno], floattypesuffixes[floattypeno], floattypesuffixes[floattypeno], floattypesuffixes[floattypeno], floattypesuffixes[floattypeno]);
  
  cl_int res;
  cl_uint platformCount = 0;
  cl_uint deviceCount = 0;
  _Bool platformchosen = false;
  _Bool devicechosen = false;  
  cl_platform_id platform;
  cl_device_id device;  
  printf_cl_error(clGetPlatformIDs(MAX_PLATFORMS, NULL, &platformCount));
  platformCount = (platformCount > MAX_PLATFORMS ? MAX_PLATFORMS : platformCount);
  printf("Detected %i OpenCL Platforms.\n", platformCount);
  if (platformCount < 1) exit(1);
  cl_platform_id* platforms = calloc(platformCount, sizeof(cl_platform_id));  
  printf_cl_error(clGetPlatformIDs(platformCount, platforms, NULL));
  for (i=0; i<platformCount; i++) {
    printf_cl_error(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount));
    printf_cl_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, NAMES_LENGTH, (void *)text, NULL));
    printf("Querying Platform No. %i - %s.\n", i, text);    
    deviceCount = (deviceCount > MAX_DEVICES ? MAX_DEVICES : deviceCount);
    if (i == platformno) {
      platform = platforms[i];
      platformchosen = true;
    }
    printf("Detected %i Devices In Platform.\n", deviceCount);
    if (deviceCount > 0) {
      cl_device_id* devices = malloc(deviceCount*sizeof(cl_device_id));
      printf_cl_error(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL));
      for (j=0; j<deviceCount; j++) {
        printf_cl_error(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, NAMES_LENGTH, text, NULL));
        printf("  Device No. %i - %s", j, text);
        if ((i == platformno) && (j == deviceno)) {
          device = devices[j];
          printf("    Selected.\n");
          size_t devextsize = 0;
          printf_cl_error(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &devextsize));
          char *devext = malloc(devextsize);
          if (devext != NULL) {
            printf_cl_error(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, devextsize, devext, NULL));
            if (strstr(devext, fpextensions[floattypeno]) == NULL) {
              printf("ERROR! %s type not supported by this device.\n", floattypes[floattypeno]);
            } else {
              devicechosen = true;
            }
            free(devext);
          } else {
            printf("malloc failed!\n");
          }
        } else {
          printf("\n");
        }
      }
      free(devices);
    }
  }
  free(platforms);
  if (!platformchosen || !devicechosen) {
    printf("No Platform/Device chosen.\n");
    printf("This program runs on one and only one device. Please select platformno deviceno to include OpenCL device.\n");
    exit(1);
  }
  cl_context ContextId = clCreateContext(NULL, 1, &device, NULL, NULL, &res);
  printf_cl_error(res);
  size_t kernel_strlens[KERNEL_COUNT];
  for (i = 0; i<KERNEL_COUNT; i++) kernel_strlens[i] = strlen(oclkernels[i]);
  cl_program ProgramId = clCreateProgramWithSource(ContextId, KERNEL_COUNT, (const char **)oclkernels, (const size_t*)kernel_strlens, &res);
  printf_cl_error(res);
  printf_cl_error(clBuildProgram(ProgramId, 1, &device, ocloptions, NULL, NULL));
  uint64_t yrange = dim[0]*dim[1]*dim[2];
  cl_mem counts_mem_obj = clCreateBuffer(ContextId, CL_MEM_WRITE_ONLY, yrange*sizeof(unsigned long), NULL, &res);
  printf_cl_error(res);
  cl_kernel kernel = clCreateKernel(ProgramId, oclkernel_names[0], &res);
  printf_cl_error(res);
  printf_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&counts_mem_obj));
  cl_command_queue CommandQueueId = clCreateCommandQueue(ContextId, device, 0, &res);
  printf_cl_error(res);
  unsigned long count = 0;  
  unsigned long *counts = malloc(yrange*sizeof(unsigned long));
  printf("Executing Kernel. Please Wait...\n");
  progstart = tstampmsec();
  printf_cl_error(clEnqueueNDRangeKernel(CommandQueueId, kernel, 3, NULL, (const size_t *)dim, (const size_t *)dimlocal, 0, NULL, NULL));
  printf_cl_error(clEnqueueReadBuffer(CommandQueueId, counts_mem_obj, CL_TRUE, 0, yrange*sizeof(unsigned long), (void *)counts, 0, NULL, NULL));
  progend = tstampmsec();
  for (uint64_t y=0; y<yrange; y++) count += counts[y];  
  printf("FG Pixel Count = %lu / %lu\n", count, yrange*F64TEST2_PIXELDIM);
  if (progend > progstart) printf("Estimated %s performance = %f Gflops\n", floattypes[floattypeno], 28*maxiterations*yrange*F64TEST2_PIXELDIM/(1000000.0f*(progend - progstart)));
  printf("Kernel Duration = %li msecs\n", progend - progstart);
  free(counts);
}
