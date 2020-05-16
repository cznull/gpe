
//#define _D

#ifdef _D

typedef double current;
typedef double2 current2;

#define cufftExec cufftExecZ2Z
#define CUFFT_cur2cur CUFFT_Z2Z

#else

typedef float current;
typedef float2 current2;

#define cufftExec cufftExecC2C
#define CUFFT_cur2cur CUFFT_C2C

#endif

