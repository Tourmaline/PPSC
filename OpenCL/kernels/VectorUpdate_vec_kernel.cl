#include  "../../common/types_kernel.h"

__kernel void VectorUpdate( __global real4 * x, __global real4 * y, real alpha, int N )
{
  int itemIDglobal = get_global_id(0);
  int globalSize = get_global_size(0);

  int i = itemIDglobal;

  while( i < N )
    { 
      x[i] += y[i]*alpha;//fma(y[i], alpha, x[i]);
      i += globalSize;
    }
}
