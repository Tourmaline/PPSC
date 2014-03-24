#include "../../common/types_kernel.h"

__kernel void VectorUpdate( __global real * x, __global real * y, real alpha, int N )
{
  unsigned int itemIDglobal = get_global_id(0);
  unsigned int globalSize   = get_global_size(0);

  unsigned int i = itemIDglobal;

  while( i < N )
    {
      x[i] += y[i]*alpha;
      i += globalSize;
    }

}
