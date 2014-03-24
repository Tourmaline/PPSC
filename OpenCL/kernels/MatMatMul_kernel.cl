#include "../../common/types_kernel.h"

__kernel void MatMatMul( __global real *A, __global real *B, __global real *C, 
			 __global int *params, __local real *Asub, __local real *Bsub)
{

  int itemIDglobalX  = get_global_id(0);  // ID in NDRange
  int itemIDglobalY  = get_global_id(1);
  int itemIDlocalX   = get_local_id(0);   // ID in work group
  int itemIDlocalY   = get_local_id(1); 
  int groupIDX       = get_group_id(0);   // ID of work group in NDRange
  int groupIDY       = get_group_id(1);
  int localSize      = get_local_size(0); // total number of work items in work group

  int NRowsA = params[0];
  int NColsA = params[1];
  int NRowsB = NColsA;	
  int NColsB = params[2]; 

  real Cval  = 0;

  // first sub-matrix of A
  int Crow = itemIDglobalY;
  int Ccol = itemIDglobalX;

  int temp_subind;
  int temp_rowB;
  int temp_colA;
  
  for (int i = 0; i < (NColsA-1)/localSize+1; i++)
      {
        temp_subind = itemIDlocalY * localSize + itemIDlocalX;
	temp_colA   = i*localSize + itemIDlocalX;
    	temp_rowB   = i*localSize + itemIDlocalY;

	if(Crow >= NRowsA || temp_colA >= NColsA)
	  Asub[temp_subind] = 0;
	else
	  Asub[temp_subind] = A[Crow*NColsA + temp_colA];
        
	if(Ccol >= NColsB || temp_rowB >= NRowsB)
	  Bsub[temp_subind] = 0;
	else
	  Bsub[temp_subind] = B[NColsB * temp_rowB + Ccol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < localSize; ++k)
            Cval += Asub[itemIDlocalY * localSize + k] * Bsub[k * localSize + itemIDlocalX];
 
       barrier(CLK_LOCAL_MEM_FENCE); 
    }
  
  if(Crow < NRowsA && Ccol < NColsB)
    C[Crow * NColsB + Ccol] =  Cval;
}

/*
Some comments:
Arguments to __kernel functions in a program cannot be declared as a pointer to a pointer(s)
*/


