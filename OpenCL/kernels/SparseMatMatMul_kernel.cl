#include "../../common/types_kernel.h"

__kernel void SparseMatMatMul(__global real *AValues, __global int *AColInd, __global int* ARowPtr,
			     __global real *BValues, __global int *BColInd, __global int* BRowPtr,
			     __global real *CValues, __global int *CColInd, __global int* CRowInd, 
			     __global int *params, __global real * tempResults, __global int * NNZ,
			     __local int *locColA /*[numWaprsLocal]*/, __local real *locValA) // one plane for each warp
{

  int PART_ROW_DIM, PART_B = params[4];
  int ITEMS_PER_ROW = params[3];

  int itemIDglobal = get_global_id(0);                        // ID in grid
  int itemIDlocal  = get_local_id(0);                         // ID in work group
  int itemIDwarp   = itemIDglobal & (ITEMS_PER_ROW - 1);      // ID in warp

  int warpIDlocal  = itemIDlocal / ITEMS_PER_ROW;             // ID of warp in work group
  int warpIDglobal = itemIDglobal / ITEMS_PER_ROW;            // ID of warp in grid
  int globalSize   = get_global_size(0);                      // total number of work items in grid
  int localSize    = get_local_size(0);                       // total number of work items in work group

  int numWarpsGlobal  = globalSize / ITEMS_PER_ROW; 
  int numWarpsLocal = localSize / ITEMS_PER_ROW;

  int tempResultsDim = numWarpsGlobal * PART_B;

  int rowC, partRowB, j, k, ColTempA, ColStartA, ColEndA;
  int RowStartB, RowEndB, RowTempB, colA, colB, startPartB, endPartB;
  real val, valA, valB;

  int NRowsA = params[0];
  int NColsA = params[1];
  int NRowsB = NColsA;
  int NColsB = params[2];
  int NRowsC = NRowsA;
  int NColsC = NColsB;
  int num_elements;

  int count;

  // for every row of C
  for(rowC = warpIDglobal; rowC < NRowsC; rowC += numWarpsGlobal)
    {
      // we devide row of B on pieces 
      // so do for every piece

      PART_ROW_DIM = PART_B;
      partRowB = 0;
      num_elements = NColsB;

      while( num_elements > 0)
      {
	  startPartB = partRowB;
	  endPartB = startPartB + PART_ROW_DIM;
	  
	  // set elements of tempResult to zero
	  for( j = itemIDwarp; j < PART_B; j += ITEMS_PER_ROW )
	    tempResults[warpIDglobal*PART_B + j] = 0; 

	  ColStartA = ARowPtr[rowC] + itemIDwarp;  // column of element in row of A for current work item in warp
	  ColEndA   = ARowPtr[rowC+1];             // the last column of this row
	  
	  ColTempA = ColStartA;

	  // if at least one work item in warp has element
	  // ( = if first item has work to do)
	  while(ColTempA - itemIDwarp < ColEndA)
	    {
	      // get element of this row of A for current work item
	      colA = ColTempA < ColEndA ? AColInd[ColTempA] : -1;
	      valA = ColTempA < ColEndA ? AValues[ColTempA] : 0.0;

	      // now every work item in warp has its own element of A
	      for(k = 0; k < ITEMS_PER_ROW; k++)
		{
		  if(itemIDwarp == k)
		    {
		      locColA[warpIDlocal] = colA;
		      if(colA != -1)
		        locValA[warpIDlocal] = valA;
		    }
		  
		 if( locColA[warpIDlocal] == -1 ) break;
		  
		  // so now locColA[warpIDlocal] is current row of B for this warp
		  
		  //current row of B
		  RowStartB = BRowPtr[locColA[warpIDlocal]] + itemIDwarp;
                  RowEndB   = BRowPtr[locColA[warpIDlocal]+1];
		  
		  RowTempB  = RowStartB;
		  
		  for(RowTempB; RowTempB < RowEndB; RowTempB += ITEMS_PER_ROW)
		    {
		      // get the element of this row of B for current work item
		      colB = BColInd[RowTempB];

		      // we are interested just in the current part of row of B
		      if( colB < startPartB ) continue;
		      if( colB >= endPartB ) break;

		      valB = BValues[RowTempB];
		      tempResults[warpIDglobal*PART_B + colB % PART_B] += locValA[warpIDlocal] * valB;
		    }
		}

	      // which is next my element in row of A
	      ColTempA += ITEMS_PER_ROW;
	      
	    } // while 


      for(j = itemIDwarp; j < PART_B/*ROW_DIM*/; j += ITEMS_PER_ROW)
        {
          val = tempResults[warpIDglobal*PART_B+j];
          if( val != 0 )
            {
              count = atomic_add(NNZ, 1);
              CValues[count] = val;
              CRowInd[count] = rowC;
              CColInd[count] = j + partRowB;
            }
        }
	
	num_elements -= PART_ROW_DIM;
	partRowB += PART_ROW_DIM;
	PART_ROW_DIM = min(num_elements, PART_B);

	} // for each part for B

    } // for every for in C 

}


