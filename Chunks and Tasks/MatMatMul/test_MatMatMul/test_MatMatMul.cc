/*
Dense matrix matrix multiplication

USAGE::
./test_MatMatMul_manager num_workers num_threads MA NA NB leave_size_max

OUTPUT:
num_workers num_threads MA NA NB leave_size_max min_time 

*/

#include <iostream>
#include "cht_matrix_lib.h"
#include "sparse_matrix_lib.h"
#include "basic_matrix_lib.h"
#include "block_sparse_matrix_lib.h"
#include "../../test_tools/test_tools.h"
#include "../../test_tools/tic_toc.h"

#define NUM_TESTS 3

//#define DEBUG

typedef double real;

template<typename LeafMatrixType>
cht::ChunkID create_matrix(cht::ChunkID const & cid_param,
			   std::vector<int> const & row,
			   std::vector<int> const & col,
			   std::vector<double> const & val) 
{
  cht::ChunkID cid_rowind = cht::registerChunk<chttl::ChunkVector<int> >(new chttl::ChunkVector<int>(row));
  cht::ChunkID cid_colind = cht::registerChunk<chttl::ChunkVector<int> >(new chttl::ChunkVector<int>(col));
  cht::ChunkID cid_values = cht::registerChunk<chttl::ChunkVector<double> >(new chttl::ChunkVector<double>(val));

  cht::ChunkID cid_matrix = 
    cht::executeMotherTask<chtml::MatrixAssignFromSparse<LeafMatrixType> >(cid_param,
									   cid_rowind,
									   cid_colind,
									   cid_values);
  cht::deleteChunk(cid_rowind);
  cht::deleteChunk(cid_colind);
  cht::deleteChunk(cid_values);  
  return cid_matrix;
}


template<typename LeafMatrixType>
void test_matmatmul(const std::vector<int> &RowInd_A, const std::vector<int> &ColInd_A, const std::vector<real> &Values_A,
		    const std::vector<int> &RowInd_B, const std::vector<int> &ColInd_B, const std::vector<real> &Values_B, int MA, int NA, int NB, int leave_size_max)
{
  
  TIME start;
  double elapsed_time[NUM_TESTS];
  
  typename LeafMatrixType::Params leaf_params;
  set_block_size<LeafMatrixType>(leaf_params, leave_size_max);
  cht::ChunkID chunk_params_A = cht::registerChunk<chtml::MatrixParams<LeafMatrixType> >(new chtml::MatrixParams<LeafMatrixType>(MA, NA, leave_size_max, 0, 0, leaf_params));
  cht::ChunkID chunk_params_B = cht::registerChunk<chtml::MatrixParams<LeafMatrixType> >(new chtml::MatrixParams<LeafMatrixType>(NA, NB, leave_size_max, 0, 0, leaf_params));
  
  
  cht::ChunkID chunk_A = create_matrix<LeafMatrixType>(chunk_params_A, RowInd_A, ColInd_A, Values_A);
  cht::ChunkID chunk_B = create_matrix<LeafMatrixType>(chunk_params_B, RowInd_B, ColInd_B, Values_B);
  
  for( int test = 0; test < NUM_TESTS; test++  )
      {

	start = tic();

	cht::ChunkID chunk_res = cht::executeMotherTask<chtml::MatrixMultiply<LeafMatrixType,false,false> >(chunk_A, chunk_B);

	elapsed_time[test] = toc(start);

#ifdef DEBUG
	double fro_norm = NA*std::sqrt(MA*NB);
	assert(chtml::normFrobenius<LeafMatrixType>(chunk_res) == fro_norm);
	std::cout << "Verified (frobenius norm)..." << std::endl;
#endif


	cht::deleteChunk(chunk_res);

      }
  
  cht::deleteChunk(chunk_A);
  cht::deleteChunk(chunk_B);
  cht::deleteChunk(chunk_params_A);
  cht::deleteChunk(chunk_params_B);
  

  double min_time = elapsed_time[0];
  for( int test = 0; test < NUM_TESTS; ++test )
    min_time = std::min(elapsed_time[test], min_time);

  std::cout  << MA  << "  " << NA  << "  " << NB <<  "  " << leave_size_max  << "  " << min_time << std::endl;
}




int main(int argc, char** argv)
{

  if( argc != 7 )
    {
      printf("Usage: %s num_workers num_threads MA NA NB leave_size_max\n", argv[0]);
      return EXIT_FAILURE;
    }
  int num_workers = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  int leave_size_max = atoi(argv[6]);

  cht::extras::setNWorkers(num_workers);
  cht::extras::setNoOfWorkerThreads(num_threads);
  cht::setOutputLevel(cht::Output::Info);
  cht::start();

   int MA = atoi(argv[3]);
  int NA = atoi(argv[4]);
  int MB = NA;
  int NB = atoi(argv[5]);  
  int NUM_A = MA*NA;
  int NUM_B = MB*NB; 

  std::vector<int> RowInd_A(NUM_A);
  std::vector<int> ColInd_A(NUM_A);
  std::vector<real> Values_A(NUM_A);
  std::vector<int> RowInd_B(NUM_B);
  std::vector<int> ColInd_B(NUM_B);
  std::vector<real> Values_B(NUM_B);


  // full matrices of 1s
  int ind = 0;
  for( int r = 0; r < MA; r++ )
    for( int c = 0; c < NA; c++ )
      {
	RowInd_A[ind]= r;
	ColInd_A[ind]= c;
	Values_A[ind] = 1;
	ind++;
      }

  ind = 0;
  for( int r = 0; r < MB; r++ )
    for( int c = 0; c < NB; c++ )
      {
	RowInd_B[ind]= r;
	ColInd_B[ind]= c;
	Values_B[ind] = 1;
	ind++;
      }

  std::cout << num_workers << "  " << num_threads << "  ";
  test_matmatmul<bml::FullMatrix<double> >(RowInd_A, ColInd_A, Values_A, RowInd_B, ColInd_B, Values_B, MA, NA, NB, leave_size_max);

  cht::stop();

  return 0;
}
