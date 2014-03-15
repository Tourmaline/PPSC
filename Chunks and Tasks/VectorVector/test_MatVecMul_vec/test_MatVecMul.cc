/*
Matrix vector multiplication

USAGE:
./test_MatVecMul_manager num_workers num_threads block_size Mfile xfile [reffile]

OUTPUT:
num_workers  num_threads  N  M  NNZ  leave_size_max  min_time 
*/



#include <iostream>
#include "cht_vec_lib.h"
#include "../BasicVector/BasicVector.h"
#include "sparse_matrix_lib.h"
#include "cht_matrix_lib.h"
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



void matvec_serial(const std::vector<int> &RowInd, const std::vector<int> &ColInd, 
		   const std::vector<real> &Values, const std::vector<real> &x,  std::vector<real> & y, int M, int N)
{

  std::vector<real> A(N*M);

  //construct dense
  for(int i = 0; i < RowInd.size(); ++i)
    A[RowInd[i]*N + ColInd[i]] = Values[i];

  y.clear();
  y.resize(M);

  for(int i = 0; i < M; ++i)
    for(int j = 0; j < N; ++j)
    {
      y[i] += A[i*N+j] * x[j]; 
    }
}



template<typename LeafMatrixType>
void test_matvecmul(const std::vector<int> &RowInd, const std::vector<int> &ColInd, 
		    const std::vector<real> &Values, const std::vector<real> &x, int N, int M, int NNZ, int leave_size_max, std::vector<real> & y)
{

  typedef typename chtvec::Vector<BasicVector<double> >::VectorParams ParamsType;
  typename LeafMatrixType::Params leaf_params;
  
  TIME start;

  double elapsed_time[NUM_TESTS];
	
  cht::ChunkID chunk_chvec_x = cht::registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(x));
  cht::ChunkID chunk_params_x = cht::registerChunk<ParamsType>(new ParamsType(x.size(), leave_size_max));
  cht::ChunkID chunk_x = cht::executeMotherTask<chtvec::VectorAssignFromChunkVector<BasicVector<real> > >(chunk_params_x, chunk_chvec_x);

  set_block_size<LeafMatrixType>(leaf_params, leave_size_max);
  cht::ChunkID chunk_params_A = cht::registerChunk<chtml::MatrixParams<LeafMatrixType> >(new chtml::MatrixParams<LeafMatrixType>(M, N, leave_size_max, 0, 0, leaf_params));	
  
  cht::ChunkID chunk_A = create_matrix<LeafMatrixType>(chunk_params_A, RowInd, ColInd, Values);

  for( int test = 0; test < NUM_TESTS; test++  )
    {
      start = tic();

      cht::ChunkID chunk_Ax = cht::executeMotherTask<chtvec::MatVecMul<LeafMatrixType, BasicVector<real> > >(chunk_A, chunk_x);

      elapsed_time[test] = toc(start);

#ifdef DEBUG
      cht::ChunkID chunk_res = cht::executeMotherTask<chtvec::ReturnVector<BasicVector<real> > >(chunk_Ax);
      
      cht::shared_ptr<chttl::ChunkVector<real> const> res;
      cht::getChunk(chunk_res, res);      
      
      
      if(!y.empty())	  
	{
	  assert(res->size() == M );
	  for(int i = 0; i < M; ++i)
	    {
	      //std::cout << res->at(i)<< " = "<< y[i] << "  ";                                                                                                       
	      assert(std::abs( res->at(i) - y[i] ) < 1e-4);
	    }
	  
	  std::cout << "Verified (with input result file)..." << std::endl;
	}
      else
	{      
	  // Compare with serial result
	  std::vector<real> y;
	  matvec_serial(RowInd, ColInd, Values, x, y, M, N);
	  
	  assert(res->size() == M );
	  for(int i = 0; i < M; ++i)
	    {
	      assert(std::abs( res->at(i) - y[i] ) < 1e-8);
	    }
	  
	  std::cout << "Verified (with serial computations)..." << std::endl;
	}
      
      
      deleteChunk(chunk_res);
      
#endif     
      deleteChunk(chunk_Ax);
      
    }
  
  deleteChunk(chunk_chvec_x);	
  deleteChunk(chunk_x);
  deleteChunk(chunk_A);
  deleteChunk(chunk_params_A);
  deleteChunk(chunk_params_x);
  
  double min_time = elapsed_time[0];
  for( int test = 0; test < NUM_TESTS; ++test )
    min_time = std::min(elapsed_time[test], min_time);
  
  std::cout << N  << "  " << M << "  " << NNZ <<  "  " << leave_size_max  << "  " << min_time << std::endl;
}




int main(int argc, char** argv)
{
  if( argc < 6 || argc > 7  )
    {
      printf("Usage: %s num_workers num_threads block_size Mfile xfile [reffile] \n", argv[0]);
      return EXIT_FAILURE;
    }

  int num_workers = atoi(argv[1]);
  int num_threads = atoi(argv[2]);    

  cht::extras::setNWorkers(num_workers);
  cht::extras::setNoOfWorkerThreads(num_threads);
  cht::setOutputLevel(cht::Output::Info);
  cht::start();

    char Mfilename[50];
    char xfilename[50];
    char yfilename[50];

    int verify = (argc == 7 ? 1 : 0);

    std::vector<int> RowInd;
    std::vector<int> ColInd;
    std::vector<real> Values;
    std::vector<real> x;
    std::vector<real> y;
  
    strcpy(Mfilename, argv[4]);
    strcpy(xfilename, argv[5]);
    if(verify)
      strcpy(yfilename, argv[6]);

    int leave_size_max = atoi(argv[3]);
    

 

    /* READ MATRIX FROM FILE IN COO FORMAT */
    
    int NRows, NCols, NNZ;

    std::ifstream Mfile;
    Mfile.open (Mfilename, std::ios::in);
    if (!Mfile.is_open())
      {
	printf("Error: cannot open file\n");
	return EXIT_FAILURE;
      }

    Mfile >> NRows;
    Mfile >> NCols;
    Mfile >> NNZ;

    RowInd.resize(NNZ);
    ColInd.resize(NNZ);
    Values.resize(NNZ);


    for( int i = 0; i < NNZ; i++)
      Mfile >> RowInd[i];
    for( int i = 0; i < NNZ; i++)
      Mfile >> ColInd[i];
    for( int i = 0; i < NNZ; i++)
      Mfile >> Values[i];
 
    Mfile.close();


    /* READ VECTOR FROM FILE */
    int N;

    std::ifstream xfile;
    xfile.open (xfilename, std::ios::in);
    if (!xfile.is_open())
      {
	printf("Error: cannot open file\n");
	return EXIT_FAILURE;
      }


    xfile >> N;
    assert(N == NCols);
    
    x.resize(N);
    for( int i = 0; i < N; i++)
      xfile >> x[i];

    xfile.close();


    int M = NRows;
    if( verify )
      {
	int M;
	std::ifstream yfile;
	yfile.open (yfilename, std::ios::in);
	if (!yfile.is_open())
	  {
	    printf("Error: cannot open file\n");
	    return EXIT_FAILURE;
	  }
	
	
	yfile >> M;
	assert(M == NRows);
	
	y.resize(M);
	for( int i = 0; i < M; i++)
	  yfile >> y[i];
	
	yfile.close();
      }

   
    std::cout << num_workers << "  " << num_threads << "  ";
    test_matvecmul<sml::CSCMatrix<double> >(RowInd, ColInd, Values, x, M, N, NNZ,leave_size_max, y);

  cht::stop();

  return 0;
}
