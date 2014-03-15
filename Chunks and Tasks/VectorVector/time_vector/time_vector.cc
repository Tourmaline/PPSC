#include <iostream>
#include "../BasicVector/BasicVector.h"
#include "cht_vec_lib.h"
#include "basic_matrix_lib.h"
#include "block_sparse_matrix_lib.h"
#include "cht_type_lib.h"
#include "../../test_tools/tic_toc.h"

#define NUM_TESTS 1

//#define DEBUG

int main(int argc, char**argv)
{
#ifndef DEBUG
  if( argc != 6 )
      {
	printf("Usage: %s num_workers num_threads N block_size {1=add,2=dot}\n", argv[0]);
	return EXIT_FAILURE;
      }
  int num_workers = atoi(argv[1]); 
  int num_threads = atoi(argv[2]);
#else
  int num_workers = 2;
  int num_threads = 4;
#endif

  cht::extras::setNWorkers(num_workers);
  cht::extras::setNoOfWorkerThreads(num_threads);
  cht::setOutputLevel(cht::Output::Info);
  cht::start();

  int test_add = 1;
  int test_dot = 1;

#ifndef DEBUG
  int N = atoi(argv[3]);
  int leave_size_max = atoi(argv[4]);
  int test = atoi(argv[5]);
  test_add = (test == 1) ? 1 : 0;
  test_dot = (test == 2) ? 1 : 0;
#else
  int N = 10; // size of vector
  int leave_size_max = 2;
#endif  

  std::vector<double> vecx(N);
  for(int i = 0; i < vecx.size(); ++i)
    vecx[i] = 1;
  
  std::vector<double> vecy(N);
  for(int i = 0; i < vecy.size(); ++i)
    vecy[i] = 1;

  TIME start;

  typedef typename chtvec::Vector<BasicVector<double> >::VectorParams ParamsType;

  double elapsed_time[NUM_TESTS];

  
  for( int test = 0; test < NUM_TESTS; test++  )
    {
      
      start = tic();
      
      cht::ChunkID chunk_chvec_x = cht::registerChunk<chttl::ChunkVector<double> >(new chttl::ChunkVector<double>(vecx));
      cht::ChunkID chunk_chvec_y = cht::registerChunk<chttl::ChunkVector<double> >(new chttl::ChunkVector<double>(vecy));
      
      cht::ChunkID chunk_params_x = cht::registerChunk<ParamsType>(new ParamsType(vecx.size(), leave_size_max));
      cht::ChunkID chunk_params_y = cht::registerChunk<ParamsType>(new ParamsType(vecy.size(), leave_size_max));
      
      cht::ChunkID chunk_vec_x = cht::executeMotherTask<chtvec::VectorAssignFromChunkVector<BasicVector<double> > >(chunk_params_x, chunk_chvec_x);
      cht::ChunkID chunk_vec_y = cht::executeMotherTask<chtvec::VectorAssignFromChunkVector<BasicVector<double> > >(chunk_params_y, chunk_chvec_y);
  

      if( test_add )
	{  
	  /* TEST VECTOR ADD */
	
	  double beta = 1;
	  cht::ChunkID chunk_beta = cht::registerChunk<chttl::Basic<double> >(new chttl::Basic<double>(beta));
	  cht::ChunkID chunk_res1 = cht::executeMotherTask<chtvec::VectorUpdate<BasicVector<double> > >(chunk_vec_x, chunk_vec_y, chunk_beta); // res = x+y
	  cht::ChunkID chunk_res2 = cht::executeMotherTask<chtvec::ReturnVector<BasicVector<double> > >(chunk_res1);
	
	  cht::shared_ptr<chttl::ChunkVector<double> const> res2;
	  cht::getChunk(chunk_res2, res2);
	  
#ifdef DEBUG  
	  assert(res2->size() == vecx.size());
	  for(int i = 0; i < res2->size(); ++i)
	    assert(res2->at(i) == 2);
#endif
	  
	  deleteChunk(chunk_beta);
	  deleteChunk(chunk_res1);
	  deleteChunk(chunk_res2);
	}
      
      if(test_dot)
	{
	  /* Test DotProduct */
	  
	  cht::ChunkID chunk_res3 = cht::executeMotherTask<chtvec::DotProduct<BasicVector<double> > >(chunk_vec_x, chunk_vec_y);
	  cht::shared_ptr<chttl::Basic<double> const> res3;
	  cht::getChunk(chunk_res3, res3);
	
#ifdef DEBUG    
	  double ref_res = 0;
	  for(int i = 0; i < vecy.size(); ++i)
	    ref_res += vecx[i]*vecy[i];
  
	  assert(ref_res == *res3);
#endif
	  deleteChunk(chunk_res3);
	}
      
      
      deleteChunk(chunk_chvec_x);
      deleteChunk(chunk_params_x);
      deleteChunk(chunk_vec_x);
      deleteChunk(chunk_chvec_y);
      deleteChunk(chunk_params_y);
      deleteChunk(chunk_vec_y);
      
      elapsed_time[test] = toc(start);
      
    }  
  
  double min_time = elapsed_time[0];
  for( int test = 0; test < NUM_TESTS; ++test )
    min_time = std::min(elapsed_time[test], min_time);

  std::cout << num_workers << "  " << num_threads << "  "  <<  N << "  " << leave_size_max << "  " << min_time << std::endl;

  cht::stop();
  return 0;
}
