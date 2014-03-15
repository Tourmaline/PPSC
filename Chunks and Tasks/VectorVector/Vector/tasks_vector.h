/*
Vector-vector tasks
*/

#ifndef TASKS_VECTOR_HEADER
#define TASKS_VECTOR_HEADER

#include "Vector.h"
#include "VectorAssignFromChunkIDs.h"
#include "cht_matrix_lib.h"
#include "cht_type_lib.h"

namespace chtvec{
  
  /* Construct Vector from given ChunkVector with given parameters */
  template<class LeafVector>
    class VectorAssignFromChunkVector : public cht::Task {
  public:
    typedef typename LeafVector::real real;
    typedef typename chtvec::Vector<LeafVector>::VectorParams ParamsType;
    cht::ID execute(ParamsType const &, chttl::ChunkVector<real> const &);
    CHT_TASK_INPUT((ParamsType, chttl::ChunkVector<real>));
    CHT_TASK_OUTPUT((chtvec::Vector<LeafVector>));
    CHT_TASK_TYPE_DECLARATION;
  };

 
  /* Compute z = x + alpha*y */
  template<class LeafVector>
    class VectorUpdate : public cht::Task {
  public:
    typedef typename LeafVector::real real;
    typedef typename Vector<LeafVector>::VectorParams ParamsType;
    cht::ID execute(Vector<LeafVector> const &, Vector<LeafVector> const &, chttl::Basic<real> const &);
    cht::ID execute(cht::ChunkID const &, cht::ChunkID const &, cht::ChunkID const &);
    CHT_TASK_INPUT((Vector<LeafVector>, Vector<LeafVector>, chttl::Basic<real>));
    CHT_TASK_OUTPUT((chtvec::Vector<LeafVector>));
    CHT_TASK_TYPE_DECLARATION;
  };
  
  /* Combine two ChunkVector-s into one ChunkVector */
  template<typename LeafVector>
    class CombineVectors : public cht::Task {
  public:
    typedef typename LeafVector::real real;
    typedef typename chtvec::Vector<LeafVector>::VectorParams ParamsType;
    cht::ID execute(ParamsType const &, chttl::ChunkVector<real> const &, chttl::ChunkVector<real> const &);
    CHT_TASK_INPUT((ParamsType, chttl::ChunkVector<real>, chttl::ChunkVector<real>));
    CHT_TASK_OUTPUT((chttl::ChunkVector<real>));
    CHT_TASK_TYPE_DECLARATION;
  };
  
  /* Construct ChunkVector from Vector */
  template<class LeafVector>
    class ReturnVector : public cht::Task {
  public:
    typedef typename LeafVector::real real;
    typedef typename chtvec::Vector<LeafVector>::VectorParams ParamsType;
    cht::ID execute(Vector<LeafVector> const &);
    cht::ID execute(cht::ChunkID const &);
    CHT_TASK_INPUT((Vector<LeafVector>));
    CHT_TASK_OUTPUT((chttl::ChunkVector<real>));
    CHT_TASK_TYPE_DECLARATION;
  };
  
  
template<class LeafVector>
  class DotProduct : public cht::Task
    {
    public:
      typedef typename LeafVector::real real;
      cht::ID execute(Vector<LeafVector> const &, Vector<LeafVector> const &);
      CHT_TASK_INPUT((Vector<LeafVector>, Vector<LeafVector>));
      CHT_TASK_OUTPUT((chttl::Basic<real>));
      CHT_TASK_TYPE_DECLARATION;
    };
  

template<class LeafMatrix, class LeafVector>
class MatVecMul : public cht::Task
{
    public:
      typedef typename LeafMatrix::real real;
      typedef typename chtvec::Vector<LeafVector>::VectorParams ParamsType;
      cht::ID execute(chtml::Matrix<LeafMatrix> const &, Vector<LeafVector> const &);
      cht::ID execute(cht::ChunkID const &, cht::ChunkID const &);
      CHT_TASK_INPUT((chtml::Matrix<LeafMatrix>, Vector<LeafVector>));
      CHT_TASK_OUTPUT((chtvec::Vector<LeafVector>));
 private:
      cht::ID mul_lowest_level(chtml::Matrix<LeafMatrix> const &, Vector<LeafVector> const &);
      CHT_TASK_TYPE_DECLARATION;
};

  /*****************************************************************************/



  /*****************************************/
  /*             VectorUpdate
  /*****************************************/


  template<class LeafVector>
    cht::ID VectorUpdate<LeafVector>::execute(Vector<LeafVector> const & x, Vector<LeafVector> const & y, chttl::Basic<real> const & alpha)
    {
      // if lowest level
      if(!x.leafVector.empty())
	{
	  //check dimensions
	  if( x.leafVector.size() != y.leafVector.size())
	    std::runtime_error("Wrong vector dimensions in VectorAdd");
	  
	  Vector<LeafVector> *z = new Vector<LeafVector>;
	  LeafVector::update(x.leafVector, y.leafVector, alpha, z->leafVector);
	  return registerChunk<Vector<LeafVector> >(z, cht::persistent);
	}
      
      Vector<LeafVector>::checkParamsEquality(x, y); 
      std::vector<cht::ID> childTasks(3);
      
      // add child vectors separately
      childTasks[0] = registerChunk<ParamsType>(new ParamsType(x.getParams()));
      for(int i = 0; i < 2; ++i)
	{
	  childTasks[i+1] = registerTask<VectorUpdate>(x.cid_child_vectors[i], y.cid_child_vectors[i], getInputChunkID(alpha), cht::persistent);
	}
      
      // put everything together
      return registerTask<VectorAssignFromChunkIDs<LeafVector> >(childTasks, cht::persistent);
    }
  


template<class LeafVector>
   cht::ID VectorUpdate<LeafVector>::execute(cht::ChunkID const & x, cht::ChunkID const & y, cht::ChunkID const & alpha)
{   
    if (x == cht::CHUNK_ID_NULL  && y == cht::CHUNK_ID_NULL)
      return cht::CHUNK_ID_NULL;

    if (y == cht::CHUNK_ID_NULL)
      return copyChunk(x);

    if (x == cht::CHUNK_ID_NULL)
      return copyChunk(y); // *alpha?

    throw std::runtime_error("Error in MatVecMul fallback execute(): none of the input ChunkIDs are NULL!");
 
}
 

  /*****************************************/
  /*    VectorAssignFromChunkVector
  /*****************************************/
  
  template<class LeafVector>
    cht::ID VectorAssignFromChunkVector<LeafVector>::execute(ParamsType const & params, chttl::ChunkVector<real> const & chunk_vec)
    {
      int N = params.N;
      int leavesSizeMax = params.leavesSizeMax;

      if (N <= leavesSizeMax)
	{
	  Vector<LeafVector> *V = new Vector<LeafVector>;
	  V->leafVector = chunk_vec; 
	  V->setParams(params); 
	  return registerChunk<Vector<LeafVector> >(V, cht::persistent);
	}
      else
	{
	  int N1, N2, temp;
	  N1 = N;
	  temp = leavesSizeMax;
	  while (temp < N) 
	    temp *= 2;
	  N1 = temp/2;
	  
	  //dimensions of subvectors
	  N1 = N1 > N ? N : N1;
	  N2 = N - N1;
	  int dim[] = {N1, N2};

	  typename chttl::ChunkVector<real>::const_iterator it = chunk_vec.begin();
	  
	  std::vector<cht::ID> childTasks(3);
	  childTasks[0] = getInputChunkID(params);
	  for (int i = 0; i < 2; ++i)
	    {    
	      std::vector<cht::ID> child_task_vector(2);
	      std::vector<double> temp_vec(it, it + dim[i]);
	      it += dim[i];
	      child_task_vector[0] = registerChunk<ParamsType>(new ParamsType(dim[i], leavesSizeMax));
    	      child_task_vector[1] = registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(temp_vec));
    	      childTasks[i+1] = registerTask<VectorAssignFromChunkVector>(child_task_vector, cht::persistent);
	    }
	  
	  return registerTask<VectorAssignFromChunkIDs<LeafVector> >(childTasks, cht::persistent);
	}
    }
  

  /*****************************************/
  /*           CombineVectors
  /*****************************************/


  template<typename LeafVector>
    cht::ID CombineVectors<LeafVector>::execute(ParamsType const & params, chttl::ChunkVector<real> const & a1, chttl::ChunkVector<real> const & a2)
    {
      // if both children are empty, then the combined vector is vector of zeros with size V.N
      if( a1.empty() && a2.empty() )
	return registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(params.N), cht::persistent);

      if(a1.empty())
	{
	  int a2_size = a2.size();
	  chttl::ChunkVector<real> res(params.N-a2.size()); 
	  res.insert(res.end(), a2.begin(), a2.end());
	  return registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(res), cht::persistent);
	}
      if(a2.empty())
	{
	  chttl::ChunkVector<real> res(a1);
	  res.resize(params.N, 0); // resize that final vector contains N elements 
	  return registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(res), cht::persistent);
	}
      else
	{
	  chttl::ChunkVector<real> res(a1); 
	  res.insert(res.end(), a2.begin(), a2.end());
	  return registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(res), cht::persistent);
	}
    }


 
  /*****************************************/
  /*             ReturnVector
  /*****************************************/


  template<class LeafVector>
    cht::ID ReturnVector<LeafVector>::execute(Vector<LeafVector> const & x)
    {
      if (!x.leafVector.empty())
	{
	  std::vector<real> vec;
	  LeafVector::ReturnSTDVector(x.leafVector, vec);
	  // lowest level
	  return registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(vec), cht::persistent);
	}
      
      std::vector<cht::ID> childTasks(3);
      ParamsType params = x.getParams();
      childTasks[0] = registerChunk<ParamsType>(new ParamsType(params.N, params.leavesSizeMax));
      for(int i = 0; i < 2; ++i)
	childTasks[i+1] = registerTask<ReturnVector>(x.cid_child_vectors[i]);

      return registerTask<CombineVectors<LeafVector> >(childTasks, cht::persistent);
    }

  template<class LeafVector>
    cht::ID ReturnVector<LeafVector>::execute(cht::ChunkID const & x)
    {
      if( x == cht::CHUNK_ID_NULL )
	{
	  // return empty vector
	  return registerChunk<chttl::ChunkVector<real> >(new chttl::ChunkVector<real>(), cht::persistent);
	}
	throw std::runtime_error("Error in ReturnVector fallback execute(): none of the input ChunkIDs are NULL!");	
    }



  /*****************************************/
  /*             DotProduct
  /*****************************************/


template<class LeafVector>
  cht::ID DotProduct<LeafVector>::execute(Vector<LeafVector> const & x, Vector<LeafVector> const & y)
  {
      // if lowest level
      if(!x.leafVector.empty())
	{
	  //check dimensions
	  if( x.leafVector.size() != y.leafVector.size())
	    std::runtime_error("Wrong vector dimensions in VectorUpdate");
	  
	  real result;
	  LeafVector::dotProduct(x.leafVector, y.leafVector, result);
	  return registerChunk<chttl::Basic<real> >(new chttl::Basic<real>(result), cht::persistent);
	}
      
      Vector<LeafVector>::checkParamsEquality(x, y); 
      std::vector<cht::ID> childTasks(2);
      
      for(int i = 0; i < 2; ++i)
	{
	  childTasks[i] = registerTask<DotProduct>(x.cid_child_vectors[i], y.cid_child_vectors[i]);
	}
      
      return registerTask<chttl::BasicAdd<real> >(childTasks, cht::persistent);    
  }


  /*****************************************/
  /*             MatVecMul
  /*****************************************/

 template<class LeafMatrix, class LeafVector>
   cht::ID MatVecMul<LeafMatrix, LeafVector>::execute(chtml::Matrix<LeafMatrix> const & A, Vector<LeafVector> const & x)
  {

    int AM = A.get_n_rows();
    int AN = A.get_n_cols();
  
    ParamsType params = x.getParams();

    //lowest level
    if( !A.leafMatrix.empty() )
      return mul_lowest_level(A, x);

    int xN = params.N;
    int leavesSizeMax = params.leavesSizeMax;

    assert(AN == xN);

    int level_A = chtml::get_level(AM, AN, leavesSizeMax);
    int level_x = chtml::get_level(xN, 1, leavesSizeMax);

    int level_y = chtml::get_level(AM, 1, leavesSizeMax);

    assert(A.leavesSizeMax == leavesSizeMax);


    if( level_A == level_x)
      if(level_x == level_y)
	{
	  std::vector<cht::ID> resultIDs(3);
	  resultIDs[0] = registerChunk<ParamsType>(new ParamsType(AM, leavesSizeMax));
	  
	  std::vector<cht::ID> matrix_input(2), vector_input(2);
	  std::vector<cht::ID> res_input(2);
	  
	  vector_input[0] = x.cid_child_vectors[0];
	  vector_input[1] = x.cid_child_vectors[1];
	  
	  cht::ID cht_alpha = registerChunk<chttl::Basic<double> >(new chttl::Basic<double>(1));
	  
	  for(int i = 0; i < 2; ++i)
	    {
	      matrix_input[0] = A.cid_child_matrices[i];
	      matrix_input[1] = A.cid_child_matrices[i+2];
	      

	      res_input[0] = registerTask<MatVecMul<LeafMatrix, LeafVector> >(matrix_input[0], vector_input[0]);
 
	      res_input[1] = registerTask<MatVecMul<LeafMatrix, LeafVector> >(matrix_input[1], vector_input[1]);
	      
	      resultIDs[i+1] = registerTask<VectorUpdate<LeafVector> >(res_input[0], res_input[1], cht_alpha, cht::persistent);
	    }
	  return registerTask<VectorAssignFromChunkIDs<LeafVector> >(resultIDs, cht::persistent);
	}
      else if(level_x > level_y)
	{
	  std::vector<cht::ID> resultIDs(2); 
	  cht::ID cht_alpha = registerChunk<chttl::Basic<double> >(new chttl::Basic<double>(1));

	  resultIDs[0] = registerTask<MatVecMul<LeafMatrix, LeafVector> >(A.cid_child_matrices[0], x.cid_child_vectors[0]);

	  resultIDs[1] = registerTask<MatVecMul<LeafMatrix, LeafVector> >(A.cid_child_matrices[2], x.cid_child_vectors[1]);
	  
	  return registerTask<VectorUpdate<LeafVector> >(resultIDs[0], resultIDs[1], cht_alpha, cht::persistent);	  
	}
      else throw std::runtime_error("Error in MatVecMul: level_A == level_x and level_x < level_y.");
    
    
    if( level_A > level_x )
      if( level_x < level_y)
      {
	  std::vector<cht::ID> resultIDs(3);
	  resultIDs[0] = registerChunk<ParamsType>(new ParamsType(AM, leavesSizeMax));
	  

	  resultIDs[1] = registerTask<MatVecMul<LeafMatrix, LeafVector> >(A.cid_child_matrices[0], getInputChunkID(x), cht::persistent);

	  resultIDs[2] = registerTask<MatVecMul<LeafMatrix, LeafVector> >(A.cid_child_matrices[1], getInputChunkID(x), cht::persistent);
	  return registerTask<VectorAssignFromChunkIDs<LeafVector> >(resultIDs, cht::persistent);
      }
      else throw std::runtime_error("Error in MatVecMul: level_A > level_x and level_x > level_y.");
    
    throw std::runtime_error("Error in MatVecMul: Reached impossible point.");
  }
 
 

 template<class LeafMatrix, class LeafVector>
   cht::ID MatVecMul<LeafMatrix, LeafVector>::mul_lowest_level(chtml::Matrix<LeafMatrix> const & A, Vector<LeafVector> const & x)
  {
    ParamsType params = x.getParams();
    Vector<LeafVector> *res = new Vector<LeafVector>;
    LeafMatrix::multiply_by_vector(A.leafMatrix, x.leafVector, res->leafVector);
    params.N = res->leafVector.size();
    res->setParams(params);
    return registerChunk<Vector<LeafVector> >(res, cht::persistent);
  }



  template<class LeafMatrix, class LeafVector>
    cht::ID MatVecMul<LeafMatrix, LeafVector>::execute(cht::ChunkID const & A,
							cht::ChunkID const & x) {
    if (A == cht::CHUNK_ID_NULL)
      {
	return cht::CHUNK_ID_NULL;
      }
    throw std::runtime_error("Error in MatVecMul fallback execute(): none of the input ChunkIDs are NULL!");
  } 


} // end of namespace

#endif
