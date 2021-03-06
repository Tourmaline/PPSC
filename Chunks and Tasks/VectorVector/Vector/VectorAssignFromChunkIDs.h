#ifndef VECTOR_ASSIGN_FROM_CHUNK_IDS_HEADER
#define VECTOR_ASSIGN_FROM_CHUNK_IDS_HEADER

#include "chunks_and_tasks.h"
#include "Vector.h"

namespace chtvec{
  
  template<class LeafVector>
    class VectorAssignFromChunkIDs: public cht::Task {
  public:
typedef typename Vector<LeafVector>::VectorParams Params;
    cht::ID execute(Params const &, cht::ChunkID const &, cht::ChunkID const &); 
    CHT_TASK_INPUT((Params, cht::ChunkID, cht::ChunkID));
    CHT_TASK_OUTPUT((chtvec::Vector<LeafVector>));
    CHT_TASK_TYPE_DECLARATION;
  };
  
  template<class LeafVector>
    cht::ID VectorAssignFromChunkIDs<LeafVector>::execute(Params const & params,
							  cht::ChunkID const & a1,
							  cht::ChunkID const & a2)
    {
      if (a1 == cht::CHUNK_ID_NULL &&
	  a2 == cht::CHUNK_ID_NULL)
	return cht::CHUNK_ID_NULL;

      Vector<LeafVector> *z  = new Vector<LeafVector>;
      z->setParams(params);
      z->cid_child_vectors[0] = a1;
      z->cid_child_vectors[1] = a2;

      return registerChunk<Vector<LeafVector> >(z, cht::persistent);
    }
  
  
}


#endif
