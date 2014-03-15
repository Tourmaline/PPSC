#ifndef VECTOR_HEADER
#define VECTOR_HEADER

#include <stdexcept>
#include <cstring>
#include "cht_type_lib.h"

namespace chtvec{
  
  template<class LeafVector>
    class Vector : public cht::Chunk 
    {
    public:
      struct VectorParams;
      
    private:
      VectorParams params;
      
    public:
      typedef LeafVector LeafVectorType;
      cht::ChunkID cid_child_vectors[2];

      // for Chunk type
      void writeToBuffer ( char * dataBuffer, 
			   size_t const bufferSize ) const;
      size_t getSize() const;
      void assignFromBuffer ( char const * dataBuffer, 
			      size_t const bufferSize);
      size_t memoryUsage() const;
      void getChildChunks(std::list<cht::ChunkID> & childChunkIDs) const;
      
      //Other
      int size() const {return params.N;}
      int getLeavesSizeMax() const {return params.leavesSizeMax;}
      
      VectorParams getParams() const;
      void setParams(VectorParams const &);
      static void checkParamsEquality(Vector<LeafVector> const &, Vector<LeafVector> const &);
      
      // if lower level
      LeafVectorType leafVector;
      

      /* VectorParams */

      struct VectorParams : public cht::Chunk 
      {
	int N; // dimension of the vector
	int leavesSizeMax;
	
	VectorParams(){}
      VectorParams(int N_, int leavesSizeMax_) : 
	N(N_), leavesSizeMax(leavesSizeMax_){}  
      VectorParams(VectorParams const & vp) : 
	N(vp.N), leavesSizeMax(vp.leavesSizeMax){}
	
	// for Chunk type
	void writeToBuffer ( char * dataBuffer, 
			     size_t const bufferSize ) const;
	size_t getSize() const;
	void assignFromBuffer ( char const * dataBuffer, 
				size_t const bufferSize);
	size_t memoryUsage() const;
	
	CHT_CHUNK_TYPE_DECLARATION;
      };
      
      
      CHT_CHUNK_TYPE_DECLARATION;
    };
  
  
  template<class LeafVector>
    void Vector<LeafVector>::writeToBuffer(char * dataBuffer, 
					   size_t const bufferSize) const
    {
      if (bufferSize != getSize())
	throw std::runtime_error("Wrong buffer size to write "
				 "to buffer in Vector::writeToBuffer.");
      if (!leafVector.empty()) 
	{
	  char* p = dataBuffer;
	  *p = 1;
	  p++;
	  int params_size = params.getSize();
	  params.writeToBuffer(p, params_size);
	  p += params_size;
	  // Now comes the leafMatrix
	  leafVector.writeToBuffer(p, bufferSize - 1 - params_size);
	}
      else 
	{
	  // This is not the lowest level
	  char* p = dataBuffer;
	  *p = 0;
	  p++;
	  int params_size = params.getSize();
	  params.writeToBuffer(p, params_size);
	  p += params_size;
	  memcpy(p, &cid_child_vectors[0], 2*sizeof(cht::ChunkID));
	} 
    }
  
  
  template<class LeafVector>
    void Vector<LeafVector>::assignFromBuffer ( char const * dataBuffer, 
						size_t const bufferSize) 
    {
      if(bufferSize < sizeof(char))
	throw std::runtime_error("Wrong buffer size to assign "
				 "from buffer in Vector.");
      if (dataBuffer[0]) 
	{
	  // This is the lowest level
	  const char* p = &dataBuffer[1];
	  int params_size = params.getSize();
	  params.assignFromBuffer(p, params_size);
	  p += params_size;
	  leafVector.assignFromBuffer(p, bufferSize-1-params_size);
	  for(int i = 0; i < 2; i++)
	    cid_child_vectors[i] = cht::CHUNK_ID_NULL;
	}
      else 
	{
	  // This is not the lowest level
	  if (bufferSize != getSize())
	    throw std::runtime_error("Vector::assignFromBuffer: wrong data buffer size for vector.");

	  leafVector.clear();
	  const char* p = &dataBuffer[1];
	  int params_size = params.getSize();
	  params.assignFromBuffer(p, params_size);
	  p += params_size;
	  memcpy(cid_child_vectors, p, 2*sizeof(cht::ChunkID));
	}
    }
  
  template<class LeafVector>
    size_t Vector<LeafVector>::getSize() const 
    {
      if (!leafVector.empty()) 
	{
	  // This is lowest level
	  return sizeof(char) + params.getSize() + leafVector.getSize();
	}
      else 
	{
	  // This is not the lowest level
	  return sizeof(char) + params.getSize() + 2*sizeof(cht::ChunkID);
	}
    }
  
  
  template<class LeafVector>
    size_t Vector<LeafVector>::memoryUsage() const 
    {
      return getSize();
    }
  
  template<class LeafVector>
    void Vector<LeafVector>::getChildChunks(std::list<cht::ChunkID> & childChunkIDs) const {
    if (!leafVector.empty()) 
      return; // This is lowest level, do nothing.
    else 
      {
	for(int k = 0; k < 2; k++) 
	  childChunkIDs.push_back(cid_child_vectors[k]);
      }
  }
  
  template<class LeafVector>
    typename Vector<LeafVector>::VectorParams Vector<LeafVector>::getParams() const
    {
      return params;
    }
  
  template<class LeafVector>
    void Vector<LeafVector>::setParams(Vector<LeafVector>::VectorParams const & vp)
    {
      params = vp;
    }
  
  template<class LeafVector>
    void Vector<LeafVector>::checkParamsEquality(Vector<LeafVector> const & x, Vector<LeafVector> const & y) 
    {
      assert(x.size() == y.size());
      assert(x.getLeavesSizeMax() == y.getLeavesSizeMax());
    }
  
  
  
  // Vector::VectorParams
  
  template<class LeafVector>
    void Vector<LeafVector>::VectorParams::writeToBuffer ( char * dataBuffer, 
							   size_t const bufferSize ) const 
    {
      if (bufferSize != getSize())
	throw std::runtime_error("Wrong buffer size to write "
				 "to buffer in VectorParams<LeafVector>::"
				 "writeToBuffer.");
      char * p = dataBuffer;
      memcpy(p, &N, sizeof(int));
      p += sizeof(int);
      memcpy(p, &leavesSizeMax, sizeof(int));
    }
  
  template<class LeafVector>
    void Vector<LeafVector>::VectorParams::assignFromBuffer ( char const * dataBuffer, 
							      size_t const bufferSize) {
    if(bufferSize  != getSize())
      throw std::runtime_error("Wrong buffer size to assign "
			       "from buffer in VectorParams.");
    char const * p = dataBuffer;
    memcpy(&N, p, sizeof(int));
    p += sizeof(int);
    memcpy(&leavesSizeMax, p, sizeof(int));
  }
  
  
  template<class LeafVector>
    size_t Vector<LeafVector>::VectorParams::getSize() const 
    {
      return 2*sizeof(int);
    }
  
  
  
  template<class LeafVector>
    size_t Vector<LeafVector>::VectorParams::memoryUsage() const 
    {
      return getSize();
    }
  
} // end of namespace

#endif


