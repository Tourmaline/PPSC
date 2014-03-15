#include "Vector.h"
#include "tasks_vector.h"
#include "VectorAssignFromChunkIDs.h"

#define CHTVEC_REGISTER_CHUNK_TYPE(chunktype)					\
  namespace chtvec {							\
    CHT_CHUNK_TYPE_SPECIALIZATION_IMPLEMENTATION(chunktype);		\
  }
#define CHTVEC_REGISTER_TASK_TYPE(tasktype)					\
  namespace chtvec {							\
    CHT_TASK_TYPE_SPECIALIZATION_IMPLEMENTATION(tasktype);		\
  }
