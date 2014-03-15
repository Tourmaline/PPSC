#include "cht_vec_lib.h"
#include "cht_type_lib.h"
#include "VectorVector/BasicVector/BasicVector.h"
#include "cht_matrix_lib.h"
#include "basic_matrix_lib.h"
#include "block_sparse_matrix_lib.h"
#include "sparse_matrix_lib.h"

CHTTL_REGISTER_CHUNK_TYPE((chttl::Basic<double>));
CHTTL_REGISTER_CHUNK_TYPE((chttl::ChunkVector<double>));
CHTTL_REGISTER_CHUNK_TYPE((chttl::ChunkVector<int>));

CHTVEC_REGISTER_CHUNK_TYPE((chtvec::Vector<BasicVector<double> >));
CHTVEC_REGISTER_CHUNK_TYPE((chtvec::Vector<BasicVector<double> >::VectorParams));

CHTTL_REGISTER_TASK_TYPE((chttl::BasicAdd<double>));
CHTVEC_REGISTER_TASK_TYPE((chtvec::VectorAssignFromChunkIDs<BasicVector<double> >));
CHTVEC_REGISTER_TASK_TYPE((chtvec::VectorAssignFromChunkVector<BasicVector<double> >));
CHTVEC_REGISTER_TASK_TYPE((chtvec::VectorUpdate<BasicVector<double> >));
CHTVEC_REGISTER_TASK_TYPE((chtvec::CombineVectors<BasicVector<double> >));
CHTVEC_REGISTER_TASK_TYPE((chtvec::ReturnVector<BasicVector<double> >));
CHTVEC_REGISTER_TASK_TYPE((chtvec::DotProduct<BasicVector<double> >));

CHTML_REGISTER_CHUNK_TYPE((chtml::Matrix< bml::FullMatrix<double> >)); 
CHTML_REGISTER_CHUNK_TYPE((chtml::MatrixParams< bml::FullMatrix<double> >));
CHTML_REGISTER_TASK_TYPE((chtml::MatrixAssignFromChunkIDs< bml::FullMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixAssignFromSparse< bml::FullMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixAdd< bml::FullMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixCombineElements<double>));
CHTML_REGISTER_TASK_TYPE((chtml::MatrixGetElements< bml::FullMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixMultiply<bml::FullMatrix<double>, false, false>));

CHTML_REGISTER_CHUNK_TYPE((chtml::Matrix<sml::CSCMatrix<double> >)); 
CHTML_REGISTER_CHUNK_TYPE((chtml::MatrixParams<sml::CSCMatrix<double>  >));
CHTML_REGISTER_TASK_TYPE((chtml::MatrixAssignFromChunkIDs<sml::CSCMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixAssignFromSparse<sml::CSCMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixAdd<sml::CSCMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixGetElements<sml::CSCMatrix<double> >)); 
CHTML_REGISTER_TASK_TYPE((chtml::MatrixMultiply<sml::CSCMatrix<double>, false, false>));


CHTVEC_REGISTER_TASK_TYPE((chtvec::MatVecMul<bml::FullMatrix<double>, BasicVector<double> >));
CHTVEC_REGISTER_TASK_TYPE((chtvec::MatVecMul<sml::CSCMatrix<double>, BasicVector<double> >));

