#ifndef TEST_TOOLS_HEADER
#define TEST_TOOLS_HEADER

// Utility to set block size (a block sparse matrix lib specific
// parameter) without breaking compatibility with other leaf matrix
// libraries

template<typename MatrixType>
void set_block_size(typename MatrixType::Params & param, int blocksize) {}
template<>
void set_block_size<bsm::BlockSparseMatrix<double> >(bsm::BlockSparseMatrix<double>::Params & param, int blocksize) {
  param.blocksize = blocksize;
}
template<>
void set_block_size<bsm::BlockSparseMatrix<float> >(bsm::BlockSparseMatrix<float>::Params & param, int blocksize) {
  param.blocksize = blocksize;
}

template<typename T>
cht::ChunkID get_vector(std::vector<T> const & vec) {
  // Assign elements
  return  cht::registerChunk<chttl::ChunkVector<T> >(new chttl::ChunkVector<T>(vec));
}

template<typename T_LeafMatrixType>
cht::ChunkID get_matrix_from_vectors(int M, int N, int leavesSizeMax, int blocksize,
				     std::vector<int> const & row,
				     std::vector<int> const & col,
				     std::vector<typename T_LeafMatrixType::real> const & val) {
  typename T_LeafMatrixType::Params leaf_params;
  set_block_size<T_LeafMatrixType>(leaf_params, blocksize);
  cht::ChunkID cid_param = cht::registerChunk<chtml::MatrixParams<T_LeafMatrixType> >(new chtml::MatrixParams<T_LeafMatrixType>(M, N, leavesSizeMax, 0, 0, leaf_params));
  
  // Assign elements
  cht::ChunkID cid_row = get_vector(row);
  cht::ChunkID cid_col = get_vector(col);
  cht::ChunkID cid_val = get_vector(val);
  cht::ChunkID cid_matrix = 
    cht::executeMotherTask<chtml::MatrixAssignFromSparse<T_LeafMatrixType> >
    (cid_param, cid_row, cid_col, cid_val);
  cht::deleteChunk(cid_param);
  cht::deleteChunk(cid_row);
  cht::deleteChunk(cid_col);
  cht::deleteChunk(cid_val);
  return cid_matrix;
}

template<typename T_LeafMatrixType>
cht::ChunkID get_matrix(int M, int N, int leavesSizeMax, int blocksize, typename T_LeafMatrixType::real valvec[]) {
  std::vector<int>  row(M*N);
  std::vector<int>  col(M*N);
  std::vector<typename T_LeafMatrixType::real> val(M*N);
  int count = 0;
  int count2 = 0;
  for (int r = 0; r<M; r++)
    for (int c = 0; c<N; c++) {
      if (valvec[count2] != 0) {
	row[count] = r;
	col[count] = c;
	val[count] = valvec[count2];
	count++;
      }
      count2++;
    }
  row.resize(count);
  col.resize(count);
  val.resize(count);
  //  std::vector<typename T_LeafMatrixType::real> val;
  //  val.insert(val.end(), valvec, valvec+M*N);
  return get_matrix_from_vectors<T_LeafMatrixType>(M, N, leavesSizeMax, blocksize, row, col, val);
}

template<typename T_LeafMatrixType>
void print_matrix(cht::ChunkID cid_matrix, std::string id_str, int M, int N, int leavesSizeMax) {
  cht::ChunkID cid_param = cht::registerChunk<chtml::MatrixParams<T_LeafMatrixType> >(new chtml::MatrixParams<T_LeafMatrixType>(M, N, leavesSizeMax, 0, 0));
  std::vector<int>  row(M*N);
  std::vector<int>  col(M*N);
  int count = 0;
  for (int r = 0; r<M; r++)
    for (int c = 0; c<N; c++) {
      row[count] = r;
      col[count] = c;
      count++;
    }
  cht::ChunkID cid_row = get_vector(row);
  cht::ChunkID cid_col = get_vector(col);  
  cht::ChunkID cid_val = 
    cht::executeMotherTask<chtml::MatrixGetElements<T_LeafMatrixType> >(cid_param,
									cid_row,
									cid_col,
									cid_matrix);
  cht::shared_ptr<chttl::ChunkVector<typename T_LeafMatrixType::real> const> values;
  cht::getChunk(cid_val, values);
  count = 0;
  std::cout << id_str << " = [" <<std::endl;
  for (int r = 0; r<M; r++) {
    for (int c = 0; c<N; c++) {
      std::cout << (*values)[count] << ", ";
      count++;
    } 
    std::cout << std::endl;
  } 
  std::cout << "]" <<std::endl;
  cht::deleteChunk(cid_param);
  cht::deleteChunk(cid_row);
  cht::deleteChunk(cid_col);
  cht::deleteChunk(cid_val);
}

template<typename Treal>
struct CompareStrict {
  static bool check(Treal const a, Treal const b, Treal thr = 0) {
    return a==b;
  }
};

template<typename Treal>
struct CompareThreshold {
  static bool check(Treal const a, Treal const b, Treal thr) {
    return std::fabs(a-b) < thr;
  }
};

template<typename T_LeafMatrixType, template<typename Treal>  class T_Compare >
void check_matrix_values( cht::ChunkID cid_matrix, int M, int N, int leavesSizeMax, 
			  typename T_LeafMatrixType::real ref_values[], 
			  typename T_LeafMatrixType::real threshold_check = 0) {
  cht::ChunkID cid_param = cht::registerChunk<chtml::MatrixParams<T_LeafMatrixType> >(new chtml::MatrixParams<T_LeafMatrixType>(M, N, leavesSizeMax, 0, 0));
  std::vector<int>  row(M*N);
  std::vector<int>  col(M*N);
  int count = 0;
  for (int r = 0; r<M; r++)
    for (int c = 0; c<N; c++) {
      row[count] = r;
      col[count] = c;
      count++;
    }
  cht::ChunkID cid_row = get_vector(row);
  cht::ChunkID cid_col = get_vector(col);  
  cht::ChunkID cid_val = 
    cht::executeMotherTask<chtml::MatrixGetElements<T_LeafMatrixType> >(cid_param,
								      cid_row,
								      cid_col,
								      cid_matrix);
  cht::shared_ptr<chttl::ChunkVector<typename T_LeafMatrixType::real> const> values;
  cht::getChunk(cid_val, values);
  count = 0;
  for (int r = 0; r<M; r++) 
    for (int c = 0; c<N; c++) {
      assert( T_Compare<typename T_LeafMatrixType::real>::check((*values)[count], ref_values[count], threshold_check) );
      //      assert((*values)[count] == ref_values[count]);
      count++;
    } 
  cht::deleteChunk(cid_param);
  cht::deleteChunk(cid_row);
  cht::deleteChunk(cid_col);
  cht::deleteChunk(cid_val);
}

#endif
