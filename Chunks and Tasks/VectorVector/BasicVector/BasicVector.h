#ifndef BASIC_VECTOR_HEADER
#define BASIC_VECTOR_HEADER

#include "ChunkBasic.h"
#include "ChunkVector.h"

template<typename T>
class BasicVector : public chttl::ChunkVector<T>
{
public:
    typedef typename std::vector<T>::size_type size_type;
    typedef T real;

    BasicVector(std::vector<T> const & x_) : chttl::ChunkVector<T>(x_) { }
    explicit BasicVector(size_type s, const T& value = T()) : chttl::ChunkVector<T>(s,value) { }
    BasicVector() {}
    BasicVector<T>& operator=(std::vector<T> const & other) {
      chttl::ChunkVector<T>::operator=(other);
      return *this;
    }

    static void ReturnSTDVector(BasicVector<T> const &, std::vector<real> &);
    static void update(BasicVector<T> const &, BasicVector<T> const &, chttl::Basic<T> const &, BasicVector<T> &);
    static void dotProduct(BasicVector<T> const &, BasicVector<T> const &, real &);

};


template<typename T>
void BasicVector<T>::update(BasicVector<T> const & x, BasicVector<T> const & y, chttl::Basic<T> const & alpha, BasicVector<T> & z)
{
  z.clear();
  z.resize(x.size()); 
  for (int ind = 0; ind < x.size(); ind++)
    z[ind] = x[ind] + alpha*y[ind];
}

template<typename T>
void BasicVector<T>::ReturnSTDVector(BasicVector<T> const & x, std::vector<real> & vec)
{
  vec = x;
}

template<typename T>
void BasicVector<T>::dotProduct(BasicVector<T> const & x, BasicVector<T> const & y, real & result)
{
  result = 0;
  for (int ind = 0; ind < x.size(); ind++)
    result += x[ind]*y[ind];
}


#endif
