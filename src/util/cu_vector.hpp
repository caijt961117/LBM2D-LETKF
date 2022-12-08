#ifndef CU_VECTOR_HPP_
#define CU_VECTOR_HPP_

#include <vector>
#include <array>
#include <iterator>

#include "cu_allocator.hpp"

namespace util {

template<class T> using cu_vector = std::vector<T, cu_managed_allocator<T>>;
template<class T, std::size_t Ny, std::size_t Nx> using cu_vec3d = std::array<std::array<cu_vector<T>, Nx>, Ny>;

// std::vector like class with runtime-const size
template<class T> class cu_ptr: protected cu_vector<T> {
  public: cu_ptr(std::size_t n): 
  cu_vector<T>(n) 
  {
    this->cu_vector<T>::shrink_to_fit();
  }

  public: ~cu_ptr() {}

  public: void swap(cu_ptr& x) {
    if(this->size() != x.size()) {
      throw std::runtime_error("cu_ptr::swap: only the same-size cu_ptr can be swaped");
    }
    this->cu_vector<T>::swap(x);
  }

  public: void reset() {
    this->cu_vector<T>::clear();
    this->cu_vector<T>::shrink_to_fit();
  }

  // prohibited or not-implemented
  private:
  cu_ptr();
  cu_ptr(const cu_ptr&);
  template<class U> cu_ptr& operator=(const U&);

  // allowed std::vector members 
  public: 
  using cu_vector<T>::operator[];
  using cu_vector<T>::at;
  using cu_vector<T>::data;
  using cu_vector<T>::front;
  using cu_vector<T>::back;
  using cu_vector<T>::begin;
  using cu_vector<T>::end;
  using cu_vector<T>::cbegin;
  using cu_vector<T>::cend;
  using cu_vector<T>::rbegin;
  using cu_vector<T>::rend;
  using cu_vector<T>::crbegin;
  using cu_vector<T>::crend;
  using cu_vector<T>::size;
  using cu_vector<T>::empty;
};

}

#endif
