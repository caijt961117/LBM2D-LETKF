// Python-like range for C++
// from: https://yuki67.github.io/post/python_like_range/

#ifndef RANGE_HPP_
#define RANGE_HPP_

namespace util {

// helper iterator class
template <typename T>
class range_iterator {
public: // types
   using value_type = T;
   using iterator_type = range_iterator<T>;
private: // variables
   value_type cur_;
   const value_type end_;
   const value_type inc_;
public: // range-based iterator adapter
   range_iterator(T begin, T end, T inc)
       : cur_(begin), end_(end), inc_(inc) {}
   void operator++() { cur_ += inc_; }
   value_type operator*() const { return cur_; }
   iterator_type begin() const { return *this; }
   iterator_type end() const { return *this; }
   bool operator!=(const iterator_type& rhs) const {
       return inc_ > 0
           ? cur_ < rhs.end_
           : cur_ > rhs.end_;
   }
public: // for cartesian_product extension
   void operator=(const iterator_type& rhs) { cur_ = rhs.cur_; } // note: end_, inc_ should be equal
}; // end class range_iterator

// implement
template<typename T> auto range(T begin, T end, T stride=T(1)) { return range_iterator<T>(begin, end, stride); }
template<typename T> auto range(T n) { return range(T(0), n, T(1)); }

// cartisian product of range
template<typename Tr1, typename Tr2> // Tr1, Tr2 should be range_iterator<T>
class range2d_iterator {
public: // types
   using value_type = std::pair<typename Tr1::value_type, typename Tr2::value_type>;
   using ranges_type = std::pair<Tr1, Tr2>;
   using iterator_type = range2d_iterator<Tr1, Tr2>;
private: // variables
   ranges_type cur_;
   const ranges_type begin_;
   const ranges_type end_;
public: // range-based iterator adapter
   range2d_iterator(Tr1 range_y, Tr2 range_x):
       cur_( { range_y.begin(), range_x.begin() } ),
       begin_( { range_y.begin(), range_x.begin() } ),
       end_( { range_y.end(), range_x.end() } )
       {}
   value_type operator*() const { return { *cur_.first, *cur_.second }; }
   iterator_type begin() const { return *this; }
   iterator_type end() const { return *this; }
   bool operator!=(const iterator_type& rhs) const { return cur_.first != rhs.end_.first; }
   void operator++() {
       ++ cur_.second;
       if(not (cur_.second != end_.second)) {
           cur_.second = begin_.second;
           ++ cur_.first;
       }
   }
};
template<typename Tr1, typename Tr2> auto cartesian_product(Tr1 range_y, Tr2 range_x) { return range2d_iterator<Tr1, Tr2>(range_y, range_x); }
template<typename T> auto range2d(T ny, T nx) { return cartesian_product(util::range(ny), util::range(nx)); }

} // end namespace util
#endif
