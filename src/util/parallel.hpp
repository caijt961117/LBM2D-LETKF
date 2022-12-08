#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <thread>
#include <vector>

namespace util {
class parallel {
  const int nt;
  private: parallel();
  public: parallel(const int& nt) noexcept: nt(nt) {}
  public: template<typename F> void work(F&& f) {
            std::vector<std::thread> t;
            for(int i=0; i<nt; i++) { t.push_back(std::thread(f, i)); }
            for(int i=0; i<nt; i++) { t[i].join(); }
          }
};
}

#endif

