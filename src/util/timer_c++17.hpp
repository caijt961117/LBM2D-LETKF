// (C) Yuta Hasegawa
//     2018: Aoki Laboratory in Tokyo Tech
//     2019: CCSE-Kashiwa in JAEA

#ifndef MY_TIMER_CPP17_HPP
#define MY_TIMER_CPP17_HPP
#include <sys/time.h>
#include <functional>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <initializer_list>
#include <tuple>

namespace util {
class timer {
  private: std::map<std::string, double> map;
  public: const std::function<double(void)> gettime;
  
  public: timer(std::function<double(void)> gettime = gettime_gettimeofday): gettime(gettime) { }

  public: template<typename F, class... As> void elapse(std::string const& tag, F&& func, As&&... args) {
    const double tmp = gettime();
    func(args...);
    const double t_elapsed = gettime() - tmp;
    map[tag] += t_elapsed; 
  }

  public: void showall(std::string const& line_prefix="") const {
     long double total = 0.;
     for( auto m : map ) {
       total += m.second;
     }
     for( auto m : map ) {
       std::cout << line_prefix << m.first << ": " 
         << m.second << " sec" 
         << " (" << m.second/total*100. << "%)"
         << std::endl;
     }
  }

  public: void fout(std::string const& file) const {
    std::ofstream ofs(file);
    ofs << "#tag,sec" << std::endl;
    for( auto m : map ) {
      ofs << m.first << ',' << m.second << std::endl;
    }
  }

  public: double operator[](std::string const& s) {
    return map[s];
  }

  private: static double gettime_gettimeofday()  { 
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
  }
};
} // namespace

#endif // ifndef MY_TIMER_H
