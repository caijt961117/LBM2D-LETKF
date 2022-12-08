// (C) Yuta Hasegawa
//     2018: Aoki Laboratory in Tokyo Tech
//     2019: CCSE-Kashiwa in JAEA

#ifndef MY_TIMER_HPP
#define MY_TIMER_HPP
#include <sys/time.h>
#include <functional>
#include <map>
#include <string>
#include <fstream>
#include <vector>
#include <initializer_list>

namespace util {
class timer {
  private: std::map<std::string, double> map;
  public: const std::function<double(void)> gettime;
  
  public: timer(std::function<double(void)> gettime = gettime_gettimeofday): gettime(gettime) { }

  public: template<typename F> void elapse(std::vector<std::string> tags, F func) {
    const double tmp = gettime();
    func();
    const double t_elapsed = gettime() - tmp;
    for ( std::string tag : tags ) { map[tag] += t_elapsed; }
  }

  public: template<typename F> void elapse(std::initializer_list<std::string> tags, F func) {
    elapse(std::vector<std::string>(tags), func);
  }
  public: template<typename F> void elapse(std::string tag, F func) {
     elapse({tag}, func);
  }

  public: void showall(std::string line_prefix="") const {
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

  public: void fout(std::string file) const {
    std::ofstream ofs(file);
    ofs << "#tag,sec" << std::endl;
    for( auto m : map ) {
      ofs << m.first << ',' << m.second << std::endl;
    }
  }

  public: double operator[](const std::string& s) {
    return map[s];
  }

  public: static double gettime_gettimeofday()  { 
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
  }
};
} // namespace

#endif // ifndef MY_TIMER_H
