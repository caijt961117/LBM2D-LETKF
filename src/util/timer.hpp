// (C) Yuta Hasegawa
//     2018: Aoki Laboratory in Tokyo Tech
//     2019-2022: CCSE-Kashiwa in JAEA

#ifndef UTIL_TIMER_HPP_
#define UTIL_TIMER_HPP_
#include <sys/time.h>
#include <functional>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include "runtime_error.hpp"

namespace util {
class timer {
private: 
    std::map<std::string, double> map;
    std::pair<std::string, double> current;
    const std::function<double(void)> gettime;

public:
    timer(std::function<double(void)> gettime = gettime_gettimeofday): 
        map(),
        current({"_uninitialized"}, gettime()),
        gettime(gettime) { }

public: // timer operators
    void clear() { map.clear(); }

    void transit(std::string label) {
        auto&& label_last = current.first;
        auto&& time_begin = current.second;
        auto&& time_now = gettime();
        // save current timer
        if(map.find(label_last) == map.end()) { map[label_last] = 0; }
        map[label_last] += time_now - time_begin;
        // start new timer
        current = std::make_pair(label, time_now);
    }

    void start(std::string label) {
        runtime_assert(current.first[0] == '_', "APIerror: timer::start() is not available when the current label is set. Use timer::transit() instead.");
        transit(label);
    }

    void stop_and_ignore_latter() { transit("_ignored"); }

public: // compond outputs
    void showall(std::string line_prefix="", bool show_ignored=false) const {
        long double total = 0.;
        for( auto m : map ) {
            if(!show_ignored && m.first[0] == '_') { continue; }
            total += m.second;
        }
        for( auto m : map ) {
            if(!show_ignored && m.first[0] == '_') { continue; }
            std::cout << line_prefix << m.first << ": " 
                << m.second << " sec" 
                << " (" << m.second/total*100. << "%)"
                << std::endl;
        }
    }

    void fout(std::string filename) const {
        std::ofstream ofs(filename);
        ofs << "#tag,sec" << std::endl;
        for( auto m : map ) {
            ofs << m.first << ',' << m.second << std::endl;
        }
    }

public: // indivisual access
    double operator[](const std::string& s) {
        return map[s];
    }

public: // helper function 
    static double gettime_gettimeofday()  { 
      timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec + 1e-6 * tv.tv_usec;
    }
};

} // namespace

#endif // ifndef UTIL_TIMER_HPP_
