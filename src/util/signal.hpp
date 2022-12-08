#ifndef UTIL_SIGNAL_HPP_
#define UTIL_SIGNAL_HPP_

#include <signal.h>

namespace util {
class signal {
  private: static volatile sig_atomic_t signal_e_flag;
  public: signal(int sig=SIGINT) { signal_e_flag = 0; ::signal(sig, handler); }
  public: bool operator!() const { return !signal_e_flag; }
  private: static void handler(int i) { signal_e_flag = 1; }
};
volatile sig_atomic_t signal::signal_e_flag; // instantiation
}

#endif

