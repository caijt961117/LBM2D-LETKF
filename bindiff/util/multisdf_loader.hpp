#ifndef MULTISDF_LOADER_H
#define MULTISDF_LOADER_H
#include "sdf_loader.hpp"
#include <cmath>
#include <vector>
#include <string>
#include "runtime_error.hpp"

namespace util {

class multisdf_loader {
  private:
    using real = sdf_loader::real;
    using geo = sdf_loader::geo;
    using boundingbox = sdf_loader::boundingbox;
  private:
    std::vector<sdf_loader> sdfs;
    boundingbox bb;
  public:
  multisdf_loader(const std::vector<std::string>& files, const real& unit = real(1), const bool& verbose = false) {
    load(files, unit, verbose);
  }
  void load(const std::vector<std::string>& files, const real unit, const bool verbose) {
    if(files.empty()) { throw STD_RUNTIME_ERROR("no sdf files indicated"); }
    sdfs.resize(files.size());
    bb.west = 1e30;
    bb.east = -1e30;
    bb.south = 1e30;
    bb.north = -1e30;
    bb.bottom = 1e30;
    bb.top = -1e30;
    for(int i=0; i<int(files.size()); i++) {
      sdfs.at(i).load(files.at(i), unit, verbose);
      bb.west   = std::min(bb.west  , sdfs.at(i).BB().west  );
      bb.east   = std::max(bb.east  , sdfs.at(i).BB().east  );
      bb.south  = std::min(bb.south , sdfs.at(i).BB().south );
      bb.north  = std::max(bb.north , sdfs.at(i).BB().north );
      bb.bottom = std::min(bb.bottom, sdfs.at(i).BB().bottom);
      bb.top    = std::max(bb.top   , sdfs.at(i).BB().top   );
    }
  }
  public: // backward compatibility
  geo LX() const { return bb.east-bb.west; }
  geo LY() const { return bb.north-bb.south; }
  geo LZ() const { return bb.top-bb.bottom; }
  geo OX() const { return bb.west; }
  geo OY() const { return bb.south; }
  geo OZ() const { return bb.bottom; }
  boundingbox BB() const { return bb; }
  real interpolate(const geo& x, const geo& y, const geo& z) const {
    real itp = sdfs.at(0).interpolate(x, y, z);
    for(int i=0; i<int(sdfs.size()); i++) {
      itp = std::min(itp, sdfs.at(i).interpolate(x, y, z));
    }
    return itp;
  }
  int inearest(const geo& x, const geo& y, const geo& z) const {
    real tar = sdfs.at(0).interpolate(x, y, z);
    int number = 0;
    for(int i=0; i<int(sdfs.size()); i++) {
      const real itp = sdfs.at(i).interpolate(x, y, z);
      if(itp < tar) { number = i; tar = itp; }
    }
    return number;
  }
  real projection_z_area_of_first_sdf() const {
    return sdfs.at(0).projection_z_area();
  }
};

}

#endif

