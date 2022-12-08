// sdf_loader
// Loading levelset data made by stl2ls (prof. aoki fmt)
//
// origin izumida
// edit hasegawa
#ifndef SDF_LOADER_H
#define SDF_LOADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdio>
#include <stdint.h>
#include <cmath>
#include "runtime_error.hpp"

namespace util {

class sdf_loader {
public: 
  typedef int32_t zahl;
  typedef float real;
  typedef double geo;
  typedef std::basic_string<char> string;
  typedef std::vector<real> array;
  struct boundingbox { geo west, east, south, north, bottom, top; };

private: // stored objects
  zahl nx, ny, nz;
  geo dx, dy, dz;
  geo lx, ly, lz;
  geo ox, oy, oz;
  boundingbox bb;
  array sdf;
  real smax, smin;
public:
  zahl NX() const { return nx; }
  zahl NY() const { return ny; }
  zahl NZ() const { return nz; }
  geo LX() const { return lx; }
  geo LY() const { return ly; }
  geo LZ() const { return lz; }
  geo DX() const { return dx; }
  geo DY() const { return dy; }
  geo DZ() const { return dz; }
  geo OX() const { return ox; }
  geo OY() const { return oy; }
  geo OZ() const { return oz; }
  boundingbox BB() const { return bb; }

public: //constructor
  sdf_loader() { sdf.resize(0); }
  sdf_loader(const std::string& filename, const real& unit = real(1), const bool& verbose = false) {
    load(filename, unit, verbose);
  }
  void load(const std::string& filename, const real& unit, const bool& verbose) {
    if(sdf.size() != 0) { throw STD_RUNTIME_ERROR("do not relead sdf"); }
    std::ifstream file(filename.c_str(), std::ifstream::binary);
    if(!file) { throw STD_RUNTIME_ERROR("could not open file: " + filename); }

    string version(8, '\0');
    file.read(reinterpret_cast<char*>(&version[0]), 7);

    zahl nlen;
    file.read(reinterpret_cast<char*>(&nlen), sizeof(zahl));

    string comment(nlen+1, '\0');
    file.read(reinterpret_cast<char*>(&comment[0]), nlen*sizeof(char));

    file.read(reinterpret_cast<char*>(&nx), sizeof(zahl));
    file.read(reinterpret_cast<char*>(&ny), sizeof(zahl));
    file.read(reinterpret_cast<char*>(&nz), sizeof(zahl));
    file.read(reinterpret_cast<char*>(&dx), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&dy), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&dz), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&lx), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&ly), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&lz), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&ox), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&oy), sizeof(geo ));
    file.read(reinterpret_cast<char*>(&oz), sizeof(geo ));
    bb.west = ox;
    bb.east = ox + lx;
    bb.south = oy;
    bb.north = oy + ly;
    bb.bottom = oz;
    bb.top = oz + lz;

    char blen;
    file.read(&blen, sizeof(char));
    if((int)(blen-'0') != sizeof(real)) {
      throw STD_RUNTIME_ERROR(std::string("sdf: loading failed: fatal byte length of FP ") + blen);
    }

    const int n = nx*ny*nz;
    try { 
      sdf.resize(n);
    } catch(const std::exception& e) {
      throw STD_RUNTIME_ERROR(e.what());
    }
    file.read(reinterpret_cast<char*>(sdf.data()), sizeof(real)*n);

    smin = sdf[0], smax = sdf[0];
    for(long i=1; i<long(sdf.size()); i++) {
      smax = std::max(smax, sdf[i]);
      smin = std::min(smin, sdf[i]);
    }

    if(verbose) {
      std::cout << "input file: " << filename << std::endl;
      std::cout << "    version: " << version << std::endl;
      std::cout << "    comment: " << comment << std::endl;
      std::cout << "    mesh: [" << nx << " x " << ny << " x " << nz << "]" << std::endl;
      std::cout << "    dx:   [" << dx << " x " << dy << " x " << dz << "]" << std::endl;
      std::cout << "    lx:   [" << lx << " x " << ly << " x " << lz << "]" << std::endl;
      std::cout << "    xoff: [" << ox << " x " << oy << " x " << oz << "]" << std::endl;
      std::cout << "    FPbyte: " << blen << std::endl;
      std::cout << "    sdf(raw): " << smin << " -- " << smax << std::endl;
    }
    if(unit != real(1)) {
      dx *= unit;
      dy *= unit;
      dz *= unit;
      lx *= unit;
      ly *= unit;
      lz *= unit;
      ox *= unit;
      oy *= unit;
      oz *= unit;
      for(long i=0; i<long(sdf.size()); i++) { sdf[i] *= unit; }
      smin = sdf[0], smax = sdf[0];
      for(long i=1; i<long(sdf.size()); i++) {
        smax = std::max(smax, sdf[i]);
        smin = std::min(smin, sdf[i]);
      }
      bb.west = ox;
      bb.east = ox + lx;
      bb.south = oy;
      bb.north = oy + ly;
      bb.bottom = oz;
      bb.top = oz + lz;
      if(verbose) {
        std::cout << "  unit modifying factor: " << unit << std::endl;
        std::cout << "    mesh: [" << nx << " x " << ny << " x " << nz << "]" << std::endl;
        std::cout << "    dx:   [" << dx << " x " << dy << " x " << dz << "]" << std::endl;
        std::cout << "    lx:   [" << lx << " x " << ly << " x " << lz << "]" << std::endl;
        std::cout << "    xoff: [" << ox << " x " << oy << " x " << oz << "]" << std::endl;
        std::cout << "    sdf(meter): " << smin << " -- " << smax << std::endl;
      }
    }

  }

public: 
  // trilinear interpolation by given coordinate
  real interpolate(const geo& x, const geo& y, const geo& z) const {
    const geo i_f = (x-ox)/dx;
    const geo j_f = (y-oy)/dy;
    const geo k_f = (z-oz)/dz;
    const int i0 = int(i_f);
    const int j0 = int(j_f);
    const int k0 = int(k_f);
    const int i1 = i0 + 1;
    const int j1 = j0 + 1;
    const int k1 = k0 + 1;
    if(0<=i0 && i1<nx && 0<=j0 && j1<ny && 0<=k0 && k1<nz) {
      const geo xd = i_f - geo(i0);
      const geo yd = j_f - geo(j0);
      const geo zd = k_f - geo(k0);
      return (
         +(
           +( (geo(1)-xd) * (geo(1)-yd) * (geo(1)-zd) * sdf.at(i0 + j0*nx + k0*nx*ny) + xd * (geo(1)-yd) * (geo(1)-zd) * sdf.at(i1 + j0*nx + k0*nx*ny))
           +( (geo(1)-xd) * (       yd) * (geo(1)-zd) * sdf.at(i0 + j1*nx + k0*nx*ny) + xd * (       yd) * (geo(1)-zd) * sdf.at(i1 + j1*nx + k0*nx*ny))
          )
         +(
           +( (geo(1)-xd) * (geo(1)-yd) * (       zd) * sdf.at(i0 + j0*nx + k1*nx*ny) + xd * (geo(1)-yd) * (       zd) * sdf.at(i1 + j0*nx + k1*nx*ny))
           +( (geo(1)-xd) * (       yd) * (       zd) * sdf.at(i0 + j1*nx + k1*nx*ny) + xd * (       yd) * (       zd) * sdf.at(i1 + j1*nx + k1*nx*ny))
          )
        );
    } else {
      return smax + geo(9);
    }
  }

  // estimate projected sectional area
  real projection_z_area() const {
    zahl count = 0;
    for(zahl j=0; j<ny; j++) {
      for(zahl i=0; i<nx; i++) {
        for(zahl k=0; k<nz; k++) {
          const zahl ijk = i + nx*j + nx*ny*k;
          if(sdf.at(ijk) < 0) {
            ++count;
            break; // break z-loop and next xy
          }
        }
      }
    }
    return count * dx * dy;
  }


};

} // namespace flow

#endif // SDF_LOADER_H

