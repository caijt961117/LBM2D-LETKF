// http://www-nlb.loni.ucla.edu/twiki/bin/view/MAST/VtkAppendedData?skin=plain
// g++ -lz
//

#ifndef ZLIB_ENCODER_HPP_
#define ZLIB_ENCODER_HPP_

#include <zlib.h>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include "runtime_error.hpp"

namespace util {

template<typename header_type = std::uintptr_t> class ZlibEncoder {
public: 
using byte_type = std::uint8_t;
constexpr static header_type UNCOMPRESSED_BLOCK_SIZE = 65536;

private:
header_type numberOfBlocks;
const header_type uncompressedBlockSize;
header_type uncompressedLastBlockSize;
std::vector<header_type> compressedBlockSizes;
std::vector<byte_type> uncompressedData;
std::vector<byte_type> compressedData;
  
public:
ZlibEncoder(): uncompressedBlockSize(UNCOMPRESSED_BLOCK_SIZE) {}

void PushArray(const void* dat, const std::intptr_t& size) {
  if(!compressedData.empty()) { throw STD_RUNTIME_ERROR("do not push after encode"); }
  for(std::intptr_t i=0; i<size; i++) {
    uncompressedData.push_back(reinterpret_cast<const byte_type*>(dat)[i]); 
  } 
}

template<typename S, typename T> void PushOne(const T& t) {
  const S s = static_cast<S>(t);
  PushArray(&s, sizeof(S));
}

void Encode() {
  //if(uncompressedData.empty()) { throw STD_RUNTIME_ERROR("do not encode before push(es)"); }
  if(!compressedData.empty()) { throw STD_RUNTIME_ERROR("do not encode more than once"); }
  compressedData.reserve(uncompressedData.size()/2);
  std::vector<byte_type> bufenc(compressBound(uncompressedBlockSize));
  const std::intptr_t iter  = uncompressedData.size() / uncompressedBlockSize;
  uncompressedLastBlockSize = uncompressedData.size() % uncompressedBlockSize;
  numberOfBlocks = iter + 1;
  compressedBlockSizes.resize(numberOfBlocks);
  for(std::intptr_t i=0; i<=iter; i++) {
    uLong sizenc = bufenc.size();
    const uLong sizeorig = i<iter ? uncompressedBlockSize : uncompressedLastBlockSize;
    const auto zerr = compress(&bufenc[0], &sizenc, uncompressedData.data() + i*uncompressedBlockSize, sizeorig);
    if(zerr != Z_OK) {
      std::string msg;
      if(zerr == Z_MEM_ERROR) { msg += "Z_MEM_ERROR"; }
      if(zerr == Z_BUF_ERROR) { msg += "Z_BUF_ERROR"; }
      throw STD_RUNTIME_ERROR(msg);
    }
    compressedBlockSizes.at(i) = sizenc;
    for(std::intptr_t j=0; j<std::intptr_t(sizenc); j++) {
        compressedData.push_back(bufenc.at(j));
    }
  }
  uncompressedData.clear();
  //std::cout << " compression: " << long(DataSize()) << "/" << long(UncompressedSize()) << std::endl;
}

header_type DataSize() const { 
  //if(compressedData.empty()) { throw STD_RUNTIME_ERROR("cannot stat Size before encoding"); }
  return compressedData.size();
}

header_type HeaderSize() const {
  //if(compressedData.empty()) { throw STD_RUNTIME_ERROR("cannot stat Size before encoding"); }
  return sizeof(header_type) * (3 + numberOfBlocks);
}

header_type TotalSize() const {
    return HeaderSize() + DataSize();
}

header_type UncompressedSize() const {
  if(compressedData.empty()) { throw STD_RUNTIME_ERROR("cannot stat Size before encoding"); }
  return (numberOfBlocks-1) * uncompressedBlockSize + uncompressedLastBlockSize;
}

void WriteEncodedDataTo(std::FILE* fp) const  {
  safe_fwrite(&numberOfBlocks           , sizeof(header_type), 1, fp);
  safe_fwrite(&uncompressedBlockSize    , sizeof(header_type), 1, fp);
  safe_fwrite(&uncompressedLastBlockSize, sizeof(header_type), 1, fp);
  safe_fwrite(&compressedBlockSizes[0], sizeof(header_type)*numberOfBlocks, 1, fp);
  safe_fwrite(&compressedData[0], compressedData.size(), 1, fp);
}

std::vector<byte_type> GetEncodedData() const {
    std::vector<byte_type> ret;
    auto&& append = [&ret](const header_type& val) {
        const byte_type* p = reinterpret_cast<const byte_type*>(&val);
        //// little endian
        // for(int i=sizeof(header_type)-1; i>=0; i--) {
        // big endian
        for(int i=0; i<int(sizeof(header_type)); i++) {
            ret.push_back(p[i]);
        }
    };
    append(numberOfBlocks);
    append(uncompressedBlockSize);
    append(uncompressedLastBlockSize);
    for(const header_type& compressedBlockSize_j: compressedBlockSizes) {
        append(compressedBlockSize_j);
    }
    ret.insert(ret.end(), compressedData.begin(), compressedData.end()); 
    assert(ret.size() == TotalSize());
    return ret;
}

private:
static void safe_fwrite(const void* ptr, size_t size, size_t count, std::FILE* stream) {
  const size_t count_result = std::fwrite(ptr, size, count, stream);

  if(count_result != count) {
    throw STD_RUNTIME_ERROR("could not write file"); 
  }
}

};

}

#endif

