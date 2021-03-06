/**
 * Copyright (c) 2008-2016, Wojciech Muła
 * Copyright (c) 2016, Kim Walisch
 * Copyright (c) 2016, Dan Luu
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef POPCNT_AVX2_HARLEY_SEAL_HPP_
#define POPCNT_AVX2_HARLEY_SEAL_HPP_

namespace AVX2_harley_seal {

__m256i popcount(const __m256i v)
{
  const __m256i m1 = _mm256_set1_epi8(0x55);
  const __m256i m2 = _mm256_set1_epi8(0x33);
  const __m256i m4 = _mm256_set1_epi8(0x0F);

  const __m256i t1 = _mm256_sub_epi8(v,       (_mm256_srli_epi16(v,  1) & m1));
  const __m256i t2 = _mm256_add_epi8(t1 & m2, (_mm256_srli_epi16(t1, 2) & m2));
  const __m256i t3 = _mm256_add_epi8(t2, _mm256_srli_epi16(t2, 4)) & m4;
  return _mm256_sad_epu8(t3, _mm256_setzero_si256());
}

void CSA(__m256i& h, __m256i& l, __m256i a, __m256i b, __m256i c)
{
  const __m256i u = a ^ b;
  h = (a & b) | (u & c);
  l = u ^ c;

}

uint64_t popcnt(const __m256i* data, const uint64_t size)
{
  __m256i total     = _mm256_setzero_si256();
  __m256i ones      = _mm256_setzero_si256();
  __m256i twos      = _mm256_setzero_si256();
  __m256i fours     = _mm256_setzero_si256();
  __m256i eights    = _mm256_setzero_si256();
  __m256i sixteens  = _mm256_setzero_si256();
  __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  for(; i < limit; i += 16)
  {
    CSA(twosA, ones, ones, data[i+0], data[i+1]);
    CSA(twosB, ones, ones, data[i+2], data[i+3]);
    CSA(foursA, twos, twos, twosA, twosB);
    CSA(twosA, ones, ones, data[i+4], data[i+5]);
    CSA(twosB, ones, ones, data[i+6], data[i+7]);
    CSA(foursB, twos, twos, twosA, twosB);
    CSA(eightsA,fours, fours, foursA, foursB);
    CSA(twosA, ones, ones, data[i+8], data[i+9]);
    CSA(twosB, ones, ones, data[i+10], data[i+11]);
    CSA(foursA, twos, twos, twosA, twosB);
    CSA(twosA, ones, ones, data[i+12], data[i+13]);
    CSA(twosB, ones, ones, data[i+14], data[i+15]);
    CSA(foursB, twos, twos, twosA, twosB);
    CSA(eightsB, fours, fours, foursA, foursB);
    CSA(sixteens, eights, eights, eightsA, eightsB);

    total = _mm256_add_epi64(total, popcount(sixteens));
  }

  total = _mm256_slli_epi64(total, 4);     // * 16
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(eights), 3)); // += 8 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(fours),  2)); // += 4 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(twos),   1)); // += 2 * ...
  total = _mm256_add_epi64(total, popcount(ones));

  for(; i < size; i++)
    total = _mm256_add_epi64(total, popcount(data[i]));


  return static_cast<uint64_t>(_mm256_extract_epi64(total, 0))
       + static_cast<uint64_t>(_mm256_extract_epi64(total, 1))
       + static_cast<uint64_t>(_mm256_extract_epi64(total, 2))
       + static_cast<uint64_t>(_mm256_extract_epi64(total, 3));
}

} // AVX2_harley_seal

uint64_t popcnt_AVX2_harley_seal(const uint8_t* data, const size_t size)
{
  uint64_t total = AVX2_harley_seal::popcnt((const __m256i*) data, size / 32);

  for (size_t i = size - size % 32; i < size; i++)
    total += lookup8bit[data[i]];

  return total;
}

#endif // POPCNT_AVX2_HARLEY_SEAL_HPP_
