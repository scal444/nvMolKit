// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/pattern_screen.h"

#include <DataStructs/ExplicitBitVect.h>

#include <boost/dynamic_bitset.hpp>
#include <iterator>
#include <stdexcept>

namespace nvMolKit {

namespace {

void fingerprintWords(const ExplicitBitVect& fingerprint, std::uint64_t (&words)[kPatternFingerprintWords]) {
  static_assert(sizeof(boost::dynamic_bitset<>::block_type) == sizeof(std::uint64_t));
  if (fingerprint.getNumBits() != kPatternFingerprintBits) {
    throw std::invalid_argument("Pattern fingerprint screening requires 2048-bit fingerprints");
  }
  boost::to_block_range(*fingerprint.dp_bits, std::begin(words));
}

}  // namespace

std::vector<std::uint64_t> packPatternFingerprintsWordMajor(const std::vector<const ExplicitBitVect*>& fingerprints) {
  const std::size_t          count = fingerprints.size();
  std::vector<std::uint64_t> packed(count * kPatternFingerprintWords);
  std::uint64_t              words[kPatternFingerprintWords];
  for (std::size_t index = 0; index < count; ++index) {
    if (fingerprints[index] == nullptr) {
      throw std::invalid_argument("Missing pattern fingerprint for a packed target");
    }
    fingerprintWords(*fingerprints[index], words);
    for (int word = 0; word < kPatternFingerprintWords; ++word) {
      packed[static_cast<std::size_t>(word) * count + index] = words[word];
    }
  }
  return packed;
}

PatternScreenQuery makePatternScreenQuery(const ExplicitBitVect* fingerprint, int numQueryAtoms) {
  PatternScreenQuery query{};
  query.numAtoms = numQueryAtoms;
  if (fingerprint == nullptr) {
    return query;
  }
  std::uint64_t words[kPatternFingerprintWords];
  fingerprintWords(*fingerprint, words);
  for (int word = 0; word < kPatternFingerprintWords; ++word) {
    if (words[word] != 0) {
      query.words[query.numWords]       = words[word];
      query.wordIndices[query.numWords] = static_cast<std::uint8_t>(word);
      ++query.numWords;
    }
  }
  return query;
}

}  // namespace nvMolKit
