// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_FILTER_CATALOG_H
#define NVMOLKIT_FILTER_CATALOG_H

#include <cuda_runtime.h>
#include <GraphMol/FilterCatalog/FilterCatalog.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/substruct/substruct_types.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

/** The official RDKit bitmask enum, including PAINS, CHEMBL, and ALL combinations. */
using FilterCatalogPreset = RDKit::FilterCatalogParams::FilterCatalogs;

using FilterCatalogProperties = std::map<std::string, std::string>;

/** Host-side information retained for a catalog entry. */
struct FilterCatalogEntry {
  unsigned int            id = 0;
  std::string             description;
  std::string             smarts;
  unsigned int            triggerCount = 1;
  FilterCatalogProperties properties;
};

/**
 * A persistent collection of substructure filters for batched molecule screening.
 *
 * Entries are copied into the catalog and receive stable, insertion-ordered IDs.
 * finalize() publishes pending entries and prepares supported queries for the
 * configured GPU. Entries which cannot use GPU execution remain correct through
 * RDKit fallback. Query methods return one result per target in input order.
 */
class FilterCatalog {
 public:
  explicit FilterCatalog(SubstructSearchConfig config = SubstructSearchConfig{});
  explicit FilterCatalog(FilterCatalogPreset preset, SubstructSearchConfig config = SubstructSearchConfig{});
  ~FilterCatalog();

  FilterCatalog(const FilterCatalog&)            = delete;
  FilterCatalog& operator=(const FilterCatalog&) = delete;
  FilterCatalog(FilterCatalog&&)                 = delete;
  FilterCatalog& operator=(FilterCatalog&&)      = delete;

  /** Add all atomic catalogs selected by preset and return the number of entries added. */
  std::size_t addPreset(FilterCatalogPreset preset);

  /** Copy a positive query into the pending generation and return its stable ID. */
  unsigned int addEntry(const RDKit::ROMol&     query,
                        std::string             description,
                        unsigned int            triggerCount = 1,
                        FilterCatalogProperties properties   = {});

  /** Parse and add a SMARTS query. */
  unsigned int addSmarts(std::string             smarts,
                         std::string             description,
                         unsigned int            triggerCount = 1,
                         FilterCatalogProperties properties   = {});

  /** Prepare every pending entry and atomically publish it to subsequent queries. */
  void finalize(cudaStream_t stream = nullptr);

  /** Number of entries visible to queries. */
  [[nodiscard]] std::size_t size() const;

  /** Number of staged entries not yet visible to queries. */
  [[nodiscard]] std::size_t pendingSize() const;

  /** Return a copy of an entry's metadata. */
  [[nodiscard]] FilterCatalogEntry getEntry(unsigned int id) const;

  /** Return whether each target triggers at least one entry. */
  [[nodiscard]] std::vector<std::uint8_t> hasMatch(const std::vector<const RDKit::ROMol*>& targets,
                                                   cudaStream_t                            stream = nullptr) const;

  /** Return the first matching entry ID for each target, or -1 when none matches. */
  [[nodiscard]] std::vector<int> getFirstMatch(const std::vector<const RDKit::ROMol*>& targets,
                                               cudaStream_t                            stream = nullptr) const;

  /** Return every matching entry ID for each target in catalog order. */
  [[nodiscard]] std::vector<std::vector<unsigned int>> getMatches(const std::vector<const RDKit::ROMol*>& targets,
                                                                  cudaStream_t stream = nullptr) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FILTER_CATALOG_H
