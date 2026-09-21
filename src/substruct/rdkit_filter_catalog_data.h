// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2015, Novartis Institutes for BioMedical Research Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Novartis Institutes for BioMedical Research Inc. nor
//   the names of its contributors may be used to endorse or promote products
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.

#ifndef NVMOLKIT_RDKIT_FILTER_CATALOG_DATA_H
#define NVMOLKIT_RDKIT_FILTER_CATALOG_DATA_H

// RDKit installs FilterCatalog.h and exports these data accessors from
// libRDKitFilterCatalog, but it does not install the Filters.h header declaring
// them. Keep this ABI-compatible declaration narrow and tied to the public
// FilterCatalogParams enum.
#include <GraphMol/FilterCatalog/FilterCatalog.h>

namespace RDKit {

struct FilterData_t {
  const char*  name;
  const char*  smarts;
  unsigned int max;
  const char*  comment;
};

struct FilterProperty_t {
  const char* key;
  const char* value;
};

unsigned int            GetNumEntries(FilterCatalogParams::FilterCatalogs catalog);
const FilterData_t*     GetFilterData(FilterCatalogParams::FilterCatalogs catalog);
unsigned int            GetNumPropertyEntries(FilterCatalogParams::FilterCatalogs catalog);
const FilterProperty_t* GetFilterProperties(FilterCatalogParams::FilterCatalogs catalog);

}  // namespace RDKit

#endif  // NVMOLKIT_RDKIT_FILTER_CATALOG_DATA_H
