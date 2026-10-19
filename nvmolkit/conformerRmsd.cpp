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

#include <GraphMol/GraphMol.h>

#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nvmolkit/array_helpers.h"
#include "src/conformer_rmsd_mol.h"
#include "src/utils/device.h"

namespace {

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

}  // namespace

BOOST_PYTHON_MODULE(_conformerRmsd) {
  boost::python::def(
    "GetAlignmentTransformBatch",
    +[](const boost::python::list& probes,
        const RDKit::ROMol&        reference,
        const boost::python::list& atomMaps,
        const boost::python::list& conformerIds,
        const int                  referenceConformerId,
        const std::uintptr_t       streamPtr) {
      auto stream = nvMolKit::acquireExternalStream(streamPtr);
      if (!stream)
        throw std::invalid_argument("Invalid CUDA stream");
      const int count = boost::python::len(probes);
      if (boost::python::len(atomMaps) != count || boost::python::len(conformerIds) != count)
        throw std::invalid_argument("Expected one atom map and conformer ID per probe");
      std::vector<const RDKit::ROMol*>              molecules;
      std::vector<std::vector<std::pair<int, int>>> maps(count);
      std::vector<int>                              ids;
      for (int item = 0; item < count; ++item) {
        molecules.push_back(boost::python::extract<const RDKit::ROMol*>(probes[item]));
        ids.push_back(boost::python::extract<int>(conformerIds[item]));
        const boost::python::object atomMap = atomMaps[item];
        for (int atom = 0; atom < boost::python::len(atomMap); ++atom) {
          const boost::python::object pair = atomMap[atom];
          if (boost::python::len(pair) != 2)
            throw std::invalid_argument("Atom map entries must be pairs");
          maps[item].emplace_back(boost::python::extract<int>(pair[0]), boost::python::extract<int>(pair[1]));
        }
      }
      auto result =
        nvMolKit::alignmentTransformBatchMol(molecules, reference, maps, ids, referenceConformerId, *stream);
      return boost::python::make_tuple(
        toOwnedPyArray(nvMolKit::makePyArray(result.rmsds, boost::python::make_tuple(count))),
        toOwnedPyArray(nvMolKit::makePyArray(result.transforms, boost::python::make_tuple(count, 4, 4))));
    });

  boost::python::def(
    "GetConformerRMSMatrixBatch",
    +[](boost::python::list& mols, const bool prealigned, const bool alignToFirstConformer, std::uintptr_t streamPtr)
      -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }

      const int numMols = boost::python::len(mols);
      if (numMols == 0) {
        return boost::python::list();
      }

      std::vector<const RDKit::ROMol*> molsVec(numMols);
      for (int i = 0; i < numMols; ++i) {
        molsVec[i] = boost::python::extract<const RDKit::ROMol*>(boost::python::object(mols[i]));
        if (!molsVec[i]) {
          throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
        }
      }

      auto buffers = nvMolKit::conformerRmsdBatchMatrixMol(molsVec, prealigned, *streamOpt, alignToFirstConformer);

      boost::python::list results;
      for (int m = 0; m < numMols; ++m) {
        const int nc       = molsVec[m]->getNumConformers();
        const int numPairs = nc >= 2 ? nc * (nc - 1) / 2 : 0;
        results.append(toOwnedPyArray(nvMolKit::makePyArray(buffers[m], boost::python::make_tuple(numPairs))));
      }
      return results;
    },
    (boost::python::arg("mols"),
     boost::python::arg("prealigned")            = false,
     boost::python::arg("alignToFirstConformer") = false,
     boost::python::arg("stream")                = 0));

  boost::python::def(
    "GetConformerRMSMatrix",
    +[](RDKit::ROMol& mol, const bool prealigned, const bool alignToFirstConformer, std::uintptr_t streamPtr)
      -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }

      const int     numConfs = mol.getNumConformers();
      const int64_t numPairs = numConfs >= 2 ? static_cast<int64_t>(numConfs) * (numConfs - 1) / 2 : 0;
      auto          buffer   = nvMolKit::conformerRmsdMatrixMol(mol, prealigned, *streamOpt, alignToFirstConformer);
      return toOwnedPyArray(nvMolKit::makePyArray(buffer, boost::python::make_tuple(numPairs)));
    },
    (boost::python::arg("mol"),
     boost::python::arg("prealigned")            = false,
     boost::python::arg("alignToFirstConformer") = false,
     boost::python::arg("stream")                = 0));
}
