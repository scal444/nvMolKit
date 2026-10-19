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
#include <limits>

#include "nvmolkit/array_helpers.h"
#include "src/coulomb_fields.h"
#include "src/utils/device.h"

namespace {

boost::python::object calcCoulombFields(const boost::python::list& molecules,
                                        const std::uintptr_t       pointPointer,
                                        const size_t               pointCount,
                                        const boost::python::list& conformerIds,
                                        const std::string&         chargeProperty,
                                        const bool                 dielectric,
                                        const double               probeCharge,
                                        const bool                 absoluteValue,
                                        const double               alpha,
                                        const double               cutoff,
                                        const double               threshold,
                                        const double               epsilon,
                                        const double               xi,
                                        const std::uintptr_t       streamPointer) {
  auto stream = nvMolKit::acquireExternalStream(streamPointer);
  if (!stream)
    throw std::invalid_argument("Invalid CUDA stream");
  if (pointCount > std::numeric_limits<size_t>::max() / 3 / sizeof(double) || (pointCount != 0 && pointPointer == 0))
    throw std::invalid_argument("Invalid Coulomb query point buffer");
  std::vector<const RDKit::ROMol*> molPointers;
  std::vector<int>                 ids;
  for (int item = 0; item < boost::python::len(molecules); ++item)
    molPointers.push_back(boost::python::extract<const RDKit::ROMol*>(molecules[item]));
  for (int item = 0; item < boost::python::len(conformerIds); ++item)
    ids.push_back(boost::python::extract<int>(conformerIds[item]));
  nvMolKit::CoulombFieldOptions options;
  options.dielectric    = dielectric;
  options.absoluteValue = absoluteValue;
  options.probeCharge   = probeCharge;
  options.alpha         = alpha;
  options.cutoff        = cutoff;
  options.threshold     = threshold;
  options.epsilon       = epsilon;
  options.xi            = xi;
  auto output           = nvMolKit::coulombFieldsMol(
    molPointers,
    cuda::std::span<const double>(reinterpret_cast<const double*>(pointPointer), 3 * pointCount),
    ids,
    chargeProperty,
    options,
    *stream);
  auto* array     = nvMolKit::makePyArray(output, boost::python::make_tuple(molPointers.size(), pointCount));
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

}  // namespace

BOOST_PYTHON_MODULE(_interactionFields) {
  boost::python::def("CalcCoulombFields", &calcCoulombFields);
}
