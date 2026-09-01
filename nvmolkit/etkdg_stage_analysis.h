// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_ETKDG_STAGE_ANALYSIS_H
#define NVMOLKIT_ETKDG_STAGE_ANALYSIS_H

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <string>

#include "src/precision_options.h"

namespace nvMolKit {

//! Run one production ETKDG minimization stage from caller-supplied coordinates.
boost::python::object analyzeETKDGStage(const boost::python::list&                  molecules,
                                        const boost::python::list&                  coordinates,
                                        const RDKit::DGeomHelpers::EmbedParameters& params,
                                        const std::string&                          stage,
                                        const std::string&                          backend,
                                        const PrecisionOptions&                     precision,
                                        bool                                        includeCpuReference);

}  // namespace nvMolKit

#endif  // NVMOLKIT_ETKDG_STAGE_ANALYSIS_H
