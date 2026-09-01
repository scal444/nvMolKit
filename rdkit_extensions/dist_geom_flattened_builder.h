#ifndef NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H
#define NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>

#include <map>

#include "src/forcefields/dist_geom.h"
namespace nvMolKit {
namespace DistGeom {

//! Force constants from RDKit's ETKDGForceConsts::AIO::Cosine bundle.
struct AllInOneForceConstants {
  double distance      = 2.15;
  double fourthDim     = 2.15;
  double chiral        = 1.0;
  double kTermAngle    = 0.1;
  double kTermImproper = 0.001;
  double kTermTorsion  = 2.15;
  double etTermScaling = 0.1;
};

inline constexpr AllInOneForceConstants kAllInOneForceConstants{};

nvMolKit::DistGeom::EnergyForceContribsHost constructForceFieldContribs(
  const int                              dim,
  const ::DistGeom::BoundsMatrix&        mmat,
  const ::DistGeom::VECT_CHIRALSET&      csets,
  double                                 weightChiral    = 1.0,
  double                                 weightFourthDim = 0.1,
  std::map<std::pair<int, int>, double>* extraWeights    = nullptr,
  double                                 basinSizeTol    = 5.0,
  double                                 distanceWeight  = 1.0);

nvMolKit::DistGeom::Energy3DForceContribsHost construct3DForceFieldContribs(
  const ::DistGeom::BoundsMatrix&                   mmat,
  const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
  const std::vector<double>&                        positions,
  int                                               dim,
  bool                                              useBasicKnowledge = true);

//! Builds the host contribution bundle for RDKit's all-in-one ETKDG force field.
//!
//! `etkdgDetails` must already contain the AIO-scaled experimental and K-ring
//! torsion constants. RDKit 2026.03.4+ does this while initializing embed args
//! for `useLegacyImplementation=false`.
nvMolKit::DistGeom::AllInOneForceContribsHost constructAllInOneForceFieldContribs(
  int                                               dim,
  const ::DistGeom::BoundsMatrix&                   mmat,
  const ::DistGeom::VECT_CHIRALSET&                 csets,
  const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
  const double*                                     topologicalDistanceMatrix,
  bool                                              useExperimentalTorsions,
  bool                                              useBasicKnowledge);

}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_FLATTENED_BUILDER_H
