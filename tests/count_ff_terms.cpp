// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dist_geom.h"
#include "dist_geom_flattened_builder.h"
#include "embedder_utils.h"
#include "test_utils.h"

constexpr int maxAtoms = 256;

std::vector<std::unique_ptr<RDKit::RWMol>> readMolecules(const std::string& filePath, unsigned int count) {
  std::vector<std::unique_ptr<RDKit::ROMol>> tempMols;

  std::string extension = std::filesystem::path(filePath).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  if (extension == ".sdf") {
    getMols(filePath, tempMols, count);
  } else if (extension == ".smi" || extension == ".smiles" || extension == ".cxsmiles") {
    std::ifstream file(filePath);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open SMILES file: " + filePath);
    }

    std::string                                line;
    std::vector<std::unique_ptr<RDKit::ROMol>> allMols;

    while (std::getline(file, line) && allMols.size() < count) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::string smiles = line.substr(0, line.find_first_of(" \t"));

      try {
        auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
        if (mol && mol->getNumAtoms() <= maxAtoms) {
          allMols.push_back(std::move(mol));
        } else if (mol) {
          std::cerr << "Warning: Molecule with SMILES " << smiles << " has more than " << maxAtoms
                    << " atoms and will be skipped." << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to parse SMILES: " << smiles << " - " << e.what() << std::endl;
      }
    }

    if (allMols.empty()) {
      throw std::runtime_error("No valid molecules found in SMILES file: " + filePath);
    }

    tempMols = std::move(allMols);
  } else {
    throw std::runtime_error(
      "Unsupported file format. Only .sdf, .smi, .smiles, and .cxsmiles files are supported. Got " + extension);
  }

  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  for (auto& tempMol : tempMols) {
    std::unique_ptr<RDKit::ROMol> mol2(RDKit::MolOps::addHs(*tempMol));
    mols.push_back(std::make_unique<RDKit::RWMol>(*mol2));
    mols.back()->clearConformers();
    RDKit::MolOps::sanitizeMol(*mols.back());
  }
  return mols;
}

void printUsage(const char* programName) {
  std::cout << "Usage: " << programName << " <file_path> <N>\n";
  std::cout << "  file_path: Path to input file (.sdf, .smi, .smiles, or .cxsmiles)\n";
  std::cout << "  N: Number of molecules to process\n\n";
  std::cout << "Example: " << programName << " molecules.smi 10\n";
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 1;
  }

  std::string filePath = argv[1];
  unsigned int numMols;
  
  try {
    numMols = std::stoul(argv[2]);
    if (numMols == 0) {
      std::cerr << "Error: N must be positive\n";
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: Invalid number of molecules: " << argv[2] << "\n";
    return 1;
  }

  if (!std::filesystem::exists(filePath)) {
    std::cerr << "Error: File does not exist: " << filePath << std::endl;
    return 1;
  }

  std::cout << "Reading molecules from: " << filePath << "\n";
  auto mols = readMolecules(filePath, numMols);
  std::cout << "Processing " << mols.size() << " molecule(s)...\n\n";

  // Accumulate statistics
  size_t totalAtoms = 0;
  size_t totalDgDistTerms = 0;
  size_t totalDgChiralTerms = 0;
  size_t totalDgFourthTerms = 0;
  size_t totalEtkdgExpTorsionTerms = 0;
  size_t totalEtkdgImproperTerms = 0;
  size_t totalEtkdgDist12Terms = 0;
  size_t totalEtkdgDist13Terms = 0;
  size_t totalEtkdgAngle13Terms = 0;
  size_t totalEtkdgLongRangeTerms = 0;

  for (size_t molIdx = 0; molIdx < mols.size(); molIdx++) {
    const auto& mol = mols[molIdx];
    const int numAtoms = mol->getNumAtoms();
    totalAtoms += numAtoms;

    // Set up embedder arguments
    auto params = DGeomHelpers::ETKDGv3;
    params.useRandomCoords = true;
    
    nvMolKit::detail::EmbedArgs eargs;
    std::vector<std::unique_ptr<RDGeom::Point>> positions;
    std::unique_ptr<ForceFields::ForceField> field;
    
    nvMolKit::DGeomHelpers::setupRDKitFFWithPos(mol.get(), params, field, eargs, positions);

    // DistGeom (4D) Force Field Terms
    auto dgParams = nvMolKit::DistGeom::constructForceFieldContribs(
      eargs.dim,
      *eargs.mmat,
      eargs.chiralCenters
    );

    totalDgDistTerms += dgParams.distTerms.idx1.size();
    totalDgChiralTerms += dgParams.chiralTerms.idx1.size();
    totalDgFourthTerms += dgParams.fourthTerms.idx.size();

    // ETKDG (3D) Force Field Terms
    std::vector<double> positions3D(numAtoms * 3, 0.0);
    for (int i = 0; i < numAtoms; i++) {
      for (int d = 0; d < 3; d++) {
        positions3D[i * 3 + d] = eargs.posVec[i * eargs.dim + d];
      }
    }

    auto etkdgParams = nvMolKit::DistGeom::construct3DForceFieldContribs(
      *eargs.mmat,
      eargs.etkdgDetails,
      positions3D,
      3,
      params.useBasicKnowledge
    );

    totalEtkdgExpTorsionTerms += etkdgParams.experimentalTorsionTerms.idx1.size();
    totalEtkdgImproperTerms += etkdgParams.improperTorsionTerms.idx1.size();
    totalEtkdgDist12Terms += etkdgParams.dist12Terms.idx1.size();
    totalEtkdgDist13Terms += etkdgParams.dist13Terms.idx1.size();
    totalEtkdgAngle13Terms += etkdgParams.angle13Terms.idx1.size();
    totalEtkdgLongRangeTerms += etkdgParams.longRangeDistTerms.idx1.size();
  }

  const double numMolsDouble = static_cast<double>(mols.size());

  // Print averages
  std::cout << "=== Average Terms per Molecule ===\n";
  std::cout << "Average atoms per molecule: " << totalAtoms / numMolsDouble << "\n\n";

  std::cout << "DistGeom (4D) Average Terms per Molecule:\n";
  std::cout << "  Distance violation terms:   " << totalDgDistTerms / numMolsDouble << "\n";
  std::cout << "  Chiral violation terms:     " << totalDgChiralTerms / numMolsDouble << "\n";
  std::cout << "  Fourth dimension terms:     " << totalDgFourthTerms / numMolsDouble << "\n";
  std::cout << "  Total DistGeom terms:       " 
            << (totalDgDistTerms + totalDgChiralTerms + totalDgFourthTerms) / numMolsDouble << "\n\n";

  std::cout << "ETKDG (3D) Average Terms per Molecule:\n";
  std::cout << "  Experimental torsion terms: " << totalEtkdgExpTorsionTerms / numMolsDouble << "\n";
  std::cout << "  Improper torsion terms:     " << totalEtkdgImproperTerms / numMolsDouble << "\n";
  std::cout << "  1-2 distance terms:         " << totalEtkdgDist12Terms / numMolsDouble << "\n";
  std::cout << "  1-3 distance terms:         " << totalEtkdgDist13Terms / numMolsDouble << "\n";
  std::cout << "  1-3 angle terms:            " << totalEtkdgAngle13Terms / numMolsDouble << "\n";
  std::cout << "  Long range distance terms:  " << totalEtkdgLongRangeTerms / numMolsDouble << "\n";
  std::cout << "  Total ETKDG terms:          " 
            << (totalEtkdgExpTorsionTerms + totalEtkdgImproperTerms + totalEtkdgDist12Terms + 
                totalEtkdgDist13Terms + totalEtkdgAngle13Terms + totalEtkdgLongRangeTerms) / numMolsDouble << "\n";

  return 0;
}

