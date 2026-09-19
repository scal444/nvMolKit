# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

nanobind_add_module(_DataStructs DataStructs.cpp)
target_link_libraries(_DataStructs PUBLIC ${RDKit_LIBS} Similarity device)
installpythontarget(_DataStructs ./)

nanobind_add_module(_Fingerprints fingerprints.cpp)
target_link_libraries(_Fingerprints PUBLIC ${RDKit_LIBS} morganFingerprint device nvtx CUDA::cudart)
installpythontarget(_Fingerprints ./)

nanobind_add_module(_arrayHelpers array_helpers.cpp)
target_link_libraries(_arrayHelpers PUBLIC device_vector)
installpythontarget(_arrayHelpers ./)

nanobind_add_module(_types types.cpp)
target_link_libraries(_types PUBLIC ${RDKit_LIBS})
installpythontarget(_types ./)

nanobind_add_module(_mmffOptimization mmffOptimization.cpp)
target_link_libraries(_mmffOptimization PRIVATE ${RDKit_LIBS} mmff_minimize ff_utils)
installpythontarget(_mmffOptimization ./)

nanobind_add_module(_uffOptimization uffOptimization.cpp)
target_link_libraries(_uffOptimization PRIVATE ${RDKit_LIBS} uff_minimize ff_utils)
installpythontarget(_uffOptimization ./)

nanobind_add_module(_batchedForcefield batchedForcefield.cpp)
target_link_libraries(
  _batchedForcefield
  PRIVATE ${RDKit_LIBS}
          mmff_minimize
          uff_minimize
          mmff_batched_forcefield
          uff_batched_forcefield
          batched_forcefield
          rdkit_mmff_flattened
          rdkit_uff_flattened
          device
          device_vector
          ff_utils)
installpythontarget(_batchedForcefield ./)

nanobind_add_module(_embedMolecules embedMolecules.cpp)
target_link_libraries(_embedMolecules PUBLIC device_vector)
target_link_libraries(_embedMolecules PRIVATE ${RDKit_LIBS} dist_geom etkdg)
installpythontarget(_embedMolecules ./)

nanobind_add_module(_clustering clustering.cpp)
target_link_libraries(_clustering PUBLIC device_vector butina device)
installpythontarget(_clustering ./)

nanobind_add_module(_conformerRmsd conformerRmsd.cpp)
target_link_libraries(_conformerRmsd PUBLIC device_vector conformer_rmsd_mol device)
target_link_libraries(_conformerRmsd PRIVATE ${RDKit_LIBS})
installpythontarget(_conformerRmsd ./)

nanobind_add_module(_substructure substructure.cpp)
target_link_libraries(_substructure PRIVATE ${RDKit_LIBS} substructure_search pinned_host_allocator)
installpythontarget(_substructure ./)

nanobind_add_module(_substructLibrary substructLibrary.cpp)
target_link_libraries(_substructLibrary PRIVATE ${RDKit_LIBS} substruct_library)
installpythontarget(_substructLibrary ./)

nanobind_add_module(_mcs mcs.cpp)
target_link_libraries(_mcs PRIVATE ${RDKit_LIBS} mcs_search CUDA::cudart)
installpythontarget(_mcs ./)

nanobind_add_module(_TFD tfd.cpp)
target_link_libraries(_TFD PUBLIC device_vector)
target_link_libraries(_TFD PRIVATE ${RDKit_LIBS} tfd_gpu CUDA::cudart)
installpythontarget(_TFD ./)
