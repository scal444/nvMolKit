// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/ROMol.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nvmolkit/nanobind/array_helpers.h"
#include "src/morgan_fingerprint.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;

struct PythonMoleculeSequence {
  nb::object                       owner;
  std::vector<const RDKit::ROMol*> molecules;
};

PythonMoleculeSequence convertMolecules(const nb::object& molecules) {
  if (!PySequence_Check(molecules.ptr())) {
    PyErr_SetString(PyExc_TypeError, "mols must be a sequence of RDKit molecules");
    throw nb::python_error();
  }
  PyObject* sequence = PySequence_Fast(molecules.ptr(), "mols must be a sequence of RDKit molecules");
  if (sequence == nullptr) {
    throw nb::python_error();
  }

  PythonMoleculeSequence converted{nb::steal<nb::object>(sequence), {}};
  const Py_ssize_t       moleculeCount = PySequence_Fast_GET_SIZE(sequence);
  PyObject* const*       items         = PySequence_Fast_ITEMS(sequence);
  converted.molecules.reserve(static_cast<std::size_t>(moleculeCount));
  for (Py_ssize_t index = 0; index < moleculeCount; ++index) {
    const RDKit::ROMol* molecule = nb::cast<const RDKit::ROMol*>(nb::handle(items[index]));
    if (molecule == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(index));
    }
    converted.molecules.push_back(molecule);
  }
  return converted;
}

template <int NumBits>
std::unique_ptr<nvMolKit::nanobind_bindings::PyArray> makePyArrayFromFlatBitVects(
  nvMolKit::AsyncDeviceVector<nvMolKit::FlatBitVect<NumBits>>& deviceVector) {
  using StorageType             = typename nvMolKit::FlatBitVect<NumBits>::StorageType;
  const std::string dtype       = nvMolKit::nanobind_bindings::getNumpyType<StorageType>();
  const int         rowCount    = deviceVector.size();
  const int         columnCount = NumBits / (8 * sizeof(StorageType));
  return nvMolKit::nanobind_bindings::makePyArray(deviceVector, dtype, nb::make_tuple(rowCount, columnCount));
}

template <int NumBits>
std::unique_ptr<nvMolKit::nanobind_bindings::PyArray> getFingerprintsDevice(
  nvMolKit::MorganFingerprintGenerator&      generator,
  const std::vector<const RDKit::ROMol*>&    molecules,
  cudaStream_t                               stream,
  const nvMolKit::FingerprintComputeOptions& computeOptions) {
  auto fingerprints = [&]() {
    nvMolKit::ScopedNvtxRange range("MorganFPBindingNativeCompute", nvMolKit::NvtxColor::kOrange);
    return generator.GetFingerprintsGpuBuffer<NumBits>(molecules, stream, computeOptions);
  }();

  nvMolKit::ScopedNvtxRange range("MorganFPBindingOutputWrapping", nvMolKit::NvtxColor::kCyan);
  return makePyArrayFromFlatBitVects<NumBits>(fingerprints);
}

std::unique_ptr<nvMolKit::nanobind_bindings::PyArray> getFingerprintsDevicePython(
  nvMolKit::MorganFingerprintGenerator& generator,
  const nb::object&                     molecules,
  int                                   numThreads,
  std::uintptr_t                        streamPointer) {
  auto convertedMolecules = [&]() {
    nvMolKit::ScopedNvtxRange range("MorganFPBindingInputConversion", nvMolKit::NvtxColor::kYellow);
    return convertMolecules(molecules);
  }();

  nvMolKit::FingerprintComputeOptions computeOptions;
  computeOptions.backend       = nvMolKit::FingerprintComputeBackend::GPU;
  computeOptions.numCpuThreads = numThreads;
  auto stream                  = [&]() {
    nvMolKit::ScopedNvtxRange range("MorganFPBindingStreamAcquisition", nvMolKit::NvtxColor::kBlue);
    return nvMolKit::acquireExternalStream(streamPointer);
  }();
  if (!stream) {
    throw std::invalid_argument("Invalid CUDA stream");
  }

  switch (generator.GetOptions().fpSize) {
    case 128:
      return getFingerprintsDevice<128>(generator, convertedMolecules.molecules, *stream, computeOptions);
    case 256:
      return getFingerprintsDevice<256>(generator, convertedMolecules.molecules, *stream, computeOptions);
    case 512:
      return getFingerprintsDevice<512>(generator, convertedMolecules.molecules, *stream, computeOptions);
    case 1024:
      return getFingerprintsDevice<1024>(generator, convertedMolecules.molecules, *stream, computeOptions);
    case 2048:
      return getFingerprintsDevice<2048>(generator, convertedMolecules.molecules, *stream, computeOptions);
    default:
      throw std::invalid_argument("Invalid fpSize: " + std::to_string(generator.GetOptions().fpSize) +
                                  ". Supported values are 128, 256, 512, 1024, 2048");
  }
}

}  // namespace

NB_MODULE(_Fingerprints, module) {
  namespace nb = nanobind;
  using namespace nb::literals;

  nb::module_::import_("rdkit.Chem.rdchem");
  nb::module_::import_("rdkit.DataStructs.cDataStructs");
  nb::module_::import_("nvmolkit._arrayHelpers");

  nb::class_<nvMolKit::MorganFingerprintGenerator>(module, "MorganFingerprintGenerator")
    .def(nb::init<const std::uint32_t, const std::uint32_t>(), "radius"_a, "fpSize"_a)
    .def(
      "GetFingerprint",
      [](nvMolKit::MorganFingerprintGenerator& generator, const RDKit::ROMol& molecule) {
        return generator.GetFingerprint(molecule);
      },
      "mol"_a)
    .def("GetFingerprintsDevice",
         &getFingerprintsDevicePython,
         "mols"_a,
         "num_threads"_a = 0,
         "stream"_a      = static_cast<std::uintptr_t>(0));
}
