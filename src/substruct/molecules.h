#include <cstdint>
#include <limits>
#include <vector>
namespace RDKit {
class ROMol;
} // namespace RDKit

namespace nvMolKit {
struct AtomData {
  static constexpr uint8_t unsetValenceVal =
      std::numeric_limits<uint8_t>::max();

  uint8_t atomicNum = 0;
  uint8_t numExplicitHs = 0;
  uint8_t explicitValence = unsetValenceVal;
  uint8_t implicitValence = unsetValenceVal;

  int8_t formalCharge = 0;
  uint8_t chiralTag = 0;
  uint8_t numRadicalElectrons = 0;
  uint8_t hybridization = 0;
  uint8_t minRingSize = 0;
  uint8_t numRings = 0;
  bool isAromatic = false;
};


struct BondData {
  std::uint8_t bondType = 0;
};

struct MoleculesHost {
  // batch-level variables, global indexed
  std::vector<int> batchAtomStarts;
  std::vector<int> batchBondStarts;
  std::vector<int> batchAtomBondStarts;
  std::vector<int> batchOtherAtomIndicesStarts;
  std::vector<int> batchBondIndicesStarts;

  // molecule level variables. These will be internal-indexed
  std::vector<AtomData> atomData;
  std::vector<BondData> bondData;
  std::vector<int16_t> atomBondStarts = {0u};
  std::vector<int16_t> otherAtomIndices;
  std::vector<int16_t> bondDataIndices;
};


void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch);
} // namespace nvMolKit