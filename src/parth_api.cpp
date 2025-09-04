//
// Parth API implementation - bridges new public API to existing implementation
//

#include "parth/parth.h"
#include "Parth.h"  // Existing implementation
#include "ParthTypes.h"

namespace PARTH {

// Convert between enum types
::PARTH::ReorderingType convertReorderingType(ReorderingType type) {
    switch (type) {
        case ReorderingType::METIS: return ::PARTH::ReorderingType::METIS;
        case ReorderingType::AMD: return ::PARTH::ReorderingType::AMD;
        case ReorderingType::AUTO: return ::PARTH::ReorderingType::METIS; // Default to METIS for AUTO
        default: return ::PARTH::ReorderingType::METIS;
    }
}

ReorderingType convertReorderingType(::PARTH::ReorderingType type) {
    switch (type) {
        case ::PARTH::ReorderingType::METIS: return ReorderingType::METIS;
        case ::PARTH::ReorderingType::AMD: return ReorderingType::AMD;
        case ::PARTH::ReorderingType::MORTON_CODE: return ReorderingType::AUTO;
        default: return ReorderingType::METIS;
    }
}

// Private implementation class
class Parth::Impl {
public:
    ::PARTH::Parth parth_impl;
    
    Impl() {
        // Set reasonable defaults
        parth_impl.setReorderingType(::PARTH::ReorderingType::METIS);
        parth_impl.setVerbose(false);
        parth_impl.setNDLevels(6);
        parth_impl.setNumberOfCores(1);
    }
};

Parth::Parth() : pImpl(new Impl()) {}

Parth::~Parth() {
    delete pImpl;
}

void Parth::setReorderingType(ReorderingType type) {
    pImpl->parth_impl.setReorderingType(convertReorderingType(type));
}

ReorderingType Parth::getReorderingType() const {
    return convertReorderingType(pImpl->parth_impl.getReorderingType());
}

void Parth::setVerbose(bool verbose) {
    pImpl->parth_impl.setVerbose(verbose);
}

bool Parth::getVerbose() const {
    return pImpl->parth_impl.getVerbose();
}

void Parth::setNDLevels(int levels) {
    pImpl->parth_impl.setNDLevels(levels);
}

int Parth::getNDLevels() const {
    return pImpl->parth_impl.getNDLevels();
}

void Parth::setNumberOfCores(int num_cores) {
    pImpl->parth_impl.setNumberOfCores(num_cores);
}

int Parth::getNumberOfCores() const {
    return pImpl->parth_impl.getNumberOfCores();
}

void Parth::setMeshPointers(int n, int* Mp, int* Mi) {
    pImpl->parth_impl.setMeshPointers(n, Mp, Mi);
}

void Parth::setMeshPointers(int n, int* Mp, int* Mi, const std::vector<int>& new_to_old_map) {
    std::vector<int> map_copy = new_to_old_map; // Make non-const copy
    pImpl->parth_impl.setMeshPointers(n, Mp, Mi, map_copy);
}

void Parth::setNewToOldDOFMap(const std::vector<int>& map) {
    std::vector<int> map_copy = map; // Make non-const copy
    pImpl->parth_impl.setNewToOldDOFMap(map_copy);
}

void Parth::computePermutation(std::vector<int>& perm, int dim) {
    pImpl->parth_impl.computePermutation(perm, dim);
}

void Parth::mapMeshPermToMatrixPerm(const std::vector<int>& mesh_perm, 
                                    std::vector<int>& matrix_perm, int dim) {
    std::vector<int> mesh_perm_copy = mesh_perm; // Make non-const copy
    pImpl->parth_impl.mapMeshPermToMatrixPerm(mesh_perm_copy, matrix_perm, dim);
}

double Parth::getReuse() const {
    return pImpl->parth_impl.getReuse();
}

int Parth::getNumChanges() const {
    return pImpl->parth_impl.getNumChanges();
}

void Parth::printTiming() const {
    pImpl->parth_impl.printTiming();
}

void Parth::resetTimers() {
    pImpl->parth_impl.resetTimers();
}

void Parth::clearParth() {
    pImpl->parth_impl.clearParth();
}

} // namespace PARTH
