#include "LinSysSolver.hpp"


#ifdef PARTH_WITH_ACCELERATE

#include "AccelerateSolver.hpp"

#endif

#ifdef PARTH_WITH_CHOLMOD

#include "CHOLMODSolver.hpp"

#endif

#ifdef PARTH_WITH_SYMPILER

#include "SympilerSolver.hpp"

#endif

#ifdef PARTH_WITH_MKL
#include "MKLSolver.hpp"
#endif

namespace PARTH_SOLVER {

    LinSysSolver *LinSysSolver::create(const LinSysSolverType type) {
        switch (type) {

#ifdef PARTH_WITH_ACCELERATE
            case LinSysSolverType::ACCELERATE:
                return new AccelerateSolver();
#endif

#ifdef PARTH_WITH_CHOLMOD
            case LinSysSolverType::CHOLMOD:
                return new CHOLMODSolver();
#endif

#ifdef PARTH_WITH_MKL
                case LinSysSolverType::MKL_LIB:
                  return new MKLSolver();
#endif
            default:
                std::cerr << "Uknown linear system solver type" << std::endl;
                return nullptr;
        }
    }


} // namespace IPC
