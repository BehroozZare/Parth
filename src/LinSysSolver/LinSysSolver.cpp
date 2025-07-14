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


#include "PureMETIS.hpp"
#include <EigenSolver.hpp>

#include "CGSolver.hpp"

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

#ifdef PARTH_WITH_SYMPILER
            case LinSysSolverType::SYMPILER:
                return new SympilerSolver();
#endif
                case LinSysSolverType::EIGEN:
                        return new EigenSolver();
#ifdef PARTH_WITH_MKL
                case LinSysSolverType::MKL_LIB:
                  return new MKLSolver();
#endif

#ifdef PARTH_WITH_BARB
            case LinSysSolverType::BARB:
                return new BarbSolver();
            case LinSysSolverType::LAZY_BARB:
                return new LazyBarbSolver();
            case LinSysSolverType::JACOBI_BARB:
                return new JacobiBarbSolver();
            case LinSysSolverType::PARALLEL_CHOLMOD:
                return new ParallelCholmodSolver();
            case LinSysSolverType::PARALLEL_LAZY_CHOLMOD:
              return new ParallelCholmodLazySolver();
            case LinSysSolverType::PURE_METIS:
                return new PureMETIS();
#endif
            case LinSysSolverType::CG:
                return new CGSolver();
            default:
                std::cerr << "Uknown linear system solver type" << std::endl;
                return nullptr;
        }
    }


} // namespace IPC
