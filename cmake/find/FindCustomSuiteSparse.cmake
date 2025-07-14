# Try to find suitesparse headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(CustomSuiteSparse)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#
# Variables defined by this module:
#
#  SUITESPARSE_FOUND              System has suitesparse libraries and headers
#  SUITESPARSE_LIBRARIES          The suitesparse library
#  SUITESPARSE_INCLUDE_DIRS       The location of suitesparse headers

find_path(SUITESPARSE_PERFIX
        NAMES include/cholmod.h
        PATHS $ENV{SUITESPARSEROOT}
        )
message(STATUS "************* ${SUITESPARSE_PERFIX}")
find_library(AMD NAMES amd HINTS ${SUITESPARSE_PERFIX}/lib)
if (AMD)
    message(STATUS "Found amd: ${AMD}")
else ()
    message(FATAL_ERROR "amd library not found")
endif ()

find_library(CAMD NAMES camd HINTS ${SUITESPARSE_PERFIX}/lib)
if (CAMD)
    message(STATUS "Found camd: ${CAMD}")
else ()
    message(FATAL_ERROR "camd library not found")
endif ()

find_library(COLAMD NAMES colamd HINTS ${SUITESPARSE_PERFIX}/lib)
if (COLAMD)
    message(STATUS "Found colamd: ${COLAMD}")
else ()
    message(FATAL_ERROR "colamd library not found")
endif ()

find_library(CCOLAMD NAMES ccolamd HINTS ${SUITESPARSE_PERFIX}/lib)
if (CCOLAMD)
    message(STATUS "Found ccolamd: ${CCOLAMD}")
else ()
    message(FATAL_ERROR "ccolamd library not found")
endif ()

find_library(CHOLMOD NAMES cholmod HINTS ${SUITESPARSE_PERFIX}/lib)
if (CHOLMOD)
    message(STATUS "Found cholmod: ${CHOLMOD}")
else ()
    message(FATAL_ERROR "cholmod library not found")
endif ()

find_library(SUITESPARSEQR NAMES spqr HINTS ${SUITESPARSE_PERFIX}/lib)
if (SUITESPARSEQR)
    message(STATUS "Found spqr: ${SUITESPARSEQR}")
else ()
    message(FATAL_ERROR "spqr library not found")
endif ()

find_library(SUITESPARSE_CONFIG NAMES suitesparseconfig HINTS ${SUITESPARSE_PERFIX}/lib)
if (SUITESPARSE_CONFIG)
    message(STATUS "Found suitesparseconfig: ${SUITESPARSE_CONFIG}")
else ()
    message(FATAL_ERROR "suitesparseconfig library not found")
endif ()

find_path(SUITESPARSE_INCLUDE_DIRS
        NAMES cholmod.h
        HINTS ${SUITESPARSE_PERFIX}/include
        )

if (SUITESPARSE_INCLUDE_DIRS)
    message(STATUS "Found suitesparse headers at: ${SUITESPARSE_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "suitesparse headers are not found")
endif ()

# BLAS.
if (IPC_WITH_MKL)
    set(BLA_VENDOR Intel10_64lp)
endif ()

find_library(RT NAMES rt HINTS ${SUITESPARSE_PERFIX}/lib ${HILTIDEPS}/lib)
if (RT)
    message(STATUS "Found rt: ${RT}")
else ()
    message(FATAL_ERROR "rt library not found")
endif ()

set(SUITESPARSE_LIBRARIES ${AMD};
        ${CAMD};
        ${COLAMD};
        ${CHOLMOD};
        ${SUITESPARSEQR};
        ${SUITESPARSE_CONFIG};
        ${RT};
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CustomSuiteSparse DEFAULT_MSG
        AMD
        CAMD
        COLAMD
        CHOLMOD
        SUITESPARSEQR
        SUITESPARSE_CONFIG
        RT
        )

mark_as_advanced(
        AMD
        CAMD
        COLAMD
        SUITESPARSEQR
        SUITESPARSE_CONFIG
        RT
)
