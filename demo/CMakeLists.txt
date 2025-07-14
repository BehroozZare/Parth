set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# CLI11
include(cli11)

## Testing SpTRSV kernel
#add_executable(${PROJECT_NAME}_SpTRSV SpTRSV_runtime.cpp)
#target_link_libraries(${PROJECT_NAME}_SpTRSV PUBLIC ${PROJECT_NAME}_lib)
#
#
## Testing Solver kernel
#add_executable(${PROJECT_NAME}_SolverTester solver_tester.cpp)
#target_link_libraries(${PROJECT_NAME}_SolverTester PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_SolverTester PUBLIC CLI11::CLI11)
#
## Testing Solver kernel
#add_executable(${PROJECT_NAME}_FrameProfiling frame_profiling.cpp)
#target_link_libraries(${PROJECT_NAME}_FrameProfiling PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_FrameProfiling PUBLIC CLI11::CLI11)
#
#
## Testing Solver kernel
#add_executable(${PROJECT_NAME}_ReorderingAnalysis ReorderingAnalysis.cpp)
#target_link_libraries(${PROJECT_NAME}_ReorderingAnalysis PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ReorderingAnalysis PUBLIC CLI11::CLI11)
#
## Testing Solver kernel
#add_executable(${PROJECT_NAME}_ParthReorderingAnalysis ParthReorderingAnalysis.cpp)
#target_link_libraries(${PROJECT_NAME}_ParthReorderingAnalysis PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ParthReorderingAnalysis PUBLIC CLI11::CLI11)
#
#
## Testing Solver kernel
#add_executable(${PROJECT_NAME}_ParthRegions parth_regions_dev.cpp)
#target_link_libraries(${PROJECT_NAME}_ParthRegions PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ParthRegions PUBLIC CLI11::CLI11)
#
## Testing Solver kernel
#add_executable(${PROJECT_NAME}_Parth_NDRegions parth_ND_region_dev.cpp)
#target_link_libraries(${PROJECT_NAME}_Parth_NDRegions PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_Parth_NDRegions PUBLIC CLI11::CLI11)
#
## Testing Solver kernel
#add_executable(${PROJECT_NAME}_FactorReuse factor_reuse_analysis.cpp)
#target_link_libraries(${PROJECT_NAME}_FactorReuse PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_FactorReuse PUBLIC CLI11::CLI11)

# Testing Solver kernel
#add_executable(${PROJECT_NAME}_MKL_Test MKL_test.cpp)
#target_link_libraries(${PROJECT_NAME}_MKL_Test PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_MKL_Test PUBLIC CLI11::CLI11)

# Testing Solver kernel
#add_executable(${PROJECT_NAME}_CHOLMOD_Test CHOLMOD_test.cpp FullDecimateAndSubdivision.cpp MeshResolutionAdjustment.cpp)
#target_link_libraries(${PROJECT_NAME}_CHOLMOD_Test PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_CHOLMOD_Test PUBLIC CLI11::CLI11)

# Testing Solver kernel
#add_executable(${PROJECT_NAME}_ACCELERATE_Test Accelerate_test.cpp)
#target_link_libraries(${PROJECT_NAME}_ACCELERATE_Test PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ACCELERATE_Test PUBLIC CLI11::CLI11)
#
## Testing Accelerate with Parth kernel
#add_executable(${PROJECT_NAME}_Smoothing Accelerate_Parth_test.cpp Laplacian_Smoother.cpp)
#target_link_libraries(${PROJECT_NAME}_Smoothing PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_Smoothing PUBLIC CLI11::CLI11)
#
## Testing Accelerate with Parth kernel
#add_executable(${PROJECT_NAME}_PAPER_TEST paper_examples.cpp)
#target_link_libraries(${PROJECT_NAME}_PAPER_TEST PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_PAPER_TEST PUBLIC CLI11::CLI11)
#
## Testing Accelerate with Parth kernel
#add_executable(${PROJECT_NAME}_TEASER Teaser.cpp)
#target_link_libraries(${PROJECT_NAME}_TEASER PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_TEASER PUBLIC CLI11::CLI11)


#add_executable(${PROJECT_NAME}_EdgeCollapsing Edge_Collapsing.cpp subdivision.cpp)
#target_link_libraries(${PROJECT_NAME}_EdgeCollapsing PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_EdgeCollapsing PUBLIC CLI11::CLI11)
#
#add_executable(${PROJECT_NAME}_TETGEN tetgen.cpp)
#target_link_libraries(${PROJECT_NAME}_TETGEN PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_TETGEN PUBLIC CLI11::CLI11)

#add_executable(${PROJECT_NAME}_REMESHER remesher.cpp)
#target_link_libraries(${PROJECT_NAME}_REMESHER PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_REMESHER PUBLIC CLI11::CLI11)

#add_executable(${PROJECT_NAME}_SUBDIVISION subdivision.cpp)
#target_link_libraries(${PROJECT_NAME}_SUBDIVISION PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_SUBDIVISION PUBLIC CLI11::CLI11)
#
#add_executable(${PROJECT_NAME}_ParthTEST ParthTestAndDebug.cpp)
#target_link_libraries(${PROJECT_NAME}_ParthTEST PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ParthTEST PUBLIC CLI11::CLI11)
#
#add_executable(${PROJECT_NAME}_Slice Slice.cpp)
#target_link_libraries(${PROJECT_NAME}_Slice PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_Slice PUBLIC CLI11::CLI11)
#
#add_executable(${PROJECT_NAME}_Demo ParthDemo.cpp)
#target_link_libraries(${PROJECT_NAME}_Demo PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_Demo PUBLIC CLI11::CLI11)


#add_executable(${PROJECT_NAME}_Demo ParthFullDemo.cpp)
#target_link_libraries(${PROJECT_NAME}_Demo PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_Demo PUBLIC CLI11::CLI11)


#add_executable(${PROJECT_NAME}_FullDecimateAndSubdivison FullDecimateAndSubdivision.cpp)
#target_link_libraries(${PROJECT_NAME}_FullDecimateAndSubdivison PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_FullDecimateAndSubdivison PUBLIC CLI11::CLI11)
#
#add_executable(${PROJECT_NAME}_ChangeRes MeshResolutionAdjustment.cpp)
#target_link_libraries(${PROJECT_NAME}_ChangeRes PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ChangeRes PUBLIC CLI11::CLI11)


##
#
#add_executable(${PROJECT_NAME}_FullRemeshDemo  FullRemeshDemo.cpp)
#target_link_libraries(${PROJECT_NAME}_FullRemeshDemo PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_FullRemeshDemo PUBLIC CLI11::CLI11)


#add_executable(${PROJECT_NAME}_FullSmoothingDemo  SmoothingRemeshingDemo.cpp)
#target_link_libraries(${PROJECT_NAME}_FullSmoothingDemo PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_FullSmoothingDemo PUBLIC CLI11::CLI11)


#add_executable(${PROJECT_NAME}_SurfaceMapsEval  SurfaceMaps_Evaluation.cpp)
#target_link_libraries(${PROJECT_NAME}_SurfaceMapsEval PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_SurfaceMapsEval PUBLIC CLI11::CLI11)
#
#

#Benchmarking
add_executable(${PROJECT_NAME}_PatchRemeshDemo  PatchRemeshDemo.cpp)
target_link_libraries(${PROJECT_NAME}_PatchRemeshDemo PUBLIC ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_PatchRemeshDemo PUBLIC CLI11::CLI11)
#
add_executable(${PROJECT_NAME}_IPCBenchmark  IPCBenchmark.cpp)
target_link_libraries(${PROJECT_NAME}_IPCBenchmark PUBLIC ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_IPCBenchmark PUBLIC CLI11::CLI11)

#add_executable(${PROJECT_NAME}_ArcSimBenchmark  ArcSimBenchmark.cpp)
#target_link_libraries(${PROJECT_NAME}_ArcSimBenchmark PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ArcSimBenchmark PUBLIC CLI11::CLI11)

#add_executable(${PROJECT_NAME}_SurfaceMapsBenchmark  SurfaceMapsBenchmark.cpp)
#target_link_libraries(${PROJECT_NAME}_SurfaceMapsBenchmark PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_SurfaceMapsBenchmark PUBLIC CLI11::CLI11)

#Limitation
#add_executable(${PROJECT_NAME}_remeshingLimit RemesherLimitationSection.cpp)
#target_link_libraries(${PROJECT_NAME}_remeshingLimit PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_remeshingLimit PUBLIC CLI11::CLI11)

#LAG test
#add_executable(${PROJECT_NAME}_LagTest Paper_LagTest.cpp)
#target_link_libraries(${PROJECT_NAME}_LagTest PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_LagTest PUBLIC CLI11::CLI11)

#Teaser
add_executable(${PROJECT_NAME}_TeaserIPC teaserIPC.cpp)
target_link_libraries(${PROJECT_NAME}_TeaserIPC PUBLIC ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_TeaserIPC PUBLIC CLI11::CLI11)

#add_executable(${PROJECT_NAME}_ArcSimTeaser ArcSimTeaser.cpp)
#target_link_libraries(${PROJECT_NAME}_ArcSimTeaser PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_ArcSimTeaser PUBLIC CLI11::CLI11)

#add_executable(${PROJECT_NAME}_remeshingTeaser teaserRemesherV2.cpp)
#target_link_libraries(${PROJECT_NAME}_remeshingTeaser PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_remeshingTeaser PUBLIC CLI11::CLI11)
#
#add_executable(${PROJECT_NAME}_OpenCloseTeaser teaserOpenClose.cpp)
#target_link_libraries(${PROJECT_NAME}_OpenCloseTeaser PUBLIC ${PROJECT_NAME}_lib)
#target_link_libraries(${PROJECT_NAME}_OpenCloseTeaser PUBLIC CLI11::CLI11)



#Rebuttal
add_executable(${PROJECT_NAME}_ParthCompressionTest  ParthCompressionTest.cpp)
target_link_libraries(${PROJECT_NAME}_ParthCompressionTest PUBLIC ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_ParthCompressionTest PUBLIC CLI11::CLI11)
################################################################################

