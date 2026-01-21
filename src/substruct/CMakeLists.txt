add_library(packed_bonds packed_bonds.cpp)
target_include_directories(packed_bonds PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(molecules molecules.cpp)
target_include_directories(molecules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  molecules
  PUBLIC device_vector flatBitVect OpenMP::OpenMP_CXX packed_bonds
  PRIVATE ${RDKit_LIBS} nvtx)

add_library(pinned_buffer_pool pinned_buffer_pool.cpp)
target_include_directories(pinned_buffer_pool
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pinned_buffer_pool PUBLIC cuda_error_check)

add_library(substruct_kernels substruct_kernels.cu)
target_include_directories(substruct_kernels
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(substruct_kernels PUBLIC molecules device_vector
                                                flatBitVect cuda_error_check
                                                device_timings)
target_compile_options(substruct_kernels PRIVATE -Xptxas=-v)

add_library(substructure_search substructure_search.cu substruct_launch_config.cpp)
target_include_directories(substructure_search
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(substructure_search PUBLIC molecules device_vector
                                                  flatBitVect cuda_error_check
                                                  pinned_buffer_pool device_timings
                                                  device substruct_kernels OpenMP::OpenMP_CXX
                                          PRIVATE ${RDKit_LIBS})
target_compile_options(substructure_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
                                                  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
