add_library(molecule_graph_labels molecule_graph_labels.cu)

add_library(molecules molecules.cpp)
target_include_directories(molecule_graph_labels
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(molecules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  molecules
  PUBLIC device_vector flatBitVect OpenMP::OpenMP_CXX
  PRIVATE ${RDKit_LIBS} nvtx)

add_library(pinned_buffer_pool pinned_buffer_pool.cpp)
target_include_directories(pinned_buffer_pool
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pinned_buffer_pool PUBLIC cuda_error_check)

add_library(substructure_search substructure_search.cu)
target_include_directories(substructure_search
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(substructure_search PUBLIC molecules device_vector
                                                  flatBitVect cuda_error_check
                                                  pinned_buffer_pool device_timings
                                                  OpenMP::OpenMP_CXX
                                          PRIVATE ${RDKit_LIBS})
target_compile_options(substructure_search PRIVATE -Xptxas=-v $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)