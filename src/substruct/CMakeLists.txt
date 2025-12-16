add_library(molecule_graph_labels molecule_graph_labels.cu)

add_library(molecules molecules.cpp)
target_include_directories(molecule_graph_labels
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(molecules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  molecules
  PUBLIC device_vector flatBitVect
  PRIVATE ${RDKit_LIBS})

add_library(substructure_search substructure_search.cu)
target_include_directories(substructure_search
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(substructure_search PUBLIC molecules device_vector
                                                  flatBitVect cuda_error_check)
