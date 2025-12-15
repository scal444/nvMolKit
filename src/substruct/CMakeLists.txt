add_library(molecule_graph_labels molecule_graph_labels.cu)

add_library(molecules molecules.cpp)
target_include_directories(molecule_graph_labels
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(molecules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  molecules
  PUBLIC device_vector
  PRIVATE ${RDKit_LIBS})
