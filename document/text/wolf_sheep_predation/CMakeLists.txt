add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/data_model.pdf
    COMMAND ${DOXYGEN_DOT_EXECUTABLE} -Tpdf -o${CMAKE_CURRENT_BINARY_DIR}/data_model.pdf ${CMAKE_CURRENT_SOURCE_DIR}/data_model.dot
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data_model.dot
)
add_custom_target(data_model_pdf
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/data_model.pdf)

add_latex_document(wolf_sheep_predation.tex
    CONFIGURE wolf_sheep_predation.tex
    DEFAULT_PDF
    MANGLE_TARGET_NAMES
)

add_dependencies(wolf_sheep_predation_pdf data_model_pdf)
