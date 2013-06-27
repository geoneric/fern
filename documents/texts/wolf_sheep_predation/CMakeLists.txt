ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/data_model.pdf
    COMMAND ${DOXYGEN_DOT_EXECUTABLE} -Tpdf -o${CMAKE_CURRENT_BINARY_DIR}/data_model.pdf ${CMAKE_CURRENT_SOURCE_DIR}/data_model.dot
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data_model.dot
)
ADD_CUSTOM_TARGET(data_model_pdf
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/data_model.pdf)

ADD_LATEX_DOCUMENT(wolf_sheep_predation.tex
    CONFIGURE wolf_sheep_predation.tex
    DEFAULT_PDF
    MANGLE_TARGET_NAMES
)

ADD_DEPENDENCIES(wolf_sheep_predation data_model_pdf)
