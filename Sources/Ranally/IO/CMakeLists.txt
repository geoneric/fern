INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
SET(SOURCES
  Attribute.cc
  DataSet.cc
  DataSetDriver.cc
  Domain.cc
  Feature.cc
  HDF5Client.cc
  HDF5DataSet.cc
  HDF5DataSetDriver.cc
  OGRClient.cc
  OGRDataSet.cc
  OGRDataSetDriver.cc
  OGRFeatureLayer.cc
  PointAttribute.cc
  PointDomain.cc
  PointFeature.cc
  PointValue.cc
  PolygonAttribute.cc
  PolygonDomain.cc
  PolygonFeature.cc
  PolygonValue.cc
  RasterAttribute.cc
  RasterValue.cc
  SpatialDomain.cc
  SpatioTemporalDomain.cc
  TemporalDomain.cc
  Util.cc
  Value.cc
)
ADD_LIBRARY(RanallyIO
  ${SOURCES}
)
SET(TEST_SOURCES
  HDF5DataSetDriverTest.cc
  HDF5DataSetTest.cc
  OGRDataSetDriverTest.cc
  Test.cc
)
ADD_EXECUTABLE(testRanallyIO
  ${TEST_SOURCES}
)
TARGET_LINK_LIBRARIES(testRanallyIO
  RanallyIO
  RanallyUtil
  ${ICU_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${GDAL_LIBRARIES}
  ${BOOST_FILESYSTEM_LIBRARY}
  ${BOOST_SYSTEM_LIBRARY}
)
ADD_CUSTOM_TARGET(runTestRanallyIO
  testRanallyIO
  DEPENDS testRanallyIO
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running RanallyIO unit tests" VERBATIM
)
ADD_CUSTOM_TARGET(copyTestData
  COMMAND ${CMAKE_COMMAND} -E copy Point.json ${CMAKE_CURRENT_BINARY_DIR}

  COMMAND ${CMAKE_COMMAND} -E copy Point.json
    ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyPoint.json
  COMMAND chmod 444 ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyPoint.json

  COMMAND ${CMAKE_COMMAND} -E copy Point.json
    ${CMAKE_CURRENT_BINARY_DIR}/WriteOnlyPoint.json
  COMMAND chmod 222 ${CMAKE_CURRENT_BINARY_DIR}/WriteOnlyPoint.json

  COMMAND ${CMAKE_COMMAND} -E copy Point.json
    ${CMAKE_CURRENT_BINARY_DIR}/ExecuteOnlyPoint.json
  COMMAND chmod 111 ${CMAKE_CURRENT_BINARY_DIR}/ExecuteOnlyPoint.json

  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyDir
  COMMAND chmod 666 ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyDir

  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Data/Tests
  COMMENT "Copying test data" VERBATIM
)
ADD_DEPENDENCIES(runTestRanallyIO
  copyTestData
)

