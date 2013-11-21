set(SZIP_FOUND FALSE)

find_library(SZIP_LIBRARY
  NAMES szip
  PATHS
  /usr/lib
  /usr/lib/odbc
  /usr/local/lib
  /usr/local/lib/odbc
  /usr/local/odbc/lib
  "C:/Program Files/SZIP/lib"
  "C:/SZIP/lib/debug"
  "C:/Program Files/Microsoft SDKs/Windows/v7.0A/Lib"
  "C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib"
  DOC "Specify the SZIP driver manager library here."
)

if(SZIP_LIBRARY)
  # if(SZIP_INCLUDE_DIR)
    set( SZIP_FOUND 1 )
  # endif()
endif()

set(SZIP_LIBRARIES ${SZIP_LIBRARY})

mark_as_advanced(SZIP_FOUND SZIP_LIBRARY SZIP_EXTRA_LIBRARIES SZIP_INCLUDE_DIR)
