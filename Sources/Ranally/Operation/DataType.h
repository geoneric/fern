#ifndef INCLUDED_RANALLY_OPERATION_DATATYPE
#define INCLUDED_RANALLY_OPERATION_DATATYPE

namespace ranally {
namespace operation {

enum DataType {
  DT_UNKNOWN=0,
  DT_NUMBER=1,
  DT_STRING=2,
  DT_RASTER=4,
  DT_FEATURE=8,
  DT_SPATIAL=DT_RASTER | DT_FEATURE,
  DT_ALL=DT_NUMBER | DT_STRING | DT_RASTER | DT_FEATURE
};

typedef unsigned int DataTypes;

} // namespace operation
} // namespace ranally

#endif
