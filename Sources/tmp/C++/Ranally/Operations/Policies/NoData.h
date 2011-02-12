#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_NODATA
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_NODATA



namespace ranally {
namespace operations {
namespace policies {

template<typename T>
struct IgnoreNoData
{
  inline static void setNoData(
         T& /* noData */)
  {
    // No-op.
  }

  inline static bool isNoData(
         T /* noData */)
  {
    // No-op.
    return false;
  }
};

template<typename T>
class TestNoDataValue
{
private:

  T      _noDataValue;

public:

  // void setNoDataValue(
  //        T value)
  // {
  //   _noDataValue = value;
  // }

  // inline void setNoData(
  //        T& noData) const
  // {
  //   noData = _noDataValue;
  // }

  // inline bool isNoData(
  //        T noData) const
  // {
  //   return noData == _noDataValue;
  // }
};

template<>
class TestNoDataValue<bool>
{
public:

  inline static void setNoData(
         bool& noData)
  {
    noData = true;
  }

  inline static bool isNoData(
         bool noData)
  {
    return noData;
  }
};

} // namespace policies
} // namespace operations
} // namespace ranally

#endif
