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

  inline void setNoData(
         bool& noData) const
  {
    noData = true;
  }

  inline bool isNoData(
         bool noData) const
  {
    return noData;
  }
};

} // namespace policies
} // namespace operations
} // namespace ranally

#endif
