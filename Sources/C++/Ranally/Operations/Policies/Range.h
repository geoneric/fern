#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_RANGE
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_RANGE



namespace ranally {
namespace operations {
namespace policies {

template<typename T>
struct DummyRange
{
  inline static bool inRange(
         T /* argument */,
         T /* result */)
  {
    // No-op.
    return true;
  }
};

} // namespace policies
} // namespace operations
} // namespace ranally

#endif
