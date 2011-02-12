#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_DOMAIN
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_DOMAIN



namespace ranally {
namespace operations {
namespace policies {

template<typename T>
struct DummyDomain
{
  inline static bool inDomain(
         T /* argument */)
  {
    // No-op.
    return true;
  }

  inline static bool inDomain(
         T /* argument1 */,
         T /* argument2 */)
  {
    // No-op.
    return true;
  }
};

} // namespace policies
} // namespace operations
} // namespace ranally

#endif
