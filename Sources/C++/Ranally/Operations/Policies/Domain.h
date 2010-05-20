#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES
#define INCLUDED_RANALLY_OPERATIONS_POLICIES



namespace ranally {
namespace operations {
namespace policies {

template<typename T>
class DummyDomain
{
  static inline inDomain(
         T /* argument */)
  {
    return true;
  }

  static inline inDomain(
         T /* argument1 */,
         T /* argument2 */)
  {
    return true;
  }
};

} // namespace policies
} // namespace operations
} // namespace ranally

#endif
