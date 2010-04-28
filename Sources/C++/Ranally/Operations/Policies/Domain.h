#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES
#define INCLUDED_RANALLY_OPERATIONS_POLICIES



namespace Ranally {
namespace Operations {
namespace Policies {

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

} // namespace Policies
} // namespace Operations
} // namespace Ranally

#endif
