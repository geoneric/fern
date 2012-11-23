#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_CLIPMASK
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_CLIPMASK



namespace ranally {
namespace operations {
namespace policies {

template<typename T>
class TestClipMaskValue
{
private:

  T      _clipMaskValue;

public:

};

template<>
struct TestClipMaskValue<bool>
{
public:

  inline static bool mask(
         bool& clipMask)
  {
    return clipMask;
  }
};

} // namespace policies
} // namespace operations
} // namespace ranally

#endif
