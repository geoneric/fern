#ifndef INCLUDED_RANALLY_FEATURE
#define INCLUDED_RANALLY_FEATURE

#include <boost/noncopyable.hpp>



namespace ranally {

//! Class for Feature instances combining Domain with an Attribute.
/*!
  \sa        .
*/
class Feature:
  private boost::noncopyable
{

  friend class FeatureTest;

public:

  virtual          ~Feature            ();

protected:

                   Feature             ();

private:

};

} // namespace ranally

#endif
