#ifndef INCLUDED_RANALLY_FEATURE
#define INCLUDED_RANALLY_FEATURE

#include <boost/noncopyable.hpp>
#include "Ranally/IO/Domain.h"



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

  Domain::Type     domainType          () const;

protected:

                   Feature             (Domain::Type domainType);

private:

  Domain::Type     _domainType;

};

} // namespace ranally

#endif
