#ifndef INCLUDED_RANALLY_FEATURE
#define INCLUDED_RANALLY_FEATURE

#include <unicode/unistr.h>
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

  UnicodeString const& name            () const;

  Domain::Type     domainType          () const;

protected:

                   Feature             (UnicodeString const& name,
                                        Domain::Type domainType);

private:

  UnicodeString    _name;

  Domain::Type     _domainType;

};

} // namespace ranally

#endif
