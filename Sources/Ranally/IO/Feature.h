#pragma once
#include <boost/noncopyable.hpp>
#include "Ranally/IO/Domain.h"
#include "Ranally/Util/String.h"



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

  String const&    name                () const;

  Domain::Type     domainType          () const;

protected:

                   Feature             (String const& name,
                                        Domain::Type domainType);

private:

  String           _name;

  Domain::Type     _domainType;

};

} // namespace ranally
