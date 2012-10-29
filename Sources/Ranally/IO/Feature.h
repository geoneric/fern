#pragma once
#include "Ranally/IO/Domain.h"
#include "Ranally/Util/String.h"


namespace ranally {

//! Class for Feature instances combining Domain with an Attribute.
/*!
  \sa        .
*/
class Feature
{

    friend class FeatureTest;

public:

                   Feature             (Feature const&)=delete;

    Feature&       operator=           (Feature const&)=delete;

    virtual        ~Feature            ();

    String const&  name                () const;

    Domain::Type   domainType          () const;

protected:

                   Feature             (String const& name,
                                        Domain::Type domainType);

private:

    String         _name;

    Domain::Type   _domainType;

};

} // namespace ranally
