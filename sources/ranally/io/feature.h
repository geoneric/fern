#pragma once
#include "ranally/core/string.h"
#include "ranally/io/domain.h"


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

    Domain::Type   domain_type         () const;

protected:

                   Feature             (String const& name,
                                        Domain::Type domain_type);

private:

    String         _name;

    Domain::Type   _domain_type;

};

} // namespace ranally
