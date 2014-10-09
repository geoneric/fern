#pragma once
#include "fern/core/string.h"
#include "fern/io/core/domain.h"


namespace fern {

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

} // namespace fern
