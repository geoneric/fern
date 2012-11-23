#pragma once
#include <memory>
#include "ranally/util/string.h"


class OGRLayer;

namespace ranally {

class Domain;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRFeatureLayer
{

    friend class OGRFeatureLayerTest;

public:

                   OGRFeatureLayer     (OGRFeatureLayer const&)=delete;

    OGRFeatureLayer& operator=         (OGRFeatureLayer const&)=delete;

                   OGRFeatureLayer     (OGRLayer* const layer);

                   ~OGRFeatureLayer    ();

    String         name                () const;

    Domain const&  domain              () const;

    template<class Domain>
    std::shared_ptr<Domain> domain     () const;

private:

    OGRLayer* const  _layer;

    std::shared_ptr<Domain> _domain;

};


template<class Domain>
std::shared_ptr<Domain> OGRFeatureLayer::domain() const
{
    return std::dynamic_pointer_cast<Domain>(_domain);
}

} // namespace ranally
