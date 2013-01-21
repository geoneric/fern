#pragma once
#include <memory>
#include "ranally/feature/domain.h"
#include "ranally/feature/fid_map.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class Geometry>
class FeatureDomain:
    public Domain,
    public FidMap<Geometry>
{

public:

                   FeatureDomain       ();

                   FeatureDomain       (FeatureDomain const&)=delete;

    FeatureDomain& operator=           (FeatureDomain const&)=delete;

                   FeatureDomain       (FeatureDomain&&)=delete;

    FeatureDomain& operator=           (FeatureDomain&&)=delete;

                   ~FeatureDomain      ();

private:

};


template<
    class Geometry>
inline FeatureDomain<Geometry>::FeatureDomain()

    : Domain(),
      FidMap<Geometry>()

{
}


template<
    class Geometry>
inline FeatureDomain<Geometry>::~FeatureDomain()
{
}

} // namespace ranally
