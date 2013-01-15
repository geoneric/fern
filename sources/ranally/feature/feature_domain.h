#pragma once
#include <memory>
#include <vector>
#include "ranally/feature/domain.h"
#include "ranally/feature/geometry.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<class Model>
class FeatureDomain:
    public Domain
{

public:

                   FeatureDomain       ()=default;

                   FeatureDomain       (FeatureDomain const&)=delete;

    FeatureDomain& operator=           (FeatureDomain const&)=delete;

                   FeatureDomain       (FeatureDomain&&)=delete;

    FeatureDomain& operator=           (FeatureDomain&&)=delete;

                   ~FeatureDomain      () noexcept(true)=default;

    void           append              (Model const& model);

    std::vector<Model> const& geometry () const;

private:

    std::vector<Model> _geometry;

};


template<class Model>
void FeatureDomain<Model>::append(
    Model const& model)
{
    _geometry.push_back(model);
}


template<class Model>
inline std::vector<Model> const& FeatureDomain<Model>::geometry() const
{
    return _geometry;
}

} // namespace ranally
