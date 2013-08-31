#pragma once
#include <set>
#include "ranally/feature/fid_map.h"
#include "ranally/feature/value.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FeatureDomainValue:
    public Value,
    public FidMap<std::set<Fid>>
{

public:

    template<class Domain>
                   FeatureDomainValue  (
                                   Domain const& domain,
                                   std::shared_ptr<Feature> const& feature);

                   FeatureDomainValue  (FeatureDomainValue const&)=delete;

    FeatureDomainValue&   operator=    (FeatureDomainValue const&)=delete;

                   FeatureDomainValue  (FeatureDomainValue&&)=delete;

    FeatureDomainValue&   operator=    (FeatureDomainValue&&)=delete;

                   ~FeatureDomainValue ();

private:

    std::shared_ptr<Feature> _feature;

};


template<
    class Domain>
inline FeatureDomainValue::FeatureDomainValue(
    Domain const& domain,
    std::map<Fid, std::set<Fid>> const& relation,
    std::shared_ptr<Feature> const& feature)

    : Value(),
      FidMap<std::set<Fid>>(),
      _feature(feature)

{
    for(typename Domain::value_type const& value: domain) {
        Fid one(value.first);
        std::set<Fid> const& many((*relation.find(value.first)).second);

        assert(relation.find(one) != relation.end());
#ifndef NDEBUG
        for(Fid fid: many) {
            assert(feature->domain()->find(fid) != feature->domain()->end());
        }
#endif

        this->insert(one, many);
    }
}

} // namespace ranally
