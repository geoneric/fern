#pragma once
#include "geoneric/uncertainty/distribution.h"
#include <boost/random/normal_distribution.hpp>


namespace geoneric {

//! Class models uncertainty using a normal distribution.
/*!
*/
template<
    typename T>
class NormalDistribution:
    public Distribution
{

public:

                   NormalDistribution  (T mean,
                                        T standard_deviation);

                   ~NormalDistribution ()=default;

                   NormalDistribution  (NormalDistribution&&)=default;

    NormalDistribution& operator=      (NormalDistribution&&)=default;

                   NormalDistribution  (NormalDistribution const&)=default;

    NormalDistribution& operator=      (NormalDistribution const&)=default;

    T              mean                () const;

    T              standard_deviation  () const;

    template<class RandomNumberGenerator>
    T              operator()          (RandomNumberGenerator&
                                           random_number_generator) const;

private:

    mutable boost::random::normal_distribution<T> _distribution;

};


template<
    typename T>
template<
    class RandomNumberGenerator>
inline T NormalDistribution<T>::operator()(
    RandomNumberGenerator& random_number_generator) const
{
    return _distribution(random_number_generator);
}

} // namespace geoneric
