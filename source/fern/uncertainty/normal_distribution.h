// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/uncertainty/distribution.h"
#include <boost/random/normal_distribution.hpp>


namespace fern {

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

} // namespace fern
