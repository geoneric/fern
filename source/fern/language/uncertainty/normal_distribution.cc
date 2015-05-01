// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/uncertainty/normal_distribution.h"


namespace fern {

template<
    typename T>
NormalDistribution<T>::NormalDistribution(
    T mean,
    T standard_deviation)

    : _distribution(mean, standard_deviation)

{
}


template<
    typename T>
T NormalDistribution<T>::mean() const
{
    return _distribution.mean();
}


template<
    typename T>
T NormalDistribution<T>::standard_deviation() const
{
    return _distribution.sigma();
}


template class NormalDistribution<float>;
template class NormalDistribution<double>;

} // namespace fern
