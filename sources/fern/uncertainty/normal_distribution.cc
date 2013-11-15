#include "fern/uncertainty/normal_distribution.h"


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
