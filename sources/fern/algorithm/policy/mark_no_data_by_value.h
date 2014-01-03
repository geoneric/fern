#pragma once


namespace fern {

template<
    class T,
    class Mask>
class MarkNoDataByValue {

public:

    void           init_no_data_policy (Mask& mask,
                                        T const& no_data_value);

    void           mark                (size_t index);

protected:

                   MarkNoDataByValue   ()=default;

                   MarkNoDataByValue   (MarkNoDataByValue&&)=delete;

    MarkNoDataByValue&
                   operator=           (MarkNoDataByValue&&)=delete;

                   MarkNoDataByValue   (MarkNoDataByValue const&)=delete;

    MarkNoDataByValue&
                   operator=           (MarkNoDataByValue const&)=delete;

                   ~MarkNoDataByValue  ()=default;

private:

    Mask           _mask;

    T              _no_data_value;

};


template<
    class T,
    class Mask>
inline void MarkNoDataByValue<T, Mask>::init_no_data_policy(
    Mask& mask,
    T const& no_data_value)
{
    _mask = mask;
    _no_data_value = no_data_value;
}


template<
    class T,
    class Mask>
inline void MarkNoDataByValue<T, Mask>::mark(
    size_t index)
{
    _mask.set(index, _no_data_value);
}

} // namespace fern
