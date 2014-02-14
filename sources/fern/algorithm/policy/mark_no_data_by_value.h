#pragma once


namespace fern {

template<
    class T,
    class Mask>
class MarkNoDataByValue {

public:

    bool           is_no_data          (size_t index) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2,
                                        size_t index3) const;

    void           mark_as_no_data     (size_t index);

    void           mark_as_no_data     (size_t index1,
                                        size_t index2);

    void           mark_as_no_data     (size_t index1,
                                        size_t index2,
                                        size_t index3);

                   MarkNoDataByValue   (Mask& mask,
                                        T const& no_data_value);

protected:

                   MarkNoDataByValue   ()=delete;

                   MarkNoDataByValue   (MarkNoDataByValue&&)=default;

    MarkNoDataByValue&
                   operator=           (MarkNoDataByValue&&)=default;

    // This class keeps a reference to a mask to update. This class doesn't
    // copy the mask: copy construction and copy assignment are not supported.

                   MarkNoDataByValue   (MarkNoDataByValue const&)=delete;

    MarkNoDataByValue&
                   operator=           (MarkNoDataByValue const&)=delete;

                   ~MarkNoDataByValue  ()=default;

private:

    Mask&          _mask;

    T              _no_data_value;

};


template<
    class T,
    class Mask>
inline MarkNoDataByValue<T, Mask>::MarkNoDataByValue(
    Mask& mask,
    T const& no_data_value)

    : _mask(mask),
      _no_data_value(no_data_value)

{
}


template<
    class T,
    class Mask>
inline bool MarkNoDataByValue<T, Mask>::is_no_data(
    size_t index) const
{
    return get(_mask, index);
}


template<
    class T,
    class Mask>
inline bool MarkNoDataByValue<T, Mask>::is_no_data(
    size_t index1,
    size_t index2) const
{
    return get(_mask, index1, index2);
}


template<
    class T,
    class Mask>
inline bool MarkNoDataByValue<T, Mask>::is_no_data(
    size_t index1,
    size_t index2,
    size_t index3) const
{
    return get(_mask, index1, index2, index3);
}


template<
    class T,
    class Mask>
inline void MarkNoDataByValue<T, Mask>::mark_as_no_data(
    size_t index)
{
    get(_mask, index) = _no_data_value;
}


template<
    class T,
    class Mask>
inline void MarkNoDataByValue<T, Mask>::mark_as_no_data(
    size_t index1,
    size_t index2)
{
    get(_mask, index1, index2) = _no_data_value;
}


template<
    class T,
    class Mask>
inline void MarkNoDataByValue<T, Mask>::mark_as_no_data(
    size_t index1,
    size_t index2,
    size_t index3)
{
    get(_mask, index1, index2, index3) = _no_data_value;
}

} // namespace fern
