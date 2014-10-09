#pragma once


namespace fern {

template<
    typename T>
class MaskedConstant
{

public:

                   MaskedConstant      ();

    explicit       MaskedConstant      (T const& value);

                   MaskedConstant      (T const& value,
                                        bool mask);

                   MaskedConstant      (MaskedConstant const&)=default;

    MaskedConstant& operator=          (MaskedConstant const&)=default;

    MaskedConstant& operator=          (T const& value);

                   MaskedConstant      (MaskedConstant&&)=default;

    MaskedConstant& operator=          (MaskedConstant&&)=default;

                   ~MaskedConstant     ()=default;

    T const&       value               () const;

    T&             value               ();

    bool const&    mask                () const;

    bool&          mask                ();

private:

    T              _value;

    bool           _mask;

};


template<
    typename T>
inline MaskedConstant<T>::MaskedConstant()

    : MaskedConstant(T(), false)

{
}


template<
    typename T>
inline MaskedConstant<T>::MaskedConstant(
    T const& value)

    : MaskedConstant(value, false)

{
}


template<
    typename T>
inline MaskedConstant<T>::MaskedConstant(
    T const& value,
    bool mask)

    : _value(value),
      _mask(mask)

{
}


template<
    typename T>
inline MaskedConstant<T>& MaskedConstant<T>::operator=(
    T const& value)
{
    _value = value;

    return *this;
}


template<
    typename T>
inline T const& MaskedConstant<T>::value() const
{
    // assert(!_mask);
    return _value;
}


template<
    typename T>
inline T& MaskedConstant<T>::value()
{
    // assert(!_mask);
    return _value;
}


template<
    typename T>
inline bool const& MaskedConstant<T>::mask() const
{
    return _mask;
}


template<
    typename T>
inline bool& MaskedConstant<T>::mask()
{
    return _mask;
}


template<
    typename T>
inline bool operator==(
    MaskedConstant<T> const& lhs,
    MaskedConstant<T> const& rhs)
{
    return
        lhs.mask() == rhs.mask() &&
        lhs.value() == rhs.value()
        ;
}

} // namespace fern