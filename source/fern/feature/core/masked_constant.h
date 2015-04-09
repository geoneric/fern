// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {

/*!
    @ingroup    fern_feature_group
    @brief      A value that can be masked.

    When adding two numbers together, the result may be a larger value then
    can be represented by the type. A MaskedConstant can be used to mark
    the result as no-data in such cases.
*/
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

                   operator T const&   () const;

                   operator T&         ();

    T const&       value               () const;

    T&             value               ();

    bool const&    mask                () const;

    bool&          mask                ();

private:

    T              _value;

    bool           _mask;

};


/*!
    @brief      Default construct an instance.

    The layered number is default constructed and the mask is set to false.
*/
template<
    typename T>
inline MaskedConstant<T>::MaskedConstant()

    : MaskedConstant(T(), false)

{
}


/*!
    @brief      Construct an instance.

    The layered number is set to @a value and the mask is set to false.
*/
template<
    typename T>
inline MaskedConstant<T>::MaskedConstant(
    T const& value)

    : MaskedConstant(value, false)

{
}


/*!
    @brief      Construct an instance.

    The layered number is set to @a value and the mask is set to @a mask.
*/
template<
    typename T>
inline MaskedConstant<T>::MaskedConstant(
    T const& value,
    bool mask)

    : _value(value),
      _mask(mask)

{
}


/*!
    @brief      Assign @a value to the instance.

    The layered number is set to @a value and the mask is set to false.
*/
template<
    typename T>
inline MaskedConstant<T>& MaskedConstant<T>::operator=(
    T const& value)
{
    _value = value;
    _mask = false;

    return *this;
}


/*!
    @brief      Return a const reference to the layered number.
*/
template<
    typename T>
inline MaskedConstant<T>::operator T const&() const
{
    return _value;
}


/*!
    @brief      Return a reference to the layered number.
*/
template<
    typename T>
inline MaskedConstant<T>::operator T&()
{
    return _value;
}


/*!
    @brief      Return a const reference to the layered number.
*/
template<
    typename T>
inline T const& MaskedConstant<T>::value() const
{
    // assert(!_mask);
    return _value;
}


/*!
    @brief      Return a reference to the layered number.
*/
template<
    typename T>
inline T& MaskedConstant<T>::value()
{
    // assert(!_mask);
    return _value;
}


/*!
    @brief      Return a const reference to the layered mask.
*/
template<
    typename T>
inline bool const& MaskedConstant<T>::mask() const
{
    return _mask;
}


/*!
    @brief      Return a reference to the layered mask.
*/
template<
    typename T>
inline bool& MaskedConstant<T>::mask()
{
    return _mask;
}


/*!
    @brief      Return whether @a lhs equals @a rhs.

    True is returned if both the layered number and the mask are equal.
*/
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
