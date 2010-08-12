//! Dummy algorithm that just returns its argument.
/*!
  \param     value Value that is returned.
  \return    \a value
  \sa        isNull(Argument const& value)

  Because of how the local function library is designed, this function is
  only called for those cells that do not need to be set to a no-data value.
  For all these cells the input value should be returned.
*/
template<typename Argument, typename Result>
inline Result setNull(
         Argument const& value)
{
  return value;
}

