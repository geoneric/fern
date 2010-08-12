//! Dummy algorith that returns Result(0).
/*!
  \param     value Input value that is not used in the calculation.
  \return    Result(0)
  \sa        setNull(Argument const& value)

  Because of how the local function library is designed, this function is
  only called for those cells that do not contain a no-data value. For all
  these cases false (Result(0)) needs to be returned.
*/
template<typename Argument, typename Result>
inline Result isNull(
         Argument const& /* value */)
{
  return Result(0);
}

