template<typename T>
struct IgnoreNoData
{
  static inline void setNoData(
         T& /* value */)
  {
    // No-op.
  }

  static inline void setNoData(
         std::vector<bool>::reference /* value */)
  {
    // No-op.
  }
};

//! Policy class which assigns a certain value to a result.
/*!
  \tparam    T Type of value to assign.
  \warning   setNoDataValue(T const&) must have been called first, before
             this policy can do anything useful.
*/
template<typename T>
class NoDataValue
{
private:

  //! Value to assign.
  T _value;

public:

  //! Assigns the layered value to \a value.
  /*!
    \param     value Variable to assign to.
  */
  inline void setNoData(
         T& value)
  {
    value = _value;
  }

  //! Sets the value to assign to \a value.
  /*!
    \param     value Value assigned to results.
  */
  void setNoDataValue(
         T const& value)
  {
    _value = value;
  }
};
