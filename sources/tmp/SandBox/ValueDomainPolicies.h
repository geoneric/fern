#ifndef INCLUDED_BOOST_FUNCTION
#include <boost/function.hpp>
#define INCLUDED_BOOST_FUNCTION
#endif



//! Dummy value domain policy.
/*!
  \tparam    T Type of input value.

  This policy can be used when domain checks are not wanted.
*/
template<typename T>
struct DontCheckValueDomain
{

  //! Checks whether \a value falls within the valid domain of values.
  /*!
    \param     value Input value to check.
    \return    true

    All values are assumed to be within the valid value domain.
  */
  static inline bool inDomain(
         T const& value)
  {
    return true;
  }
};



//! Value domain policy using a unary predicate.
/*!
  \tparam    T Type of input value.

  A unary predicate is used to decide whether a value is within the domain
  of valid values or not. If the predicate returns true, the value is assumed
  to be within the valid domain.

  With this policy very simple and very complex domains can be configured:
  - Even values.
  - Odd values.
  - Values larger than x and smaller than y.
  - ...

  A predicate must be set using setDomainPredicate(Predicate const&) before
  this policy can be used. If not, a
  "std::runtime_error: call to empty boost::function" is thrown.
*/
template<typename T>
struct DomainByPredicate
{
private:

  //! Type of function to use for checking values. Unary predicate.
  typedef boost::function<bool (T)> Predicate;

  //! Copy of predicate.
  Predicate        _predicate;

public:

  //! Set the predicate to use to \a predicate.
  /*!
    \param     predicate Predicat to use.
  */
  void setDomainPredicate(
         Predicate const& predicate)
  {
    _predicate = predicate;
  }

  //! Checks whether \a value falls within the valid domain of values.
  /*!
    \param     value Input value to check.
    \return    The result of calling the predicate using \a value as its
               argument.
  */
  inline bool inDomain(
         T const& value)
  {
    return _predicate(value);
  }
};
