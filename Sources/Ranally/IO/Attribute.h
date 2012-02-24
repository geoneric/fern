#ifndef INCLUDED_RANALLY_ATTRIBUTE
#define INCLUDED_RANALLY_ATTRIBUTE

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>



namespace ranally {

class Domain;
class Feature;
class Value;

//! Class for Attribute instances.
/*!
  \sa        .
  \todo      Because of the risk of trying to boil the ocean, we forget about
             uncertainty in the attribute's spatio-temporal domain, for now.
*/
class Attribute:
  private boost::noncopyable
{

  friend class AttributeTest;

public:

  virtual          ~Attribute          ();

protected:

                   Attribute           ();

private:

  // TODO Is there a data structure that asserts that only one of the
  //      values for _value and _feature is set? _valueOrFeature
  //      Or do we allow values at all levels in the hierarchy (aggregates,
  //      pyramids, ...)?

  boost::scoped_ptr<Domain> _domain;

  //! Feature containing domain and attribute.
  boost::scoped_ptr<Feature> _feature;

  boost::scoped_ptr<Value> _value;

};

} // namespace ranally

#endif
