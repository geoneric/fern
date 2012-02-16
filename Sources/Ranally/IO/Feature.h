#ifndef INCLUDED_RANALLY_FEATURE
#define INCLUDED_RANALLY_FEATURE

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>



namespace ranally {

class Attribute;
class Domain;

//! Class for Feature instances combining Domain with an Attribute.
/*!
  A feature is a combination of a domain with an associated attribute. This
  can be, for example, a multi point spatial domain with a tree_biomass
  attribute. The layered attribute contains attribute values and/or a
  nested feature (for example, to describe the biomass of the individual
  leaves using a multi-polygon feature).

  \sa        .
*/
class Feature:
  private boost::noncopyable
{

  friend class FeatureTest;

public:

                   Feature             ();

  /* virtual */    ~Feature            ();

protected:

private:

  boost::scoped_ptr<Domain> _domain;

  boost::scoped_ptr<Attribute> _attribute;

};

} // namespace ranally

#endif
