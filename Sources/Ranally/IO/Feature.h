#ifndef INCLUDED_RANALLY_FEATURE
#define INCLUDED_RANALLY_FEATURE

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>



namespace ranally {

class Attribute;
class Domain;

//! Class for Feature instances combining Domain with an Attribute.
/*!
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
