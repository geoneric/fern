#ifndef INCLUDED_RANALLY_ATTRIBUTE
#define INCLUDED_RANALLY_ATTRIBUTE

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>



namespace ranally {

class Feature;

//! Class for attribute instances.
/*!
  An attribute is either:

  * An uncertain spatio-temporal description of the attribute's variation in
    values, or a generalization thereof (information about the uncertainty,
    spatial variation, and/or temporal variation is missing). Spatial variation
    can be described in 1D, 2D and 3D. This is simply called the attribute's
    value, even though the values may well take gigabytes of storage space.
  * An uncertain spatio-temporal description of the attribute's domain (or
    a generalization thereof), with an attribute attached. This is what makes
    the definition recursive.

  Can we assume that attribute value variability is always continuous?
  - A raster value is always a spatial description of a continuous attribute?
  - There is no such thing as a nominal raster? Boolean, nominal and ordinal
    attribute values should be modeled using geometries with scalar values,
    like polygons. The modeling environment should be able to combine discrete
    and continuous attributes.

  Say we want to model a forest's biomass with an attribute, and say we
  have information about the spatial variation of biomass per leave(!). This
  can be modeled like this:

  * A forest contains a number of trees, so the forest_biomass attribute
    contains a multipoint geometry, and a tree_biomass attribute.
  * A tree contains a number of leaves, so the tree_biomass attribute contains
    a multipolygon geometry, containing a leave_biomass attribute.
  * A leave has a spatial (or spatio-temporal, or uncertain spatio-temporal)
    description of the actual variation in biomass values. This is where the
    recursion stops. We have reached the actual values.

  All kinds of attributes can be modelled like this:

  * Stream networks per continent.
  * Elevation per planet.
  * Humans walking trough a park.
  * Etc, etc, etc.

  Modeling attributes like this generalizes both traditional raster and feature
  data models in one unifying data model. Rasters are considered values in
  this model. They are one of the end points of the recursion definition,
  like scalars. Traditional features are defined using the attribute's
  geometry and a scalar attribute value.

  One way to look at this is that the attribute's spatio-temporal geometry
  positions the attribute's values in space and time.

  \sa        .
  \todo      Because of the risk of trying to boil the ocean, we forget about
             uncertainty in the attribute's spatio-temporal geometry, for now.
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

  //! Feature containing geometry and attribute.
  boost::scoped_ptr<Feature> _feature;

};

} // namespace ranally

#endif
