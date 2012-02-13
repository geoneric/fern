#ifndef INCLUDED_RANALLY_ATTRIBUTE
#define INCLUDED_RANALLY_ATTRIBUTE

#include <boost/noncopyable.hpp>



namespace ranally {

//! Class for attribute instances.
/*!
  An attribute is a either:

  * An uncertain spatio-temporal description of the attribute's variation in
    values, or a generalization thereof (information about the uncertainty,
    spatial variation, and/or temporal variation is missing). Spatial variation
    can be described in 1D, 2D and 3D. This is simply called the attribute's
    value.
  * An uncertain spatio-temporal description of the attributes geometry (or
    a generalization thereof), with an attribute attached. This is what makes
    the definition recursive.

  Say we want to model a forest's biomass with an attribute, and say we
  have information about the spatial variation of biomass per leave(!). This
  can be modeled like this:

  * A forest contains a number of trees, so the forest_biomass attribute
    contains a multipoint feature geometry, and a tree_biomass attribute.
  * A tree contains a number of leaves, so the tree_biomass attribute contains
    a multipolygon feature geometry, containing a leave_biomass attribute.
  * A leave has a spatial (or spatio-temporal, or uncertain spatio-temporal)
    description of the actual variation in biomass values. This is where the
    recursion stops. We have reached the actual values.

  All kinds of attributes can be modelled like this:

  * Stream networks per continent.
  * Elevation per planet.
  * Humans walking trough a park.
  * Etc, etc, etc.

  Modeling attributes like this generalizes both traditional raster and feature
  data models. Rasters are considered values in this model. They are one of the
  end points of the recursion definition, like scalars. Traditional features
  are defined using the feature's geometry and a scalar attribute value.

  \sa        .
*/
class Attribute:
  private boost::noncopyable
{

  friend class AttributeTest;

public:

                   Attribute               ();

  /* virtual */    ~Attribute              ();

protected:

private:

};

} // namespace ranally

#endif
