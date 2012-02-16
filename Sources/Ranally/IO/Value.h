#ifndef INCLUDED_RANALLY_VALUE
#define INCLUDED_RANALLY_VALUE

#include <boost/noncopyable.hpp>



namespace ranally {

//! A value is a property of an attribute's domain.
/*!
  A value can consist of multiple values describing the continuous variation
  over a feature's, possibly uncertain spatio-temporal, value domain.

  Examples of values are:
  * A single value.
  * A raster of values representing a continuous field.
  * A timeseries of values representing a continuous changing value.

  Discrete value changes are modeled using a Domain, not by a value. Using a
  domain one can record the positions in space and/or time that an attribute's
  value changes.

  \sa        .
  \todo      How to approach current boolean/nominal/ordinal rasters? Really
             model them using polygons and values? Will that work in math
             models?
*/
class Value:
  private boost::noncopyable
{

  friend class ValueTest;

public:

                   Value               ();

  /* virtual */    ~Value              ();

protected:

private:

};

} // namespace ranally

#endif
