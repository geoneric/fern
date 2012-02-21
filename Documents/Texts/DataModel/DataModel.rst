Data model
==========
Below is a very high level graph of the data model as described in this document. What folows is a description of each of the classes of information that are part of the data model.

.. graphviz::

   digraph DataModel {
     feature1[
       label="Feature"
     ]
     feature2[
       label="Feature",
       color="grey40",
       fontcolor="grey40"
     ]
     feature3[
       label="Feature"
       color="grey80",
       fontcolor="grey80"
     ]
     domain1[
       label="Domain"
     ]
     domain2[
       label="Domain"
       color="grey40",
       fontcolor="grey40"
     ]
     attribute1[
       label="Attribute"
     ]
     attribute2[
       label="Attribute"
       color="grey40",
       fontcolor="grey40"
     ]
     value1[
       label="Value"
     ]
     value2[
       label="Value"
       color="grey40",
       fontcolor="grey40"
     ]

     feature1 -> domain1;
     feature1 -> attribute1;
     attribute1 -> feature2;
     attribute1 -> value1;

     feature2 -> domain2;
     feature2 -> attribute2;
     attribute2 -> feature3;
     attribute2 -> value2;
  }

Feature
-------
A feature is a combination of a domain with an associated attribute.

In space, a feature is some spatial entity that has a position in space and an associated value. Examples of such features are houses, roads, cities, rivers, boats, planes, etc.

In time, a feature is a collection of moments in time for which an attribute value is available. Examples of temporal features are temperature per geological era, number of plant species per interglacial, gross income per interbellum, tax rate per government period, etc.

Examples of spatio-temporal features are the spatial distribution of a certain plant in time, speed of cars driving on a certain highway, etc.

A feature is a phenomenon whose attribute plays a role in the environment model.

In traditional raster models, features are implicitly present in the model description. Often the feature being modelled is (a part of) the earth, like a continent, or a administrative area. It is the feature's attribute values that are modelled in raster models.

In multi-agent models, features are first class citizens. In fact, agents can be considered to be the same as features.

In feature models, like traditional polygon overlay models, the features are the points, lines and polygons.

Domain
------
The domain organizes a feature's attributes in space and time. It defines where and when a feature's attribute values are defined. Without the information from the domain it is impossible to use a feature's attributes.

A domain can contain information about the spatial and/or the temporal domain of the attributes. A spatial domain is very comparable to a traditional feature's geometry, like the coordinates of a multi-point feature.

Аttribute
---------
An attribute is either:

* An uncertain spatio-temporal description of the attribute's variation in
  values, or a generalization thereof (information about the uncertainty,
  spatial variation, and/or temporal variation is missing). Spatial variation
  can be described in 1D, 2D and 3D. This is simply called the attribute's
  value, even though the values may well take gigabytes of storage space.
* An uncertain spatio-temporal description of the attribute's domain (or
  a generalization thereof), with an attribute attached. This is what makes
  the definition recursive.

Can we assume that attribute value variability is always continuous? Consider land-use as computed from satelite imagery. Treating such values as a (multi-polygon) feature with associated value will not fly.

* A raster value is always a spatial description of a continuous attribute?
* There is no such thing as a nominal raster? Boolean, nominal and ordinal
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

Value
-----
A value can consist of multiple values describing the continuous variation
over a feature's, possibly uncertain spatio-temporal, value domain.

Examples of values are:

* A single value.
* A raster of values representing a continuous field.
* A timeseries of values representing a continuous changing value.

Discrete value changes are modeled using a Domain, not by a value. Using a
domain one can record the positions in space and/or time that an attribute's
value changes.

TODO: A value can also be a distribution of values, in case there is an error associated with the value.

Recursion
---------
From the graph above, it shows that Feature is defined by itself, recursively. A small scale feature can be defined by a larger scale feature, if more detailed information is available. Or, a large scale feature, can be aggregated to a smaller scale feature.

Take, for example, the biomass of a forrest. Given that biomass information is available per leave per tree, biomass of the forrest could be modelled using a forrest_biomass feature (see graph below).

.. graphviz::

   digraph ForrestBiomass {
     ordering="out"

     forrestFeature[
       label="feature: forrest"
     ]
     forrestDomain[
       label="domain: polygon per\nforrest-patch"
     ]
     forrestAttribute[
       label="attribute: biomass"
     ]
     forrestValue[
       label="value: biomass per\npatch"
     ]

     treeFeature[
       label="feature: tree"
     ]
     treeDomain[
       label="domain: point per\ntree"
     ]
     treeAttribute[
       label="attribute: biomass"
     ]
     treeValue[
       label="value: biomass per\ntree"
     ]

     leaveFeature[
       label="feature: leave"
     ]
     leaveDomain[
       label="domain: polygon per\nleave"
     ]
     leaveAttribute[
       label="attribute: biomass"
     ]
     leaveValue[
       label="value: biomass per\nleave"
     ]

     forrestFeature -> forrestDomain;
     forrestFeature -> forrestAttribute;
     forrestAttribute -> treeFeature;
     forrestAttribute -> forrestValue;

     treeFeature -> treeDomain;
     treeFeature -> treeAttribute;
     treeAttribute -> leaveFeature;
     treeAttribute -> treeValue;

     leaveFeature -> leaveDomain;
     leaveFeature -> leaveAttribute;
     leaveAttribute -> leaveValue;
  }

