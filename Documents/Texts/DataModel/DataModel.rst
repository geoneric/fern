Data model
==========
Below is a very high level graph of the data model as described in this document. It shows a feature at the top. This is a real world object or agent, that has attributes defined over a domain. The domain contains the spatio-temporal coordinates where the feature's attributes are located. An attribute is a property of a feature, like its color or height. Each attribute has a domain (the same one as the one the enclosing feature has), and a value and/or a larger scale sub-feature. The data model is recursive. Small ѕcall features may be defined by larger scale features. For example, the earth feature may have a soil class attribute attached to the national_park sub-feature.

.. graphviz::

   digraph DataModel {
     feature1[
       label="Feature"
     ]
     feature2[
       label="Feature",
       color="grey60",
       fontcolor="grey60"
     ]
     feature3[
       label="Feature"
       color="grey90",
       fontcolor="grey90"
     ]
     domain1[
       label="Domain"
     ]
     domain2[
       label="Domain"
       color="grey60",
       fontcolor="grey60"
     ]
     attribute1[
       label="Attribute"
     ]
     attribute2[
       label="Attribute"
       color="grey60",
       fontcolor="grey60"
     ]
     value1[
       label="Value"
     ]
     value2[
       label="Value"
       color="grey60",
       fontcolor="grey60"
     ]

     feature1 -> domain1 [label="1"];
     feature1 -> attribute1 [label="*"];
     attribute1 -> domain1 [label="1"];
     attribute1 -> feature2 [label="?", color="grey60", fontcolor="grey60"];
     attribute1 -> value1 [label="? (if feature) | + (if !feature)"];

     feature2 -> domain2 [label="1", color="grey60", fontcolor="grey60"];
     feature2 -> attribute2 [label="*", color="grey60", fontcolor="grey60"];
     attribute2 -> domain2 [label="1", color="grey60", fontcolor="grey60"];
     attribute2 -> feature3 [label="?", color="grey90", fontcolor="grey90"];
     attribute2 -> value2 [label="? (if feature) | + (if !feature)", color="grey60", fontcolor="grey60"];
  }

What folows is a description of each of the classes of information that are part of the data model.

Feature
-------
A feature is a combination of a domain with zero or more associated attributes.

In space, a feature is some spatial entity that has a position in space, and associated attributes. Examples of such features are houses, roads, cities, rivers, boats, planes, continents, etc.

In time, a feature is a collection of moments in time for which an attribute value is available. Examples of temporal features are temperature per geological era, number of plant species per interglacial, gross income per interbellum, tax rate per government period, etc.

Examples of spatio-temporal features are the spatial distribution of a plant species in time, speed of cars driving on a highway, etc.

A feature is a phenomenon whose attribute plays a role in the environment model.

In traditional raster models, features are implicitly present in the model description. Often the feature being modelled is (a part of) the earth, like a continent, or an administrative area. It is the feature's attribute values that are modelled in raster models, and most of the times, these are all attributes of the same feature.

In multi-agent models, features are first class citizens. In fact, agents can be considered to be a specialization (without the recursion) of features as described in this document.

In feature models, like traditional polygon overlay models, the features are the points, lines and polygons. Such features can also be considered to be a specialization (without the recursion and with one attribute containing one value per feature) of features as described in this document.

All kinds of features can be modelled like this:

* Stream networks per continent.
* Elevation per planet.
* Humans walking trough a park.
* Country per continent.
* Province per country.
* Etc, etc, etc.

A feature has exactly one domain. We may want to addopt the convention that when the domain is absent, the attribute should be considered present in all spatio-temporal locations. In that case, the attribute's value can only be a single value.

A feature has zero or more attributes. All these attributes have values for all spatio-temporal locations in the feature's domain, either directly, or indirectly using a larger scale sub-feature.

Domain
------
The domain organizes a feature's attributes in space and time. It defines where and when a feature's attribute values are defined. Without the information from the domain it is impossible to interpret and use a feature's attributes.

A domain can contain information about the spatial and/or the temporal domain of the attributes. A spatial domain is very comparable to a traditional feature's geometry, like the coordinates of a multi-point feature.

In a domain, the feature-items are defined. Each of these items has a unique id which is used to lookup attribute values. For example, a country feature will have a domain consisting of multi-polygon feature-items. Each of these feature-items defines the borders of a single country, whose attributes can be looked up using the feature-item id.

The same domain is referenced by the enclosing feature, as well as each of the feature's attributes. This makes it easier to work with the attributes without a reference to the enclosing feature.

Attribute
---------
An attribute is a spatio-temporal description of an uncertain property of a feature.

An attribute is either, or both:

* An uncertain spatio-temporal description of the attribute's variation in values, or a generalization thereof (information about the uncertainty, spatial variation, and/or temporal variation is missing). Spatial variation can be described in 1D, 2D and 3D. This is simply called the attribute's value, even though the values may well take gigabytes of storage space.
* A larger scale feature containing the same attribute. This is what makes the definition recursive.

..
   Modeling attributes like this generalizes both traditional raster and feature data models in one unifying data model. Rasters are considered values in this model. They are one of the end points of the recursive definition, like scalars. Traditional features are defined using the attribute's geometry and a scalar attribute value.

Value
-----
A value consists of one or more values describing the variation over a feature's, possibly spatio-temporal, value domain.

Examples of values are:

* A single value per feature-item in the domain.
* A probability distribution of a value per feature-item in the domain.
* A regular discretisized collection of values per item in the domain, like a raster in 2D space, or a regular timeseries in time.
* A probability distribution of a regular discretisized collection of values per item in the domain.

A result of all this is that a raster's values, for example, are stored in the `Attribute`'s `Value`. The polygon describing the raster's extent is stored in the `Feature`'s `Domain`. This extent does not necessarely be a rectangle. For example, imagine a country feature with a national_park sub-feature, with a height attribute, whose values are stored in a raster.
An example of a (spatio-)temporal attribute is a river feature with a tributary sub-feature, with a discharge attribute, which is measured at regular intervals, except during the winter when all the water is frozen. The begin and end date/times are stored in the `Domain` while the arrays of values are stored in the `Value`.

Discrete value changes are modeled using a Domain, not by a value. Using a domain one can record the positions in space and/or time that an attribute's value changes. TODO No.

TODO: A value can also be a distribution of values, in case there is an error associated with the value.

Recursion
---------
From the graph above, it shows that feature is defined by itself, recursively. Attributes of a small scale feature can be defined by a larger scale feature, if more detailed information is available. Or, a large scale feature, can be aggregated to a smaller scale feature.

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

The main reason for the data model to be recursive is that it allows us to treat traditional rasters, features and agents in a uniform way. Rasters end up in the attribute's values. Traditional features and agents are a specialization of the features described here.

Misc
====
* Features are allowed to overlap, for example when 2D trees in a forrest are represented as (horizontal) polygons instead of points.

