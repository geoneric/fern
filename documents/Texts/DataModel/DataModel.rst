Data model
==========
Below is a very high level graph of the data model as described in this document. It shows a feature at the top. This is a real world object or agent, that has attributes defined over a domain. The domain contains a definition of the spatio-temporal coordinates where the feature's attributes are located. An attribute is a property of a feature, like its color or height. Each attribute has a domain (the same one as the one the enclosing feature has), and a value and/or a larger scale sub-feature. The data model is recursive. Small scale features may be defined by larger scale features. For example, the `earth` feature may have a `soil_class` attribute attached to the `national_park` sub-feature.

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
       color="grey80",
       fontcolor="grey80"
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
     feature1 -> feature2 [label="*", color="grey60", fontcolor="grey60"];
     attribute1 -> domain1 [label="1"];
     attribute1 -> value1 [label="1"];

     feature2 -> domain2 [label="1", color="grey60", fontcolor="grey60"];
     feature2 -> attribute2 [label="*", color="grey60", fontcolor="grey60"];
     feature2 -> feature3 [label="*", color="grey80", fontcolor="grey80"];
     attribute2 -> domain2 [label="1", color="grey60", fontcolor="grey60"];
     attribute2 -> value2 [label="1", color="grey60", fontcolor="grey60"];
  }

What folows is a description of each of the classes of information that are part of the data model.

Feature
-------
A feature is a combination of a domain with zero or more associated attributes, and zero or more sub-features. For example, a deer feature can be modelled using a mobile point feature that contains the current location, and may have attributes like day_of_birth and weight, and may have sub-features like the minimum bounding rectangle of the first degree family members.

Another example of a feature with a sub-feature is a police car mobile point feature that also has a polygon sub-feature that contains the area that can be reached within 10 minutes of driving time. As the car moves, this area changes based on the surrounding road network.

In space, a feature is some spatial entity that has a position in space, and associated attributes. Examples of such features are houses, roads, cities, rivers, boats, planes, continents, etc.

In time, a feature is a collection of moments in time for which an attribute value is available. This can be one or more periods in time, or one or more moments in time. Examples of temporal features are average temperature per geological era, number of animal species going extinct per mass extinction event, number of plant species per interglacial, gross income per interbellum, tax rate per government period, etc.

Given that all attributes are spatio-temporal attributes, all features have a spatio-temporal domain, which means that all features are spatio-temporal features.

Examples of spatio-temporal features are the spatial distribution of a plant species in time, speed of cars driving on a highway, etc.

A feature is a phenomenon whose attributes play a role in the environmental model.

In traditional raster models, features are implicitly present in the model description. Often the feature being modelled is (a part of) the earth, like a continent, or an administrative area. It is the feature's attribute values that are modelled in raster models, and most of the times, these are all attributes of the same feature.

In multi-agent models, features are first class citizens. In fact, agents can be considered to be a specialization (without the recursion) of features as described in this document. This is, ofcourse, not without a reason. The envisioned modelling environment must be able to handle both traditional rasters and features.

In feature models, like traditional polygon overlay models, the features are the points, lines and polygons. Such features can also be considered to be a specialization (without the recursion and with one attribute containing one value per feature) of features as described in this document.

All kinds of features can be modelled like this:

* Stream networks per continent.
* Elevation per planet.
* Humans walking trough a park.
* Country per continent.
* Province per country.
* Etc, etc, etc.

A feature has exactly one domain, so it is not possible to model humans by points and volumes in one and the same feature, for example.

[ But this can be modelled using sub-features? ]

A feature has zero or more attributes. All these attributes have values for all spatio-temporal locations in the feature's domain, either directly, or indirectly using a larger scale sub-feature. Missing values are explicitly marked as such.

Domain
------
The domain organizes a feature's attributes in space and time. It defines where and when a feature's attribute values are defined. Without the information from the domain it is impossible to interpret and use a feature's attributes.

A domain contains information about the spatial and/or the temporal domain of the attributes. A spatial domain is very comparable to a traditional feature's geometry, like the coordinates of a multi-point feature.

In a domain, the feature-items are defined. Each of these items has a unique id which is used to lookup attribute values. For example, a country feature will have a domain consisting of multi-polygon feature-items. Each of these feature-items defines the borders of a single country, whose attributes can be looked up using the feature-item id.

The same domain is referenced by the enclosing feature, as well as each of the feature's attributes. This makes it easier to work with the attributes without a reference to the enclosing feature.

A spatial domain means attribute values vary with space. A temporal domain means attribute values vary with time. A mobile domain means the attribute's position changes with time. Any combination is possible, including a domain that is neither spatial, temporal and mobile. This means a constant value is stored that is constant through space and time.

[ Again, all feature attributes are uncertain, spatial and temporal. The domain can be spatial or not, meaning that the domain contains spatial coordinates over which the attribute's values change. For a non-spatial domain there is only one such coordinate, or even none. Maybe we should speak of spatial explicit and spatial non-explicit. Non-spatial is a silly and confusing word. Same for temporal. ]

[ I think a mobile domain doesn't need to be temporal, but maybe they do. I am thinking about a temporal constant value that does change position over time, like the color of a driving car. A mobile domain does need to be spatial. ]

The temporal coordinates with which the mobility is modelled are independent of the temporal coordinates with which the attribute value variation is modelled.

Attribute
---------
An attribute is a spatio-temporal description of an uncertain property of a feature.

An attribute contains an uncertain spatio-temporal description of the attribute's variation in values, or a generalization thereof (information about the uncertainty, spatial variation, and/or temporal variation is missing). Spatial variation can be described in 1D, 2D and 3D. This is simply called the attribute's value, even though the values may well take gigabytes of storage space.

Modeling attributes like this generalizes both traditional raster and feature data models in one unifying data model. Rasters are considered values in this model. Traditional features are defined using the attribute's geometry (stored in the domain) and a scalar attribute value.

Value
-----
A value consists of one or more values describing the variation over a feature's, possibly spatio-temporal, value domain.

Examples of values are:

* A single value per feature-item in the domain.
* A regular discretisized collection of values per item in the domain, like a raster in 2D space, or a regular timeseries in time.
* A probability distribution of a value per feature-item in the domain.
* A probability distribution of a regular discretisized collection of values per item in the domain.

A result of all this is that a raster's values, for example, are stored in the `Attribute`'s `Value`. The polygon describing the raster's extent is stored in the `Feature`'s `Domain`. This extent does not necessarely have to be a rectangle. For example, imagine a country feature with a national_park sub-feature, with a height attribute, whose values are stored in a matrix.
An example of a (spatio-)temporal attribute is a river feature with a tributary sub-feature, with a discharge attribute, which is measured at regular intervals, except during the winter when all the water is frozen. The begin and end date/times are stored in the `Domain` while the arrays of values are stored in the `Value`.

.. important::

   Discrete value changes are modeled using a Domain, not by a Value. Using a domain one can record the positions in space and/or time where/when an attribute's value changes.

Recursion
---------
From the graph above, it shows that Feature is defined by itself, so recursively. There are multiple reasons for this, like:
- Attributes of a small scale feature can be defined by larger scale features. This is useful if the same attribute values are used at multiple spatio-temporal scales. The obvious example where this is useful is in visualization, but it can also be done to guide the paralellization of the model run.
- A feature's attributes are tied to different domains. In the police car example mentioned above, some police car's attributes are tied to the police car's point feature (car id, driver id, etc), and some attributes are tied to the police car's service area (its area and the properties of the neighborhood covered by the area, for example).

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
     forrestFeature -> treeFeature;
     forrestAttribute -> forrestValue;

     treeFeature -> treeDomain;
     treeFeature -> treeAttribute;
     treeFeature -> leaveFeature;
     treeAttribute -> treeValue;

     leaveFeature -> leaveDomain;
     leaveFeature -> leaveAttribute;
     leaveAttribute -> leaveValue;
  }

Another example is some attribute that needs to be visualized at different spatial scales:

.. graphviz::

   digraph Elevation {
     ordering="out"

     feature1[
       label="feature: earth"
     ]
     feature1Domain[
       label="domain: earth"
     ]
     feature1Attribute[
       label="attribute: height"
     ]
     feature1Value[
       label="value: height at 1:1000000000"
     ]

     feature2[
       label="feature: earth"
     ]
     feature2Domain[
       label="domain: earth"
     ]
     feature2Attribute[
       label="attribute: height"
     ]
     feature2Value[
       label="value: height at 1:000000"
     ]

     feature3[
       label="feature: earth"
     ]
     feature3Domain[
       label="domain: earth"
     ]
     feature3Attribute[
       label="attribute: height"
     ]
     feature3Value[
       label="value: height at 1:000"
     ]

     feature1 -> feature1Domain;
     feature1 -> feature1Attribute;
     feature1 -> feature2;
     feature1Attribute -> feature1Value;

     feature2 -> feature2Domain;
     feature2 -> feature2Attribute;
     feature2 -> feature3;
     feature2Attribute -> feature2Value;

     feature3 -> feature3Domain;
     feature3 -> feature3Attribute;
     feature3Attribute -> feature3Value;
  }

Misc
----
* Features are allowed to overlap, for example when 2D trees in a forrest are represented as (horizontal) polygons instead of points.

