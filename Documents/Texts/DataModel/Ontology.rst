Model ontology
==============
This section uses the term attribute to mean the collection of values that describe a certain (environmental) phenomenon. For most attributes in environmental models, there is a correspondence between the attribute and some real-world phenomenon, but this is not a requirement. Empirical values are also treated as attributes.

Modellers are interested in attributes, not technicalities. Modellers model concentrations, income, volumes, speeds, directions, etc. The fact that there are things like agents, points, rasters, vectors, single precision floating points, unsigned integers, etc, are an implementation detail that may or may not be relevant to the modeller. It may not matter to him as long as requirements in execution speed, data size, accuracy, etc are met. Since environmental attributes matter more to humans than implementation details, working with attributes results in a cleaner, more correct, design, than working with concepts like rasters and features.

Since the purpose of modelling is to translate input attributes to output attributes, the focus is on attributes, not the model. The model is not the goal, the model results are. (Good model results (predictions) may allow the modeller to learn something about the modelled processes, but model results rarely are *that* good ;-) )

That's why I approach data types starting with attribute types. The next sections describe what the differences are between the various kinds of attributes that are used in model studies.

Attribute type categorizations
------------------------------
The categorizations mentioned below divide the attribute types in various partitions. By combining the categories, it must be possible to define all attribute types that are elevant in environmental modelling. If not, then we are missing a categorization.

This section does not mention implementation details like raster and vector because these notions are only relevant in software. In the end these concepts matter and the modeller may have to know about them, but they are not relevant for describing the attribute type categorizations.

This section does not deal with uncertainty, whether or not attributes are spatial, and whether or not attributes are temporal. The rule is:

*All attributes are uncertain spatio-temporal attributes.*

The fact that information about the spatio-temporal variation of an attribute's values is missing means that the attribute's values don't vary in space and/or time. Unmeasured values (unavailable, no-data values) are explicitly marked as such.

In case information about the uncertainty in an attribute's values is missing, this most probably means the information is not available. In this case, for practical reasons, one must assume that the attribute's values are known.

Continuous versus discrete variation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Space

  * Spatially continous

    * These attributes are defined along at least one spatial dimension. Along each of these dimensions, the attribute is either missing or known. In principle, the attribute could be known at all locations.
    * 1D: ...
    * 2D: height, land-use (by x, y), wind direction
    * 3D: air pressure, wind direction

  * Spatially discrete

    * These attributes are tied to features that have a finite extent in space.
    * Points: concentration (by gauging station)
    * Lines: length of traffic jam (by road segment), concentration (by river segment), business (by railway segment)
    * Polygons: average income (by municipality), land-use (by parcel)

  * Maybe all attributes are spatially discrete? A spatial field attribute, like height for example, is tied to (a piece of) the earth.

* Time

  * Temporal continuous

    * These attributes are defined along the time dimension. The attribute is either missing or known. In principle, the attribute could be known at all moments in time.
    * temperature

  * Temporal discrete

    * Temporal discrete attribute. These attributes are tied to periods that have a finite extent in time.
    * speed and direction during mass movement

*Continuous attributes (spatial and/or temporal) are not coupled to specifіc features (spatial and/or temporal). These attributes have values / are defined within a certain spatial and/or temporal extent.*

*Discrete attributes (spatial and/or temporal) are coupled to specific features (spatial and/or temporal. These attributes have values / are defined for the spatial and/or temporal extent of these features only.*

Mobile versus stationary
~~~~~~~~~~~~~~~~~~~~~~~~
* Space

  * Spatial mobile

    * These attributes (potentially) change their spatial location.
    * attribute by river, attribute by road, attribute by individu

  * Spatial stationary

    * These attributes don't change spatial location, ever.
    * attribute by house, attribute by road segment, attribute by railway segment

Spatial mobility may depend on the time scale. Spatial object which are stationary on short time scales, may be mobile on larger time scales.

Whether or not an attribute is mobile or ѕtationary, is only relevant for discrete attributes. The possible type of mobility (spatial and/or temporal) of a specific attribute depends on the nature of the 'discreteness' of the attribute (spatial and/or temporal discrete). A spatially discrete attribute can, in theory, be spatially mobile. A temporally discrete attribute can, in theory, be temporally mobile.

For example, the attribute `speedOfCar` is spatially discrete (each car is at a specific spatial location) but temporal continuous (each car's speed changes continous in time, at all times does the car have a speed). In this case, the car is spatially mobile, but not temporal.

Given the coordinates of the spatial and/or temporal objects to which a discrete attribute's values are coupled, mobility means that these coordinates change in space and/or time.

Temporal mobility means that the date/times or time periods for which an attribute is defined changes. I don't think this ever happens(?).

Attributes versus functions
---------------------------
An attribute contains values that represent the state of the attribute. Functions calculate attribute values, based on the state of one or more other attributes. In a way, functions are very much like attributes. They just need to perform some calculation before being able to provide the new attribute's state values. Or, the other way around, reading existing attribute values is like executing some identity function that simply returns the attribute's current state values.

*Attributes are very similar to functions. Both are attribute value providers.*

Functions versus models
-----------------------
A functions accepts input attributes and calculates the state values of output attributes. Models (including model components) do the same thing. The difference between the two is a matter of scale / hierarchy. Whether or not a function or model uses iteration to calculate the result іs of no relevance and can be considered an internal detail.

*Functions are very similar to models. Both are attribute value providers.*

Combining the rules above:

*Attributes, functions and models are all attribute providers. They differ wrt the amount of effort that needs to be done to generate the output attribute's state values. Apart from that, the difference is one of scale/complexity/hierarchy.*

Modelling environment
---------------------
All attributes are passive, in the sense that they are just values and there is no behavioural logic coupled to the attribute that is able to change the attribute values. All attributes are input to operations that return newly calculated attribute values. This is common usage in map algebra implementations, but agent based models tend to use a more object oriented type of approach that couples behaviour with attribute values. The same functionality can be achieved by defining functions that recieve attributes that are coupled to spatially discrete objects, for example. This results in a general algebraic modelling language where operations accept all kinds of attributes, creating new attributes.

TODO Can map algebra and agent based modelling be merged?

Additional ideas
----------------
* There are two entities: features and attributes.
* Features:

  * AKA object, agent.
  * Features are spatial and mobile.
  * Set of spatial (2D, 3D) coordinates that (potentially) vary in time.
  * Feature types are: (multi) points, (multi) lines, (multi) polygons,
    (multi) volumes.

* Attributes:

  * Attributes are uncertain spatio-temporal numbers.
  * Not every attribute has known spatial and/or temporal varying values, or has known uncertainty properties.

* Every attribute is attached to a feature.

  * Since features themselves are spatial and mobile, this means that uncertain spatio-temporal attributes are tied to a spatial mobile feature.
  * Examples:
    * For every tree in a forest we may want to record the spatial (2D or 3D) variation in biomass.

* TODO Is this a fully recursive definition, or do we want to stop at one level of feature + attribute? In the recursive case, you could model the leaves of a tree:

  * trees: multi point features with biomass attribute
  * tree_biomass: multi polygon feature with biomass attribute
  * leave_biomass: polygon with biomass attribute as a field with values.
  * In the recursive case, an attribute iѕ defined as having

    * a (field of) values (recursion stops) or
    * a feature + attribute combination

