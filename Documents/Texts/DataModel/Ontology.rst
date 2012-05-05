Model ontology
==============
This section uses the term attribute to mean the collection of values that describe a certain (environmental) phenomenon, like height, soil class, speed, gravity, distance to a road. All values used in environmental modelling are attribute values.

A feature is a combination of a spatio-temporal domain with one or more attributes. A feature's domain defines the locations in space and time for the feature's attribute values. All attribute values are coupled to some feature.

There is always a correspondence between an attribute and some real-world phenomenon. Empirical values are also treated as attributes. Empirical values are tied to the feature earth, or space in general, for example.

Modellers are interested in attributes, not technicalities. Modellers model concentrations, income, volumes, speeds, directions, etc. The fact that there are things like agents, points, rasters, vectors, single precision floating points, unsigned integers, etc, are an implementation detail that may or may not be relevant to the modeller. It may not matter to him as long as requirements in execution speed, data size, accuracy, etc are met. Since environmental attributes matter more to humans than implementation details, working with attributes results in a cleaner, more correct, design, than working with concepts like rasters, features, time steps, and Monte Carlo sample numbers.

In the next section, we describe ways to categorize attributes. This will improve our understanding of what attributes are. After that, we move on to the functions that are used to manipulate attributes in a model. Finally, we have something to say about the modelling environment.

Attribute type categorizations
------------------------------
The categorizations mentioned below divide the attribute types in various partitions. By combining the categories, it must be possible to define all attribute types that are elevant in environmental modelling. If not, then we are missing a categorization.

This section does not mention implementation details like raster and vector because these notions are only relevant in software. In the end these concepts matter and the modeller may have to know about them, but they are not relevant for describing the attribute type categorizations.

This section does not deal with uncertainty, whether or not attributes are spatial, and whether or not attributes are temporal. The rule is:

.. important::

   All attributes are uncertain spatio-temporal attributes.

The fact that information about the spatio-temporal variation of an attribute's values is missing means that the attribute's values don't vary in space and/or time. Unmeasured values (unavailable, no-data values) are explicitly marked as such.

In case information about the uncertainty in an attribute's values is missing, this most probably means the information is not available. In this case, for practical reasons, one must assume that the attribute's values are known.

Continuous versus discrete variation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When we consider the type of variation of attribute values, we can categorize the attributes in continuous and discrete types.

Spatially continous
"""""""""""""""""""
* These attributes are defined along at least one spatial dimension. Along each of these dimensions, the attribute is either missing or known. In principle, the attribute could be known at all locations.
* 1D: height along a profile, land-use along a profile
* 2D: height, land-use (by x, y), wind direction
* 3D: air pressure, wind direction

Spatially discrete
""""""""""""""""""
* These attributes are tied to entities that have a finite extent in space.
* Points: concentration (by gauging station)
* Lines: length of traffic jam (by road segment), concentration (by river segment), business (by railway segment)
* Polygons: average income (by municipality), land-use (by parcel)

Temporal continuous
"""""""""""""""""""
* These attributes are defined along the time dimension. The attribute is either missing or known. In principle, the attribute could be known at all moments in time.
* temperature

Temporal discrete
"""""""""""""""""
* Temporal discrete attribute. These attributes are tied to periods that have a finite extent in time.
* speed and direction during mass movement

And now to something completely different...
""""""""""""""""""""""""""""""""""""""""""""
Given the above it seems that there are at least two types of attributes: those with discretely varying values and those with continuously varying values. But even a continuously varying attribute is always tied to a discrete entity. A DEM, for example, contains continuously varying attribute value, but thos values are linked to the area for which these values are defined. This may be research area, a country, a continent, the earth, a set of planets, all planets, etc.

So, if we agree that all attribute values are eventually linked to a descrete entity (which we call a feature-item later on), then the thing that is different between the continuously and discretely varying attribute value is the fact that a discretely varying attribute has a single value per feature-item, and a continuously varying attribute has a collection of values (vector, 2D matrix, 3D matrix) per feature-item. More about this in the Data Model section.

Mobile versus stationary
^^^^^^^^^^^^^^^^^^^^^^^^
Another way to look at attributes is from the standpoint of mobility. Obviously, some attributes travel through space in time and other don't. (We consider temporal mobility once that is an option in real-life.)

Spatial mobile
""""""""""""""
* These attributes (potentially) change their spatial location.
* attribute by river, attribute by road, attribute by individu

Spatial stationary
""""""""""""""""""
* These attributes don't change spatial location, ever.
* attribute by house, attribute by road segment, attribute by railway segment

Spatial mobility depends on the time scale. Spatial object which are stationary on short time scales, may be mobile on larger time scales.

Generalizing:

.. important::

   All attributes are spatially mobile.

But some of them just don't move within the modelled time period.

Functions
---------
Attributes versus functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
An attribute contains values that represent the state of the attribute. Functions calculate attribute values, based on the state of one or more other attributes. In a way, functions are very much like attributes. They just need to perform some calculation before being able to provide the new attribute's state values. Or, the other way around, reading existing attribute values is like executing some identity function that simply returns the attribute's current state values, unchanged.

.. important::

   Attributes are very similar to functions. Both are attribute value providers.

Functions versus models
^^^^^^^^^^^^^^^^^^^^^^^
A function accepts input attributes and calculates the state values of output attributes. Models (including user defined functions) do the same thing. The difference between the two is a matter of scale / hierarchy. Whether or not a function or a model uses iteration to calculate the result is of no relevance and can be considered an internal detail.

.. important::

   Functions are very similar to models. Both are attribute value providers.

Combining the rules above:

.. important::

   Attributes, functions and models are all attribute providers. They differ wrt the amount of effort that needs to be done to generate the output attribute's state values. Apart from that, the difference is one of scale/complexity/hierarchy.

Modelling environment
---------------------
All attributes are passive, in the sense that they are just values and there is no behavioural logic coupled to the attribute that is able to change the attribute values. All attributes are input to operations that return newly calculated attribute values. This is common usage in map algebra implementations, but agent based models tend to use a more object oriented type of approach that couples behaviour with attribute values. The same functionality can be achieved by defining functions that recieve attributes that are coupled to spatially discrete objects, for example. This results in a general algebraic modelling language where operations accept all kinds of attributes, creating new attributes.

