Glossary
========

..
   TODO period, episode

.. glossary::
   :sorted:

   attribute
      The collection of :term:`values <value>` that describe a certain (environmental) phenomenon, like height, soil class, speed, gravity, distance to a road. All values used in environmental modelling are attribute values. Every attribute is coupled to a :term:`feature`.

   domain
      Defines the locations in space and time for a :term:`feature's <feature>` :term:`attribute` values. The domain organizes the values in the spatio-temporal attribute space. Note that a :term:`raster` definition is not part of the domain specification. The domain is described in terms of the spatio-temporal feature coordinates.

      spatial:

      .. hlist::

         * points
         * multi-points
         * lines
         * multi-lines
         * polygons
         * multi-polygons

      temporal:

      .. hlist::

         * moments
         * periods / eras

   feature
      A combination of a spatio-temporal :term:`domain` with one or more :term:`attributes <attribute>`.

   feature-item
      Part of a feature that cannot be broken down into smaller parts. For example, in case of a point feature, the feature-items are the individual points. In case of a polygon feature, the feature-items are the individual polygons. Every feature-item has zero or more :term:`values <value>` associated with it.

   matrix
   vector
      An ordered collection of attribute values that collectively fill the spatio-temporal space defined by the associated :term:`feature-item`'s :term:`domain`. The values are separated from each other by a constant spatio-temporal distance. For example, a feature-item could contain a bounding box and a 2D matrix with height values. This is comparable to a classic DEM :term:`raster`. Or it could contain an era definition and a vector with discharge values. This is comparable tot a classic timeseries.

   raster
      A concept from the time life was easy, beer was cheap, and the wheather was always nice. Forget about it, it is of no use anymore, except when exchanging data with other software.

   sub-feature
     A larger scalar feature that is part of the definition of a smaller scale feature. For example, the earth feature may have continent sub-features, which may have country sub-features, etc.

   value
   attribute value
     A property of an attribute, often representing a scalar quantity or a nominal id. Examples are a specific soil class, or the speed of a car. An attribute value is represented by a single value or a :term:`matrix` of values.

