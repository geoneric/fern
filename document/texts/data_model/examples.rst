Examples
========
In this section, the ideas from the previous sections are explained in the context of popular use-cases.

Data sets
---------

DEM of an area
~~~~~~~~~~~~~~


Land-use from satelite imagery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Spatio-temporal agents
~~~~~~~~~~~~~~~~~~~~~~


Models
------

Wolve sheep predation
~~~~~~~~~~~~~~~~~~~~~
This model is inspired by the `NetLogo Wolf Sheep Predation model <http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation>`_.

.. literalinclude:: wolves_and_sheep.ran
   :language: python

Heatbugs
~~~~~~~~
This model is inspired by the `NetLogo Heatbugs model <http://ccl.northwestern.edu/netlogo/models/Heatbugs>`_.

.. literalinclude:: heatbugs.ran
   :language: python

Connecting agents
~~~~~~~~~~~~~~~~~
In multi agent modelling agents often want to keep track of a group of other agents that fulfill a certain requirement. Let's say agents need to know about those agents that have at least 5 neighbors within a certain distance.

.. code-block:: python

   # Read initial house features (points).
   houses = ...
   radius = 100

   # Select those houses that have at least 5 neighbors within a certain
   # distance. This is just a subset of the houses point feature.
   # This makes use of a predicate to do the selection. The predicate is
   # evaluated per feature-item (point in this case).
   # nearby -> Select feature-items that are located within a certain radius
   #   of the current feature-item.
   # count -> Count the number of feature-items passed in.
   # >= -> Select current feature-item if boolean operation evaluates to true.
   clustered_houses = houses[count(nearby(radius)) >= 5]

   # We now have those houses that are *in the center* of a cluster. Let's
   # select all the houses that are part of the cluster, including the ones at
   # the border.
   clustered_houses = select_by_buffer(houses, buffer(clustered_houses, radius))

   # Group houses into clusters using the radius used before. This creates
   # multi-point features per cluster. These could be small communities.
   house_clusters = cluster_by_distance(clustered_houses, radius)

   # Determine distance between each house to the nearest cluster of houses.
   houses.distance_to_cluster = distance(houses, house_clusters)

   # ...
