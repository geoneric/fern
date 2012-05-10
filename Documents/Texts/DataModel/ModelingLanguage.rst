Modeling language
=================

Selecting attributes
--------------------
The proposed data model has implications on the modeling language that can be used to define an environmental model. Because of the hierarchical nature of the data model, it is easy to get confused as to what attribute an identifier refers to. For example, a forest feature may have a biomass attribute, but this attribute's actual values may be attached to the leaves three hierarchical levels down in the feature-attribute hierarchy. Or the biomass attribute that is available at the forest level is an aggregate which may or may not be what the modeler wants to use in his model's expressions.

One way to deal with this is to use `.property` notation on a feature, where `property` can be the name of a feature one level down in the hierarchy, or the name of an attribute at the current level.

.. code-block:: python

   forest = ...
   forestBiomass = forest.biomass
   treeBiomass = forest.tree.biomass
   leaveBiomass = forest.tree.leave.biomass

Selecting feature-items
-----------------------
In a lot of models it is necessary to be able to select a subset of the feature-items in a feature, based on some criterium. There are various ways to do that.

Select using a function:

.. code-block:: python

   children = select(persons, persons.age < 16)

This could probably work.

Select using an if statement:

.. code-block:: python

   if persons.age < 16:
     # Use persons, which in this block only represents the children.

This seems conceptually wrong. Lot of implicit stuff going on.

Using fancy indexing:

.. code-block:: python

   children = persons[age < 16]

The condition is evaluated in the context of the feature.

Fancy indexing is used in Numpy, Xslt, R, .... Pretty cool! Not sure if this can handle all cases.

Selecting feature-items returns a new feature instance containing references to the feature-items in the source feature instance. If the source feature instance goes out of scope, or is explicitly deleted, the feature instance containing the subset is still usable. For example, the next snippet in effect removes all children from the persons feature:

.. code-block:: python

   # Overwrite the original reference by a new one, containing a subset of
   # the feature-items.
   persons = persons[age >= 16]


