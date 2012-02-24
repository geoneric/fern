Modeling language
=================
The proposed data model has implications on the modeling language that can be used to define an environmental model. Because of the hierarchical nature of the data model, it is easy to get confused as to what attribute an identifier refers to. For example, a forest feature may have a biomass attribute, but this attribute's actual values may be attached to the leaves three hierarchical levels down in the feature-attribute hierarchy. Or the biomass attribute that is available at the forest level is an aggregate which may or may not be what the modeler wants to use in his model's expressions.

One way to deal with this is to use `.property` notation on a feature, where `property` can be the name of a feature one level down in the hierarchy, or the name of an attribute at the current level.

.. code-block:: python

   forest = ...
   forestBiomass = forest.biomass
   treeBiomass = forest.tree.biomass
   leaveBiomass = forest.tree.leave.biomass


