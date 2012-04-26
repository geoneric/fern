Introduction
============

This document describes concepts that are common to feature-based and field-based modelling with uncertain spatio-temporal feature attributes. The ultimate goal is to be able to develop a modelling environment for uncertain attributes of spatio-temporal phenomena, whether they are descritized as features or as fields.

Separate modelling approaches
-----------------------------
Currently, the agent-based modelling and field-based modelling are two seperate worlds. Although in some agent-base modelling environments it is possible to use information from rasters, and although in field based modelling agents are sometimes represented by raster cells, conceptually these modelling worlds have very little in common. They lack a set of common concepts.

Lack of format for uncertain spatio-temporal attributes
-------------------------------------------------------
Information about the temporal domain and the uncertainty in attribute values is hardly ever stored along with the attribute's values. There is a lack of data set formats to store uncertain spatio-temporal attribute values.

No support for modelling techniques
-----------------------------------
Although modelling environments exist that allow users to model spatial and spatio-temporal phenomena, when it comes to handling uncertainty with a Monte Carlo analysis, or assimilation of known attribute values, for example, the user often has to revert to lower level implementation techniques. This situation can be improved upon by adding support for these modelling techniques to the modelling environment.

