Introduction
============
This document describes concepts that are common to feature-based and field-based modelling with uncertain spatio-temporal feature attributes. The ultimate goal is to develop a modelling environment for uncertain attributes of spatio-temporal phenomena, whether they are descritisized as features or as fields.

Yet another modelling language???
---------------------------------
In our view, a modelling language is a language for expressing environmental models, by modellers. Modellers are domain experts who are not necessarely knowledgeable or interested in software development. They want to use an environment with a high level of abstraction. A modelling language, like a script language or a graphical language, provides the means for the domain expert to express his ideas about the phenomena being modelled. Most domain experts aren't able to express such ideas in lower level languages like C++, C#, Java or even, to a lesser extent, Python. These languages make it necessary that the domain expert needs to know things that are not directly related to expressing a model, like memory management, file management, error handling, etc.

Another reason to provide a modelling environment directly to the domain expert, instead of asking a software developer to develope models for the domain expert, is that important decisions that have to be made during the development of the model get taken by the domain expert, instead of the developer. Like software development, model development is a highly itterative process, and decissions about the implementation need to be made continuously. Only for the most trivial models can the domain expert provide the software developer with the full specification of the model beforehand. In most cases the requirements of the model get adjusted continuously, based on the model's performance.

[Compare with waterfall method vs Agile. What about an Agile method of model development? -> Iterations are too small/fast to be feaseable, comparable with edit/compile/run cycles. Domain expert needs to be in control.]

Separate modelling approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, there is no high level, integrated, feature-based and field-based modelling environment. Although in some agent-base modelling environments it is possible to use information from rasters, and although in field based modelling agents are sometimes represented by raster cells, conceptually these modelling worlds have very little in common. They lack a set of common concepts.

Lack of format for uncertain spatio-temporal attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Information about the temporal domain and the uncertainty in attribute values is hardly ever stored along with the attribute's values. There is a lack of data set formats to store uncertain spatio-temporal attribute values.

No support for modelling techniques
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Although modelling environments exist that allow users to model spatial and spatio-temporal phenomena, when it comes to handling uncertainty with a Monte Carlo analysis, or assimilation of observation values, for example, the user often has to revert to lower level implementation techniques. This situation can be improved upon by adding support for these modelling techniques to the modelling environment.

