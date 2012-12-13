Introduction
============
This document describes concepts that are common to feature-based and field-based modelling with uncertain spatio-temporal feature attributes. The ultimate goal is to develop a modelling environment for uncertain attributes of spatio-temporal phenomena, descritisized as features or as fields.

Yet another modelling language???
---------------------------------
In our view, a modelling language is a language for expressing environmental models, by modellers. Modellers are domain experts who are not necessarely knowledgeable or interested in software development. They need an environment with a high level of abstraction. A modelling language, like a scripting language or a graphical language for example, provides the means for the domain expert to express his ideas about the phenomena being modelled. Most domain experts aren't able to express such ideas in lower level languages like C++, C#, Java or even Python. The use of these languages require the domain expert to know things that are not directly related to expressing a model, like managing computer memory, managing files, handling errors, etc.

Another reason to provide a modelling environment directly to the domain expert, instead of asking a software developer to develop models for the domain expert, is that important decisions that have to be made during the development of the model get taken by the domain expert, instead of the developer. Like software development, model development is a highly iterative process, and decissions about the implementation need to be made continuously during the development of a model. Only for the most trivial models can the domain expert provide the software developer with the full specification of the model beforehand. In most cases the requirements of the model will be adjusted continuously, based on the model's qualitative and quantitative performance.

[Compare with waterfall method vs Agile. What about an Agile method of model development? -> Iterations are too small/fast to be feaseable, comparable with edit/compile/run cycles. Not feasible to work with developer unless he's sitting in the same room. Domain expert needs to be in control.]

Separate modelling approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, there is no high level, integrated, feature-based and field-based modelling environment. Although in some agent-based modelling environments it is possible to use information from rasters, and although in field based modelling agents are sometimes represented by raster cells, conceptually these modelling approaches/paradigms have very little in common. They lack a set of common concepts that allow the modeller to handle fields and features in an integrative manner.

[Discuss what we mean by high level. Explicit looping is not high-level. Class definitions are not high level.]

Lack of format for uncertain spatio-temporal attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Information about the temporal domain and the uncertainty in attribute values is hardly ever stored along with the attribute's values. There is a lack of data set formats to store uncertain spatio-temporal attribute values. Without such formats, modelling environments cannot persist uncertain spatio-temporal attribute values.

No support for modelling techniques
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Although modelling environments exist that allow users to model spatial and spatio-temporal phenomena, when it comes to handling uncertainty with a Monte Carlo analysis, or assimilation of observation values, for example, the user often has to revert to lower level implementation techniques. This situation can be improved upon by adding support for these modelling techniques to the modelling environment.

