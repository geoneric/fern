<map version="0.7.1">
<node TEXT="Modelling engine">
<node TEXT="Properties" FOLDED="true" POSITION="right">
<edge WIDTH="thin"/>
<font NAME="SansSerif" SIZE="12"/>
<node TEXT="Easy to extent">
<node TEXT="New operations"/>
<node TEXT="New data formats"/>
<node TEXT="Submodels/3rd party model components/plugins"/>
</node>
<node TEXT="Fast execution"/>
<node TEXT="Scripting interface and interactive "/>
<node TEXT="Operations on multidimensional data" FOLDED="true">
<node TEXT="Spatial"/>
<node TEXT="Temporal"/>
<node TEXT="Uncertain"/>
<node TEXT="Scenarios/alternatives"/>
</node>
<node TEXT="Works with very large data sets"/>
<node TEXT="Possibility to call out to other applications (R, MatLab, whatever)"/>
<node TEXT="Embeddable in other applications"/>
<node TEXT="Works with named attributes, not databases/files/memory"/>
<node TEXT="Works with various types of data">
<node TEXT="Gridded"/>
<node TEXT="Features"/>
</node>
<node TEXT="Python extension, simple things simple"/>
<node TEXT="Works with attributes instead of data sets"/>
<node TEXT="Engine has OpenMI interface"/>
<node TEXT="Internationalized and localizable"/>
<node TEXT="Thread safe operations that are capable of utilizing multiple processing cores"/>
<node TEXT="Dual licensing schema">
<node TEXT="Open source license"/>
<node TEXT="Closed source license">
<node TEXT="License key"/>
<node TEXT="License period"/>
</node>
<node TEXT="Trial of a month"/>
</node>
</node>
<node TEXT="Users" FOLDED="true" POSITION="left">
<node TEXT="Research institutes"/>
<node TEXT="Commercial companies"/>
</node>
<node TEXT="Coolest features" FOLDED="true" POSITION="right">
<node TEXT="Integration of space, time and scenarios"/>
<node TEXT="Handling of uncertainty"/>
<node TEXT="Nested data set, with a data set as attribute"/>
<node TEXT="Free, powerful, flexible, &apos;featureful&apos;"/>
<node TEXT="Extensible library of modelling frameworks">
<node TEXT="Particle filtering"/>
<node TEXT="Monte Carlo analyses"/>
<node TEXT="Simulation modelling"/>
<node TEXT="Kalman filtering"/>
<node TEXT="Plugins of frameworks"/>
</node>
<node TEXT="Fast"/>
<node TEXT="High level"/>
<node TEXT="System can handle value dimensions (units). This is optional. (Only relevant for scalar data?)"/>
</node>
<node TEXT="Scope" FOLDED="true" POSITION="left">
<node TEXT="Does not include">
<node TEXT="Data management tools"/>
<node TEXT="Digitizing tools"/>
<node TEXT="Anything not related to simulation modelling"/>
<node TEXT="Creating nice pictures"/>
</node>
<node TEXT="Does include">
<node TEXT="Library of operations"/>
<node TEXT="Data management tool for nested data set format"/>
<node TEXT="Exploratory data analysis tools"/>
</node>
<node TEXT="&apos;MatLab/R/Python for simulation modelling&apos;"/>
<node TEXT="Can be used to create higher level applications with GUI&apos;s and other functionality"/>
</node>
<node TEXT="Priorities" FOLDED="true" POSITION="right">
<node TEXT="First phase">
<node TEXT="It is important to get all/most of the coolest features in the prototype"/>
<node TEXT="Features in favor of completeness"/>
<node TEXT="Design document containing all details"/>
<node TEXT="Well documented">
<node TEXT="User documentation"/>
<node TEXT="Developer documentation"/>
</node>
<node TEXT="Get the mission statement very clear. Is building this system really the right thing to do?"/>
</node>
<node TEXT="Second phase">
<node TEXT="Scientific articles"/>
<node TEXT="Acquisition for funding new developments"/>
</node>
</node>
<node TEXT="Finances" FOLDED="true" POSITION="left">
<node TEXT="www.nlnet.nl"/>
<node TEXT="Individual investment of time during first phase"/>
</node>
<node TEXT="Misc" FOLDED="true" POSITION="right">
<node TEXT="Support for agent based modelling"/>
<node TEXT="Iterations configurable, iteration context"/>
<node TEXT="Aggregation"/>
<node TEXT="Optimization"/>
<node TEXT="Names">
<node TEXT="GeoLab">
<node TEXT="Tools for a computer lab for spatial simulation models"/>
</node>
<node TEXT="Fiveval">
<node TEXT="From a legend about Grutte Pier. Just a safe and unusual name."/>
</node>
<node TEXT="AttributeAlgebra">
<node TEXT="It is not about  maps or data bases, it&apos;s about attributes. Bit too general. Not clear this is all about spatial simulation models."/>
</node>
<node TEXT="Attrimundo / attribundo">
<node TEXT="It is about attributes and a non-existing word. Does not mention geo, spatial, models though"/>
</node>
</node>
</node>
<node TEXT="Reasons" FOLDED="true" POSITION="left">
<node TEXT="Map algebra is very successful. The same approach can be used to abstract more, including time, scenarios, uncertainty, modelling frameworks."/>
<node TEXT="Simulation modelling involves more hacking than necessary (time, scenarios, uncertainty, optimization, etc). No integrated approach available."/>
<node TEXT="Current solution require to much technical knowledge from the user (databases, programming)."/>
<node TEXT="There is no open platform targetted at simulation modelling."/>
</node>
<node TEXT="Design" POSITION="right">
<node TEXT="Implementation">
<node TEXT="Be able to support larger data sets then available memory."/>
<node TEXT="Other code">
<node TEXT="http://gmt.soest.hawaii.edu"/>
</node>
</node>
<node TEXT="Language" FOLDED="true">
<node TEXT="Python inspired (3)."/>
<node TEXT="Task is deduced from data properties and nesting of sections."/>
<node TEXT="Nesting">
<node TEXT="Sections determine the iteration context. Sections can be nested."/>
<node TEXT="Sections relate to dimensions: space, time, scenarios."/>
<node TEXT="In section X the engine implicitly iterates over dimension X. All expressions are evaluated for eacht location in X. In the nested section X | Y, iteration takes place in both X and Y. For each location in X, all location of Y are visited and the expressions evaluated."/>
<node TEXT="Map algebra has one invisible section: space. We need a way to be able to switch iteration contexts, hence the notion of sections."/>
<node TEXT="How expressions in an iteration context are evaluated should be obvious."/>
<node TEXT="There is a correspondence between the iteration contexts a modelling engine supports and the type of data sets that can be used. In map algebra only rasters can be used."/>
</node>
<node TEXT="Nested data structures become useful, for example to store a spatial attribute per spatial location (raster of rasters). This can be generalized to the notion of nested data sets."/>
</node>
<node TEXT="Components">
<node TEXT="OperationFramework" FOLDED="true">
<node TEXT="LocalOperation">
<node TEXT="Select values from current location"/>
</node>
<node TEXT="FocalOperation">
<node TEXT="Select values from neighborhood / kernel"/>
</node>
<node TEXT="ZonalOperation">
<node TEXT="Select values from zones"/>
</node>
<node TEXT="GlobalOperation">
<node TEXT="Select values from everywhere"/>
</node>
</node>
<node TEXT="PythonExtension" FOLDED="true">
<node TEXT="Simple extension for calling the operations"/>
</node>
<node TEXT="Operations" FOLDED="true">
<node TEXT="Operation families, one name, multiple implementations, selectable by keywords triggering argument lists"/>
<node TEXT="Operations are named by what they do, not by how they are implemented: interpolate instead of idw and kriging"/>
<node TEXT="Plugins"/>
</node>
<node TEXT="Language" FOLDED="true">
<node TEXT="Syntax tree"/>
</node>
<node TEXT="ModellingFramework" FOLDED="true">
<node TEXT="Plugins"/>
</node>
<node TEXT="Meta programming library">
<node TEXT="Functional syntax, with overloaded operators"/>
<node TEXT="Storable expressions/scripts which can be executed"/>
<node TEXT="Upon execution properties of the execution environment can be set (nr of cores, memory, etc)"/>
<node TEXT="See Boost::Spirit"/>
</node>
</node>
</node>
<node TEXT="Terms" FOLDED="true" POSITION="left">
<node TEXT="ModelSolution">
<node TEXT="Combination of ModelEngine and ModelDefinition"/>
</node>
<node TEXT="ModelEngine">
<node TEXT="Piece of software capable of executing a ModelDefinition"/>
</node>
<node TEXT="ModelDefinition">
<node TEXT="Piece of information containing all information required to execute the model"/>
</node>
</node>
</node>
</map>
