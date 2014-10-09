File format
===========
In this section we try to map the data model to a HDF5 file format. These are just initial thoughts that will change.

Feature
-------
A feature is stored in an HDF5 file. Multiple, unrelated features can be stored in a single HDF5 file. This is up to the user.

A feature is a group.

The path name of each feature equals the name of the feature. These names must be unique per HDF5 file. Features are stored in the root group.

.. code-block:: bash

    /feature_1
    /feature_2
    /...
    /feature_n

Domain
------
Every feature contains a domain. The domain is stored directly under the feature.

A domain is a data set.

The path name of the domain is `__domain__`, which can be considered to be a reserved keyword. This name cannot be used to name feature-attributes.

.. code-block:: bash

    /feature/__domain__

Attribute
---------
An attribute is a group.

The path name of each attribute equals the name of each attribute. These names must be unique per feature.

.. code-block:: bash

    /feature/attribute_1
    /feature/attribute_2
    /feature/...
    /feature/attribute_n

Value
-----
If an attribute has a value, it is stored at `__value__`, which is a reserved keyword.

A value is a data set.

.. code-block:: bash

    /feature/attribute/__value__

