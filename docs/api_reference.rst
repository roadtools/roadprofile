.. _api-reference:

#############
API reference
#############

This is the entire list of modules, classes, functions and minor objects used in RoadProfile.

Filtering
---------
Filtering algorithms that can remove interpolate invalid readings, detect outliers, perform frequency band filtering and enveloping.

.. currentmodule:: roadprofile

.. automodule:: roadprofile

    .. autofunction:: interpolate_dropouts
    .. autofunction:: envelope


Texture Metrics
---------------
Texture metrics that can be calculated from a texture profile.

.. autofunction:: calculate_mpd

.. autofunction:: calculate_tpa
