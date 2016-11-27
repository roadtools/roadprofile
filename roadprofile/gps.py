from math import pi as PI
import numpy as np
from numpy import cos, sin, arccos, sqrt, fabs

# Latitude <-> wgs_N
# Longitude <-> wgs_E

RAD_EARTH_METER = 6373000
DEGREES2RADIANS = PI/180.0

def calc_curvature_radi(wgs_N, wgs_E):
    curves = np.zeros(wgs_N.shape)
    dist_ab = distance_on_unit_sphere(wgs_N[:-1], wgs_E[:-1], wgs_N[1:], wgs_E[1:])
    dist_c = distance_on_unit_sphere(wgs_N[:-2], wgs_E[:-2], wgs_N[2:], wgs_E[2:])
    curves[1:-1] = circum_circle_radius(dist_ab[:-1], dist_ab[1:], dist_c)
    # To get arrays of same size as input.
    curves[0] = curves[1]
    curves[-1] = curves[-2]
    return curves

def calc_lengths(wgs_N, wgs_E):
    return distance_on_unit_sphere(wgs_N[:-1], wgs_E[:-1], wgs_N[1:], wgs_E[1:])

def circum_circle_radius(a, b, c):
# Circumcircle radius calculation from http://www.mathopenref.com/trianglecircumcircle.html
# or https://en.wikipedia.org/wiki/Circumscribed_circle#Other_properties
    try:
        divider = sqrt(fabs((a+b+c) * (b+c-a) * (c+a-b) * (a+b-c)))
        return (a * b * c) / divider
    except ZeroDivisionError:
        return 10000

def distance_on_unit_sphere(lat1, long1, lat2, long2):
    # NOTE this guy has more accurate calculation methods here: https://github.com/balzer82/LatLon2Meter/blob/master/LatLon2Meter.py
    # From http://www.johndcook.com/python_longitude_latitude.html
    #if lat1 == lat2 and long1 == long2:
    #    return 0
    phi1 = (90.0 - lat1) * DEGREES2RADIANS
    phi2 = (90.0 - lat2) * DEGREES2RADIANS

    # theta = longitude
    theta1 = long1 * DEGREES2RADIANS
    theta2 = long2 * DEGREES2RADIANS

    # Compute spherical distance from spherical coordinates.
    dot_product = (sin(phi1) * sin(phi2) * cos(theta1 - theta2) +
           cos(phi1) * cos(phi2))
    #if dot_product > 1:
    #    return 0
    arc = arccos(dot_product)
    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc * RAD_EARTH_METER
