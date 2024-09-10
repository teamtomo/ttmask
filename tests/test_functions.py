from ttmask import cone, cube, cuboid, cylinder, ellipsoid, map2mask, sphere, tube
import numpy as np

def test_cone():
    mask = cone(100, 50, 50, 0, 1)
    assert mask.shape == (100, 100, 100)
    assert mask.sum() > np.pi * 24**2 * (50 / 3) # Volume of cone
    assert mask.sum() < np.pi * 25**2 * 50 # Volume of cylinder

def test_cube():
    mask = cube(100, 50, 0,1)
    assert mask.shape == (100, 100, 100)
    # Test against volume of cube +- center and subpixel issues
    assert mask.sum() > 50**3
    assert mask.sum() < 52**3

def test_cuboid():
    mask = cuboid(100, (50,40,30), 0, 1)
    assert mask.shape == (100, 100, 100)
    # Test against volume of cuboid +- center and subpixel issues
    assert mask.sum() > 50 * 40 * 30
    assert mask.sum() < 52 * 42 * 32

#def test_curved_surface():
#    mask = curved_surface(100, 50, 50, 0, 1)
#    assert mask.shape == (100, 100, 100)
#    assert mask.sum() > 2 * np.pi * 25**2 # Area of cylinder
#    assert mask.sum() < 2 * np.pi * 25 * 50 # Area of cylinder

def test_cylinder():
    mask = cylinder(100, 50, 50, 0, 0, 1)
    assert mask.shape == (100, 100, 100)
    assert mask.sum() > np.pi * 25**2 * 48 # Volume of cylinder
    assert mask.sum() < np.pi * 25**2 * 51 # Volume of cylinder

def test_ellipsoid():
    mask = ellipsoid(100, (50,40,30), 0, 1,0)
    assert mask.shape == (100, 100, 100)
    # Test against volume of ellipsoid +- center and subpixel issues
    assert mask.sum() > 24 * 19 * 14 * 4/3 * np.pi
    assert mask.sum() < 26 * 21 * 16 * 4/3 * np.pi

def test_sphere():
    mask = sphere(100, 50, 0, 1,0)
    assert mask.shape == (100, 100, 100)
    assert mask.sum() > 4/3 * np.pi * 24**3 # Volume of sphere
    assert mask.sum() < 4/3 * np.pi * 26**3 # Volume of sphere

def test_tube():
    mask = tube(100, 50, 50, 0, 0, 1)
    assert mask.shape == (100, 100, 100)
    assert mask.sum() > np.pi * 24**2 * 48 # Volume of tube
    assert mask.sum() < np.pi * 26**2 * 52 # Volume of tube