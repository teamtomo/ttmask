# Overview

*ttmask* is a Python package for generating masks and references for cryo-EM data analysis. 

*ttmask* can be used to create masks of various geometries, or from existing cryo-EM maps. 

---
# Installation
The package is available for download via pip.

```shell

pip install ttmask

```
# Quickstart

To view the available options in ttmask and understand which parameters are required, make use of the '--help' provision. 

```shell

ttmask --help

```
Then choose the command you desire and fill in the options, like so : 

```shell

ttmask cylinder --sidelength 200 --pixel-size 0.5 --cylinder-height 60 --cylinder-diameter 20

```

In this example, we have generated a cylinder 60 angstrom height and 20 angstrom across, in a 200 x 200 x 200 pixel box at 0.5 angstrom per pixel.

Now we could think about making the cylindrical mask hollow and adding a soft edge to the mask : 


```shell

ttmask cylinder --sidelength 200 --pixel-size 0.5 --cylinder-height 60 --cylinder-diameter 20 --soft-edge-width 3 --wall-thickness 3

```

[//]: # (![type:video]&#40;./videos/costa_rica.mp4&#41;)