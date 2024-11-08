# Cylinder

To view the available options for the cylinder command, make use of the '--help' provision. 

```shell

ttmask cylinder --help

```

## Commands and Options

Option | Usage                                                                                                                                                                                                                                          |
------------ |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
--sidelength | This specifies the box size. e.g. "--sidelength 100" for a 100 x 100 x 100 pixel array.                                                                                                                                                                          | 
--cylinder-height | The height of your cylinder in angstrom. 
--cylinder-diameter  | The diameter of your cylinder in angstrom, measured from outermost edge to edge.                                                                                                                                                               |
--soft-edge-width | The number of pixels over which a soft edge will be added.                                                                                                                                                                                     |
--pixel-size  | The desired pixel size of your output in angstroms.                                                                                                                                                              |
--output | Name of your output file                                                                                                                                                                                                        |
--wall-thickness  | If specified (in angstrom), the cylinder will be hollow, with this thickness wall. The thickness is not added to the cylinder-diameter, but instead is added inward of that boundary such that your specified diameter still remains accurate. |
--centering | The default is "--centering standard", in which the shape is placed at the center of the box (i.e. [sidelength/2, sidelength/2, sidelength/2]). However, if you would like the shape to 'appear' centered within an even box size then use "--centering visual", which shifts the center half a pixel. Alternatively, one may use "--centering custom" together with the "--center" flag (see next table entry). 
--center | If using "--centering custom" then specify the custom center here. E.g. for a box size of 100, if you want to shift the mask 10 pixels in Z, then specify the new center using "--center 40 50 50". The convention here is "--center z y x" (depth, height, width).                                                                                                                      




    