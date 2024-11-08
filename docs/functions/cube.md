#  Cube

To view the available options for the cube command, make use of the '--help' provision. 

```shell

ttmask cube --help

```

## Commands and Options

Option | Usage                                                                                                                                                                                                      |
------------ |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
--sidelength | This specifies the box size. e.g. "--sidelength 100" for a 100 x 100 x 100 pixel array.                                                                                                                                      | 
--cube-sidelength  | The edge length of your cube in angstrom.                                                                                                                                                                  |
--soft-edge-width | The number of pixels over which a soft edge will be added.                                                                                                                                                                                                                                                                                                                                                       |
--pixel-size  | The desired pixel size of your output in angstroms.                                                                                                                                                                                                                                                                                                                    |
--output | Name of your output file                                                                                                                                                                               |
--wall-thickness  | If specified (in angstrom), the cube will be hollow. The walls of the cube will be given this thickness, but will be added inwards of the boundary, such that cube-sidelength dimensions are not exceeded. |
--centering | The default is "--centering standard", in which the shape is placed at the center of the box (i.e. [sidelength/2, sidelength/2, sidelength/2]). However, if you would like the shape to 'appear' centered within an even box size then use "--centering visual", which shifts the center half a pixel. Alternatively, one may use "--centering custom" together with the "--center" flag (see next table entry). 
--center | If using "--centering custom" then specify the custom center here. E.g. for a box size of 100, if you want to shift the mask 10 pixels in Z, then specify the new center using "--center 40 50 50". The convention here is "--center z y x" (depth, height, width).                                                                                                                                  




    