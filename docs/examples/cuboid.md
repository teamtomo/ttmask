#  Cuboid

To view the available options for the cuboid command, make use of the '--help' provision. 

```shell

ttmask cuboid --help

```

## Commands and Options

Option | Usage                                                                                                                                                                                                           |
------------ |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
--sidelength | This specifies the your box size. e.g. a 100 x 100 x 100 pixel array.                                                                                                                                           | 
--cuboid-sidelengths  | The edge lengths of your cuboid in angstrom, specied as three space seperated floats e.g. 50 50 50. These correspond to the depth, height and width of your cuboid respectively (i.e. zyx)                      |
--soft-edge-width | The number of pixels over which a soft edge will be added.                                                                                                                                                      |
--pixel-size  | The desired pixel size of your output. If left blank, 1 will be taken as default.                                                                                                                               |
--output | Name of your output file, e.g. cuboid5.mrc                                                                                                                                                                      |
--wall-thickness  | If specified (in angstrom), the cuboid will be hollow. The walls of the cube will be given this thickness, but will be added inwards of the boundary, such that cuboid-sidelengths dimensions are not exceeded. |




    