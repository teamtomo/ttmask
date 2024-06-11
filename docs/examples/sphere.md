#  Sphere

To view the available options for the sphere command, make use of the '--help' provision. 

```shell

ttmask sphere --help

```

## Commands and Options

Option | Usage                                                                                                                                                                                                                                      |
------------ |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
--sidelength | This specifies the your box size. e.g. a 100 x 100 x 100 pixel array.                                                                                                                                                                      | 
--sphere-diameter  | The diameter of your sphere in angstrom, measured from outermost edge to edge.                                                                                                                                                             |
--soft-edge-width | The number of pixels over which a soft edge will be added.                                                                                                                                                                                 |
--pixel-size  | The desired pixel size of your output. If left blank, 1 will be taken as default.                                                                                                                                                          |
--output | Name of your output file, e.g. sphere5.mrc                                                                                                                                                                                                 |
--wall-thickness  | If specified (in angstrom), the sphere will be hollow, with this thickness wall. The thickness is not added to the sphere-diameter, but instead is added inward of that boundary such that your specified diameter still remains accurate. |




    