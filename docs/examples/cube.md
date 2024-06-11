#  Cube

To view the available options for the cube command, make use of the '--help' provision. 

```shell

ttmask cube --help

```

## Commands and Options

Option | Usage                                                                                                                                                                                                      |
------------ |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
--sidelength | This specifies the your box size. e.g. a 100 x 100 x 100 pixel array.                                                                                                                                      | 
--cube-sidelength  | The edge length of your cube in angstrom.                                                                                                                                                                  |
--soft-edge-width | The number of pixels over which a soft edge will be added.                                                                                                                                                 |
--pixel-size  | The desired pixel size of your output. If left blank, 1 will be taken as default.                                                                                                                          |
--output | Name of your output file, e.g. cube5.mrc                                                                                                                                                                   |
--wall-thickness  | If specified (in angstrom), the cube will be hollow. The walls of the cube will be given this thickness, but will be added inwards of the boundary, such that cube-sidelength dimensions are not exceeded. |




    