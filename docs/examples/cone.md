#  Cone

To view the available options for the cone command, make use of the '--help' provision. 

```shell

ttmask cone --help

```

## Commands and Options

Option | Usage                                                                             |
------------ |-----------------------------------------------------------------------------------| 
--sidelength | This specifies the your box size. e.g. a 100 x 100 x 100 pixel array.             | 
--cone-height  | The height of your cone in angstrom.                                              |
--cone-base-diameter | The diameter of the base (i.e. widest diameter) of the cone.                      
--soft-edge-width | The number of pixels over which a soft edge will be added.                        |
--pixel-size  | The desired pixel size of your output. If left blank, 1 will be taken as default. |
--output | Name of your output file, e.g. cone1.mrc                                          |

Note : Currently hollow cones are not available due to some quirks as to how this needs to be specified.                                                                    




    