#  Ellipsoid

To view the available options for the ellipsoid command, make use of the '--help' provision. 

```shell

ttmask ellipsoid --help

```

## Commands and Options

Option | Usage                                                                                                                                                                                                                                               |
------------ |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
--sidelength | This specifies the your box size. e.g. a 100 x 100 x 100 pixel array.                                                                                                                                                                               | 
--ellipsoid-dimensions  | The lengths of your ellipsoid in angstrom, specied as three space seperated floats e.g. 50 30 30. These correspond to the depth, height and width of your ellipsoid respectively (i.e. zyx)                                                         |
--soft-edge-width | The number of pixels over which a soft edge will be added.                                                                                                                                                                                          |
--pixel-size  | The desired pixel size of your output. If left blank, 1 will be taken as default.                                                                                                                                                                   |
--output | Name of your output file, e.g. ellipsoid1.mrc                                                                                                                                                                                                       |
--wall-thickness  | If specified (in angstrom), the ellipsoid will be hollow, with this thickness wall. The thickness is not added to the ellipsoid-dimensions, but instead is added inward of that boundary such that your specified dimensions still remain accurate. |




    