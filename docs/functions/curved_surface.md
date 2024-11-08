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
--centering | The default is "--centering standard", in which the shape is placed at the center of the box (i.e. [sidelength/2, sidelength/2, sidelength/2]). However, if you would like the shape to 'appear' centered within an even box size then use "--centering visual", which shifts the center half a pixel. Alternatively, one may use "--centering custom" together with the "--center" flag (see next table entry). 
--center | If using "--centering custom" then specify the custom center here. E.g. for a box size of 100, if you want to shift the mask 10 pixels in Z, then one could specify the new center using "--center 40 50 50". The convention here is "--center z y x" (depth, height, width).                                                                                                                                    

Note : Currently hollow cones are not available due to some quirks as to how this needs to be specified.                                                                    




    