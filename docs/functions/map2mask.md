# Map to Mask

To view the available options for the map2mask command, make use of the '--help' provision. 

```shell

ttmask map2mask --help

```

## Commands and Options

Option | Usage                                                                                                |
---------------------------|------------------------------------------------------------------------------------------------------| 
--input-map | The input map to be binarized (mrc file format).                                                     | 
--binarization-threshold  | All values greater than this threshold are set to 1 and all values below this threshold are set to 0 |
--padding-width | The number of pixels by which to extend the mask, adding a given number of 1s.  |
--soft-edge-width | The number of pixels over which a soft edge will be added.                                           |
--pixel-size  | The desired pixel size of your output. If left blank, 1 will be taken as default.                    |
--output-mask | Name of your output file                                                                             |





    