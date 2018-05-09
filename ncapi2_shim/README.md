# ncapi2_shim

This directory contains a shim python file that allows NCAPI v1 applications run with an installation of NCAPI v2.  
To use the shim do the following to your existing NCAPI v1 program:

Replace the mvnc import line in your program, probably something like this:

```python
from mvnc import mvncapi as mvnc
```

with the following: 

```python
import sys
sys.path.insert(0, "../../ncapi2_shim")
import mvnc_simple_api as mvnc
```

After you do this your program should run as it did before with a few caveats.  There are a few options in NCAPI v1 that were undocumented features and are no longer available such as:

* mvncapi.DeviceOption.OPTIMISATION_LIST
* mvncapi.GraphOption.ITERATIONS

If your NCAPI v1 program references these options, when you try to run with the ncapi2_shim you will get an error.  You can you can safely just remove those references in the program and then the program should run as before.

    

