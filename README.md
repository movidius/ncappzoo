# Example code for the Movidius Neural Compute Stick.

See the LICENSE file for details of reuse and redistribution of repo files.

# Create shell scripts in /usr/local/bin for easy access

sudo cat > /usr/local/bin/ncCompile

#! /bin/bash

python3 /home/[user]/workspace/mvncsdk/bin/mvNCCompile.pyc $*

"ctrl-d"

sudo cat > /usr/local/bin/ncCProfile

#! /bin/bash

python3 /home/[user]/workspace/mvncsdk/bin/mvNCProfile.pyc $*

"ctrl-d"

sudo cat > /usr/local/bin/ncCheck

#! /bin/bash

python3 /home/[user]/workspace/mvncsdk/bin/mvNCCheck.pyc $*

"ctrl-d"

