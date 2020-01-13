Modify this template as needed to best reflect your specific pull request.

# Type

Please state what type of addition your pull request is:    
- [ ] Add App |   
- [ ] Add Network |   
- [ ] Repository Enhancement |    
- [ ] Bugfix (App) |  
- [ ] Bugfix (Network) |  
- [ ] Bugfix (ncappzoo) |     
- [ ] Content Update (Documentation) |    

# Description

Please include a summary of the addition or change that you are making. If making a change to an existing application or network, make sure to state which network or app you are changing. If you're making a change to an existing network or app, don't forget to update the relevant code author documentation. Failure to update ownership and authorship documentation or not including enough information for maintainers to update documentation may result in requested changes to your pull request or your pull request being closed without merging:


### Issue \# [ ]

If your change fixes a known bug, please list the issue. If your change fixes an unknown bug, please submit an issue before submitting your pull request. A pull request to fix a bug with an app, network, or the App Zoo will be prioritized if it has a matching issue.

# Testing

#### Tested?: [ ]
#### OS (Include Version): [ ]
#### OpenVINO Version: [ ]
#### Python Version: [ ]
#### Additional Info:

How has your change been tested? Have you tested your change on Ubuntu with an Intel&reg; Movidius&trade; Neural Compute Stick or Intel&reg; Neural Compute Stick 2? Describe your testing procedure here, including OS, installed OpenVINO version, Python version, and applicable compiler versions.

If you have not tested your change, make sure to clearly label that it has **not been tested.**

# Checklist

- [ ] I have self-reviewed my code
- [ ] I have commented my code, especially in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation:
    - [ ] I have included an AUTHORS file with my app or network inclusion (or this is not applicable)
    - [ ] I have updated the app or network readme to include information about my inclusion (or this is not applicable)
    - [ ] I have updated the OWNERS file on the top level directory of the ncappzoo with the relevant information (or this is not applicable)
- [ ] Any dependent changes have been merged and published in downstream modules.