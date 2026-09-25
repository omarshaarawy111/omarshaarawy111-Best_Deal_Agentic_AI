# For deployment issues handling
import truststore
truststore.inject_into_ssl()         
import sys
import runpy

if __name__ == "__main__":
    # Simulate terminal 
    # Like running Modal in shell
    # argv[1:]: grabs whatever arguments were passed to this script
    sys.argv = ['modal'] + sys.argv[1:] 
    # Finds the installed modal package, and executes it
    runpy.run_module('modal', run_name='__main__')