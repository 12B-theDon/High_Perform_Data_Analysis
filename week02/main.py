# sudo -i
# Print Code Profiling
from julia1 import calculate_Julia_set, build_Julia_set
from print_julia import printVer_build_Julia_set
from detector_with_time import timefn, timefnVer_calculate_Julia_set
#print(" ==== Begin: print ===")
#output = printVer_build_Julia_set(desired_width = 1000, max_iterations = 300)
#print(" ====  End : print ===")

# apt install time
# vim ~./bashrc
# alias time = '/usr/bin/time'
print(" ==== Begin: decorator ===")
zs = [complex(0.3, 0.5) for _ in range(100000)]
cs = [complex(-0.8, 0.156) for _ in range(100000)]
output = timefnVer_calculate_Julia_set(300, zs, cs)
print(" ====  End : decorator ===")

# timeit module
# python3 -m timeit -n 5 -r 1 -s "import julia1" "julia1.build_Julia_set(desired_width = 1000, max_iterations = 300)"

# python3 -m timeit -n 5 -r 2 -s "import julia1" "julia1.build_Julia_set(desired_width = 1000, max_iterations = 300)"

# python3 -m timeit -n 5 -r 10 -s "import julia1" "julia1.build_Julia_set(desired_width = 1000, max_iterations = 300)"

# Ubuntu time command: real, user, sys
## Single CPU
# docker run -it -v /home/mhlee/Documents/High_Performance_Data_Analysis/week02/:/workspace --cpus 1 --name ubuntu_single_CPU ubuntu:latest
# apt update && apt install -y python3 python3-pip
# apt install time
# vim ~./bashrc
# alias time = '/usr/bin/time'
# time -p python3 julia1.py

## Multi CPUs
# docker run -it -v /home/mhlee/Documents/High_Performance_Data_Analysis/week02/:/workspace --name ubuntu_all_CPUs ubuntu:latest
# apt install time
# vim ~./bashrc
# alias time = '/usr/bin/time'

# time -p python3 julia1.py

#cProfile
# python3 saveCprofile_julia1.py 

#snakeviz
# pip3 install snakeviz

#py-spy
# pip3 install py-spy
# py-spy record -o profile.svg -- python3 julia1.py
