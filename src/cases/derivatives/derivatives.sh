#!/bin/bash
# script to take derivatives of spins outputs
# usage:
#   ./derivative.sh

# set options
num_procs=4
mem=5g
runtime=30m
fields=(u v w)
ts=(0 1)

##### User need not change anything below this line #####

# function to change variable in spins.conf
function change_spins_conf {
sed -i "s/^${1}.*/$1 = $2/" spins.conf
}

# adjust spins.conf, and then run derivative.x
for ii in "${ts[@]}"; do
    change_spins_conf deriv_sequence $ii
    for var in "${fields[@]}"; do
        change_spins_conf deriv_file $var

        #### take derivative
        # locally on sharcnet
        /opt/sharcnet/openmpi/1.6.2/intel/bin/mpirun -np $num_procs derivative.x
        # submit a sharcnet job
        #sqsub -q kglamb -f mpi -n $numprocs -r ${runtime} -o derivative.log --mpp=${mem} -j derivative derivative.x
        # on local machine
        #mpirun -np $numprocs derivative.x
    done
done

