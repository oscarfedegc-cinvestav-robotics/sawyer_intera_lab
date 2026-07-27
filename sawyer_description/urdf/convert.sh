cd /home/nexus/Documents/GitHub/sawyer_intera_lab/sawyer_description/urdf/

conda init
conda activate xacro

xacro sawyer.urdf.xacro > sawyer.urdf

conda init
conda deactivate

conda init
conda activate copt
/home/nexus/miniforge3/envs/copt/bin/python /home/nexus/Documents/GitHub/sawyer_intera_lab/sawyer_description/urdf/reduce_sawyer.py