#!/bin/sh

# -- job description --
#SBATCH --job-name="hvd-2node-2GPU-2CPU"

# -- resource allocation --
#SBATCH --partition=GPUQ		
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB

# -- I/O --
#SBATCH --output=climatejob_%j.out
#SBATCH --error=climatejob_%j.err
#SBATCH --export=ALL

#Add the following to ignore any incidental GPUs you may have: os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Hide the GPUs from tensorflow, if any
#To requisition a live node: salloc --nodes=1 --partition=GPUQ --gres=gpu:1 --time=00:30:00
export OMP_NUM_THREADS=$SLURM_NTASKS
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,0,0
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL

module purge
module load fosscuda/2019b
module load NCCL/2.4.8
module load TensorFlow/2.1.0-Python-3.7.4
module load GDAL/3.0.2-Python-3.7.4

source venv-climate/bin/activate

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Job NTASKS: ${SLURM_NTASKS}"
echo "== Job NPROCS: ${SLURM_NPROCS}"
echo "== Job NNODES: ${SLURM_NNODES}"
NODE_LIST=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:2,/g' )
NODE_LIST=${NODE_LIST%?}
echo "== Node list: ${NODE_LIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
echo "Command: mpirun -np $SLURM_NTASKS -H $NODE_LIST -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=^lo -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_openib_verbose 1 --mca btl_tcp_if_include ib0 python main.py"

mpirun -np $SLURM_NTASKS -H $NODE_LIST -bind-to none -map-by slot -x NCCL_SOCKET_IFNAME=^lo -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_openib_verbose 1 --mca btl_tcp_if_include ib0 python main.py
#horovodrun -np ${SLURM_NTASKS} -H $NODE_LIST  python main.py --variable_update horovod

echo "== Job's done at $(date)"

