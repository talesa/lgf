from plumbum import local
from plumbum.cmd import sed, awk, git
import time

def inspect_gpus(memory_threshold=500,
                 gpu_util_threshold=5,
                 allow_lightly_used_gpus=False,
                 share_with=tuple(),
                 max_nr_processes=2,
                 upper_memory_threshold=3000,
                 upper_gpu_util_threshold=30,
                 average=1):
    """
    Scan servers for free GPUs, print availability and return a list of free GPUs that can used to
    start jobs on them.
    Requirements:
        ~/.ssh/config needs to be set up so that connecting via `ssh <server>` works. Fos OSX,
        an entry can look like this:
        Host mulga
            User maxigl
            HostName mulga.cs.ox.ac.uk
            BatchMode yes
            ForwardAgent yes
            StrictHostKeyChecking no
            AddKeysToAgent yes
            UseKeychain yes
            IdentityFile ~/.ssh/id_rsa
    Args:
        verbose (bool):           If True, also print who is using the GPUs
        memory_threshold (int):
        gpu_util_threshold (int): When used memory < memory_threshold and
                                  GPU utilisation < gpu_util_threshold,
                                  then the GPU is regarded as free.
        allow_lightly_used_gpus (bool):
        share_with (tuple of strings):
        upper_memory_threshold (int):
        upper_gpu_util_threshold (int): If `allow_lightly_used_gpus=True` and memory and gpu
                                        utilisation are under the upper thresholds and there
                                        is so far only one process executed on that GPU who's
                                        user is in in the list `share_with`, then the GPU will
                                        be added to the list of GPUs that can be used to start jobs.
    Return:
        free_gpus: List of dictionaries, each containing the following keys:
                   'gpu_nr': Number of the free GPU
                   'double': Whether someone is already using that GPU but it's still considered
                             usuable (see `allow_lightly_used_gpus`)
    """

    r_smi = local["nvidia_smi"]
    r_ps = local["ps"]
    averaged_gpu_data = []
    for avg_idx in range(average):
        fieldnames = ['index', 'gpu_uuid', 'memory.total', 'memory.used',
                    'utilization.gpu', 'gpu_name']
        output = r_smi("--query-gpu=" + ",".join(fieldnames),
                    "--format=csv,noheader,nounits").replace(" ", "")

        gpu_data = []
        for line in output.splitlines():
            gpu_data.append(dict([(name, int(x)) if x.strip().isdigit() else (name, x)
                            for x, name in zip(line.split(","), fieldnames)]))
        if avg_idx == 0:
            averaged_gpu_data = gpu_data
            for gpu_idx in range(len(averaged_gpu_data)):
                averaged_gpu_data[gpu_idx]['utilization.gpu'] /= average
                averaged_gpu_data[gpu_idx]['memory.used'] /= average
        else:
            for gpu_idx, data in enumerate(gpu_data):
                averaged_gpu_data[gpu_idx]['utilization.gpu'] += data['utilization.gpu'] / average
                averaged_gpu_data[gpu_idx]['memory.used'] += data['memory.used'] / average
        time.sleep(1.)

    gpu_data = averaged_gpu_data


    # Find processes and users
    for data in gpu_data:
        data['nr_processes'] = 0
        data['users'] = []

    output = r_smi("--query-compute-apps=pid,gpu_uuid",
                   "--format=csv,noheader,nounits").replace(" ", "")

    gpu_processes = []
    for line in output.splitlines():
        gpu_processes.append([int(x) if x.strip().isdigit() else x for x in line.split(",")])

    for process in gpu_processes:
        pid = process[0]
        user = (r_ps['-u', '-p'] | sed['-n', '2p'] | awk['{{print $1}}'])(pid)
        serial = process[1]
        for data in gpu_data:
            if data['gpu_uuid'] == serial:
                data['users'].append(user.strip())
                data['nr_processes'] += 1

    free_gpus = []

    for data in gpu_data:
        # Is it free?
        if (data['memory.used'] < memory_threshold and
            data['utilization.gpu'] < gpu_util_threshold):

            free_gpus.append({'gpu_nr': data['index'],
                              'occupation': 0})
                              # 'session': getSession(data['index'])})
        elif (allow_lightly_used_gpus and
            data['memory.used'] < upper_memory_threshold and
            data['utilization.gpu'] < upper_gpu_util_threshold and
            data['nr_processes'] < max_nr_processes and
            data['users'][0] in share_with):

            free_gpus.append({'gpu_nr': data['index'],
                              'occupation': data['nr_processes']})

    return free_gpus




datasets_dims = {
    'mnist': 28*28,
    'fashion-mnist': 28*28,
    'miniboone': 43,
    'gas': 8,
    'power': 6,
    'hepmass': 21,
    'bsds300': 21,
}



import subprocess
import os
import torch
import itertools

gpus_list = list(range(8))
num_gpus = len(gpus_list)


filename = 'main.py'

def execute_process(params, gpuid):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    dataset, model, lr, num_flow_layers, num_ortho_vecs, num_householder = params
    command = ['python', filename,
               '--dataset', dataset,
               '--model', model,
               '--config', f'num_flow_layers={num_flow_layers}',
               '--config', f'num_ortho_vecs={int(num_ortho_vecs * datasets_dims[dataset])}',
               '--config', f'num_householder={int(num_householder * datasets_dims[dataset])}',
               '--config', f'lr={lr}'
               ]
    command = list(map(str, command))
    print(f"CUDA_VISIBLE_DEVICES={str(gpuid)} {' '.join(command)}")
    subprocess.Popen(command, env=env)
    # return


params_list = []

datasets = [
    # 'mnist',
    'miniboone',
    # 'gas',
    # 'power',
]
models = [
    "sylvester-orthogonal",
    "sylvester-householder",
    "sylvester-triangular",
    "sylvester-exponential",
    "sylvester-cayley"
]
lrs = [5e-4]

num_flow_layers = [32]

# multiplier for the number of dimensions of the dataset
num_ortho_vecs = [1.]
num_householder = [1.]

tries = 1
params_list += list(itertools.product(datasets, models, lrs[::-1], num_flow_layers, num_ortho_vecs, num_householder)) * tries

# lrs = [5e-3, 2e-3, 8e-4, 5e-4, 2e-4, 8e-5, 5e-5]

print(params_list)

while params_list:
    free_gpus = inspect_gpus(
        memory_threshold=5500,
        gpu_util_threshold=50,
        average=2,
        share_with=('agolinsk',))
    if free_gpus:
        params = params_list.pop()
        execute_process(params, free_gpus[0]['gpu_nr'])
    time.sleep(30.)

# while params_list:
#     free_gpus = inspect_gpus(average=2)
#     if [i['gpu_nr'] for i in free_gpus if i['gpu_nr'] == 6]:
#         params = params_list.pop()
#         execute_process(params, 6)
#     time.sleep(30.)

print("Done")