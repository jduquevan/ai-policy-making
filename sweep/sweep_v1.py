import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'ppo_epochs': [1, 2, 3],
        'use_aa': [True],
        'aa_gamma': [0.99, 0.96, 0.9, 0.8],
        'aa_beta': [0.5, 1, 2, 3],
        'chunk_length': [15, 20, 25],
        'use_rnn': [False],
        'pi_lr': [1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
        'val_lr': [1e-6, 1e-5, 1e-4],
        'entropy_coef': [0.001, 0.01, 0.1],
        'self_play': [False],
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    # submit this job using slurm
    command = gen_command(config)
    if fake_submit:
        print('fake submit')
    else:
        os.system(command)
    print(command)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_job(fake_submit=fake_submit)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()