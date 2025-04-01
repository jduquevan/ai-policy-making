import os
import random

SEEDS = list(range(43, 53))
random.shuffle(SEEDS)

def gen_command(config):
    command = "sbatch sweep/run_job_v1_ayoub.slurm"
    for value in config.values():
        command += f" {value}"
    return command

def run_random_job(fake_submit: bool = True):
    hparams = {
        'ppo_epochs': [2],
        'use_aa': [True],
        'aa_gamma': [0.96],
        'aa_beta': [1],
        'entropy_coef': [0.05],
    }

    config = {'seed': SEEDS.pop()}    # Pop one unique seed each call
    for key, values in hparams.items():
        config[key] = random.choice(values)

    command = gen_command(config)
    if fake_submit:
        print('fake submit')
    else:
        os.system(command)
    print(command)

def main(num_jobs: int, fake_submit: bool = True):
    for _ in range(num_jobs):
        run_random_job(fake_submit=fake_submit)

if __name__ == '__main__':
    import fire
    fire.Fire()
    
    
    
# import os
# import random

# def gen_command(config):
#     command = "sbatch sweep/run_job_v1_ayoub.slurm"
#     for key, value in config.items():
#         command += " {}".format(value)
#     return command

# def run_random_job(fake_submit: bool = True):
#     hparams = {
#         'seed': [52],   
#         'ppo_epochs': [2],
#         'use_aa': [True],                           
#         'aa_gamma': [0.96],         
#         'aa_beta': [1.5],                  
#         'entropy_coef': [0.01],         
#     }

#     config = {}
#     for key, values in hparams.items():
#         if key == 'seed':
#             config[key] = random.choice(values)
#             values.remove(config[key])       # remove so it won’t repeat
#         else:
#             config[key] = random.choice(values)

#     command = gen_command(config)
#     if fake_submit:
#         print('fake submit')
#     else:
#         os.system(command)
#     print(command)

# def main(num_jobs: int, fake_submit: bool = True):
#     for i in range(num_jobs):
#         run_random_job(fake_submit=fake_submit)

# if __name__ == '__main__':
#     import fire
#     fire.Fire()
    
    
    
# import os
# import random

# def gen_command(config):
#     command = "sbatch sweep/run_job_v1_ayoub.slurm 43"
#     for key, value in config.items():
#         command += " {}".format(value)
#     return command


# # def run_random_job(fake_submit: bool = True):
# #     hparams = {
# #         'ppo_epochs': [1, 2, 3],
# #         'use_aa': [True],                           # Always True
# #         'aa_gamma': [0.99, 0.96, 0.9, 0.8],         # +++
# #         'aa_beta': [1.5],                  # +++
# #         # 'aa_beta': [0.5, 1, 1.5, 2, 3],                  # +++
# #         'entropy_coef': [0.01, 0.03, 0.05, 0.07],         # Important
# #     }


# def run_random_job(fake_submit: bool = True):
#     hparams = {
#         'ppo_epochs': [2],
#         'use_aa': [True],                           # Always True
#         'aa_gamma': [0.96],         # +++
#         'aa_beta': [1.5],                  # +++
#         # 'aa_beta': [0.5, 1, 1.5, 2, 3],                  # +++
#         'entropy_coef': [0.01],         # Important
#     }



#     # sample a random config
#     config = {}
#     for key, values in hparams.items():
#         config[key] = random.choice(values)

#     # submit this job using slurm
#     command = gen_command(config)
#     if fake_submit:
#         print('fake submit')
#     else:
#         os.system(command)
#     print(command)

# def main(num_jobs: int, fake_submit: bool = True):
#     for i in range(num_jobs):
#         run_random_job(fake_submit=fake_submit)

# if __name__ == '__main__':
#     # use fire to turn this into a command line tool
#     import fire
#     fire.Fire()