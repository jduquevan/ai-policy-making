if __name__ == '__main__':
    from jax_pbt.controller.ippo_controller import IPPOController
    from jax_pbt.config import get_base_parser, generate_parameters
    import numpy as np
    import wandb
    import os, glob
    from flax.training import checkpoints
    import jax

    parser = get_base_parser()
    args = parser.parse_args()
    print(args)
    config = generate_parameters(domain=args.env_config_name, debug=args.debug, wandb_project="InvestESG", config_from_arg=vars(args))
    print(config)
    from jax_pbt.env.investesg.investesg_env import InvestESGConst, InvestESGEnv
    env_fn = InvestESGEnv(config=config)
    from jax_pbt.model.models import MLPConfig
    model_config = {'obs': MLPConfig(hidden_layers=2, hidden_size=256)}
    
    def get_epoch_env_const(env_fn: InvestESGEnv, current_env_step: int, **kwargs) -> InvestESGConst:
        return InvestESGConst(
            max_steps=env_fn._env.max_steps,
            shaped_reward_factor=0.0 if env_fn.reward_shaping_steps is None else 1.0 - current_env_step / env_fn.reward_shaping_steps
        )

    # --- Check for existing checkpoints ---
    save_dir = os.path.join(args.save_directory, args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_*"))
    if checkpoint_files:
        print("Checkpoint found, loading the latest checkpoint...")
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=None)
        resume_runner_state = (
            ckpt['rng'],  # Use the stored PRNG key directly.
            ckpt['train_state'],
            ckpt['agent_state'],
            ckpt['env_state'],
            ckpt['all_obs']
        )
    else:
        print("No checkpoint found, starting training from scratch.")
        resume_runner_state = None

    controller = IPPOController([args], env_fn=env_fn, model_config_lst=[model_config])
    if args.debug:
        import jax
        jax.config.update("jax_disable_jit", True)
    rng = jax.random.PRNGKey(config.seed)
    if args.eval_points > 0:
        eval_at_steps = list((np.arange(args.eval_points + 1) * args.total_env_steps / args.eval_points).astype(int))
    else:
        eval_at_steps = list(np.arange(args.total_env_steps)[::args.eval_step_interval]) + [args.total_env_steps]
    agent_roles_lst = []
    for i in range(env_fn.num_agents):
        agent_roles = []
        for j in range(args.num_envs):
            agent_roles.append((j, i))
        agent_roles_lst.append(agent_roles)
    from jax_pbt.utils import RoleIndex
    agent_roles_lst = [
        RoleIndex(jax.numpy.array(x, dtype=int)[:, 0], jax.numpy.array(x, dtype=int)[:, 1]) for x in agent_roles_lst
    ]
    
    # Pass the resume state (if any) to the controller
    runner_state = controller.run(
        rng,
        agent_roles_lst,
        get_epoch_env_const=get_epoch_env_const,
        eval_at_steps=eval_at_steps,
        resume_runner_state=resume_runner_state  # New parameter for resuming
    )

    # (Optionally, if you need to save a final checkpoint or arguments, do so here)
    import pickle
    with open(os.path.join(save_dir, "args.pkl"), "wb") as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)