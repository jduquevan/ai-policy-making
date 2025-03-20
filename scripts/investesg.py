import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
from flax.training import checkpoints

def tree_allclose(tree1, tree2, atol=1e-6, rtol=1e-6):
    flat1, _ = jax.tree_util.tree_flatten(tree1)
    flat2, _ = jax.tree_util.tree_flatten(tree2)
    for a, b in zip(flat1, flat2):
        if a.shape != b.shape:
            print("Shape mismatch:", a.shape, b.shape)
            return False
        if not jnp.allclose(a, b, atol=atol, rtol=rtol):
            return False
    return True

def compare_checkpoint_and_resume_weights(ckpt_dict, resume_runner_state):
    """
    Compare model parameters (weights) in the checkpoint with those in the resumed runner state.
    resume_runner_state has structure:
      (rng, train_state_lst, agent_state_lst, env_state, all_obs)
    ckpt_dict['train_state'] is a dict with keys '0', '1', ...; for each agent, key '0' holds
    the actor checkpoint and key '1' holds the critic checkpoint.
    """
    _, resume_train_state, _, _, _ = resume_runner_state
    ckpt_train_state = ckpt_dict['train_state']

    for i in range(len(resume_train_state)):
        resume_actor, resume_critic = resume_train_state[i]
        ckpt_actor = ckpt_train_state[str(i)]['0']  # actor checkpoint
        ckpt_critic = ckpt_train_state[str(i)]['1']   # critic checkpoint

        actor_equal = tree_allclose(resume_actor.params, ckpt_actor['params'])
        critic_equal = tree_allclose(resume_critic.params, ckpt_critic['params'])

        print(f"Agent {i}:")
        print(f"  Actor policy layer keys: {list(resume_actor.params['params']['policy_layer'].keys())}")
        print(f"  Actor weights equal: {actor_equal}")
        print(f"  Critic weights equal: {critic_equal}")

def update_resume_runner_state_from_checkpoint(ckpt_dict, resume_runner_state):
    """
    Update the resumed runner state so that model parameters exactly match those saved in the checkpoint.
    This function updates only the parameters (not the optimizer state).
    
    Expected structure:
      resume_runner_state = (rng, train_state_lst, agent_state_lst, env_state, all_obs)
      where train_state_lst is a list of tuples: (actor_train_state, critic_train_state)
      
      ckpt_dict['train_state'] is a dict with string keys '0', '1', ... where each agent's value is:
          { '0': actor checkpoint (with key 'params'),
            '1': critic checkpoint (with key 'params') }
    """
    rng, resume_train_state, agent_state_lst, env_state, all_obs = resume_runner_state
    new_train_state_lst = []
    for i, (actor_resume, critic_resume) in enumerate(resume_train_state):
        ckpt_agent = ckpt_dict['train_state'][str(i)]
        ckpt_actor_params = ckpt_agent['0']['params']
        ckpt_critic_params = ckpt_agent['1']['params']
        
        actor_updated = actor_resume.replace(params=ckpt_actor_params)
        critic_updated = critic_resume.replace(params=ckpt_critic_params)
        
        new_train_state_lst.append((actor_updated, critic_updated))
    
    return (rng, new_train_state_lst, agent_state_lst, env_state, all_obs)


if __name__ == '__main__':
    from jax_pbt.controller.ippo_controller import IPPOController
    from jax_pbt.config import get_base_parser, generate_parameters
    from jax_pbt.buffer.ppo_buffer import PPOAgentState
    import numpy as np
    import os, glob
    from flax.training import checkpoints
    import jax

    parser = get_base_parser()
    args = parser.parse_args()
    print(args)
    config = generate_parameters(
        domain=args.env_config_name,
        debug=args.debug,
        wandb_project="InvestESG",
        config_from_arg=vars(args)
    )
    print(config)
    
    from jax_pbt.env.investesg.investesg_env import InvestESGConst, InvestESGEnv
    env_fn = InvestESGEnv(config=config)
    from jax_pbt.model.models import MLPConfig
    model_config = {'obs': MLPConfig(hidden_layers=2, hidden_size=256)}

    # Instantiate the controller.
    controller = IPPOController([args], env_fn=env_fn, model_config_lst=[model_config])
    
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    rng = jax.random.PRNGKey(config.seed)
    if args.eval_points > 0:
        eval_at_steps = list((np.arange(args.eval_points + 1) * args.total_env_steps / args.eval_points).astype(int))
    else:
        eval_at_steps = list(np.arange(args.total_env_steps)[::args.eval_step_interval]) + [args.total_env_steps]
    
    # Compute agent roles.
    agent_roles_lst = []
    for i in range(env_fn.num_agents):
        agent_roles = []
        for j in range(args.num_envs):
            agent_roles.append((j, i))
        agent_roles_lst.append(agent_roles)
    from jax_pbt.utils import RoleIndex
    agent_roles_lst = [
        RoleIndex(jax.numpy.array(x, dtype=int)[:, 0], jax.numpy.array(x, dtype=int)[:, 1])
        for x in agent_roles_lst
    ]
    
    # --- Check for existing checkpoints ---
    save_dir = os.path.join(args.save_directory, args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_*"))
    
    env_const = env_fn.get_default_const()
    
    if checkpoint_files:
        print("Checkpoint found, loading the latest checkpoint...")
        target = controller.init_runner_state(rng, env_const, agent_roles_lst)
        resume_runner_state = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=target)
        ckpt_dict = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=None)
        
        # Update the resumed runner state so that parameters match the checkpoint.
        resume_runner_state = update_resume_runner_state_from_checkpoint(ckpt_dict, resume_runner_state)
        compare_checkpoint_and_resume_weights(ckpt_dict, resume_runner_state)
    else:
        print("No checkpoint found, starting training from scratch.")
        resume_runner_state = None
    
    runner_state = controller.run(
        rng,
        agent_roles_lst,
        get_epoch_env_const=lambda **kwargs: env_const,
        eval_at_steps=eval_at_steps,
        resume_runner_state=resume_runner_state
    )