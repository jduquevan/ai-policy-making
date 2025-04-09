import os
import glob
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import matplotlib.pyplot as plt

from jax_pbt.controller.ippo_controller import IPPOController
from jax_pbt.env.investesg.investesg_env import InvestESGEnv
from jax_pbt.model.models import MLPConfig
from jax_pbt.config import get_base_parser, generate_parameters
from jax_pbt.buffer.ppo_buffer import PPOAgentState
from jax_pbt.utils import RoleIndex, rng_batch_split, select_env_agent

######################################
# 1) Forced actions
######################################
def always_defect_action():
    """Defector takes zero action in all of mitigation, greenwashing, and resilience. 3D action: (0,0,0.0)."""
    return jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

def always_cooperate_action():
    """Cooperative policies consistently invest 0.5% of their capital in mitigation only. 3D action: (0.005, 0, 0)."""
    return jnp.array([0.005, 0.0, 0.0], dtype=jnp.float32)

######################################
# 2) Extract final capitals
######################################
def extract_company_capitals_from_obs(all_obs, num_companies=5):
    """
    'all_obs' is a dict: {'obs' -> jnp.array([num_envs, num_agent, obs_dim])}.
    We pick the first agent's array (identical for all), then parse out
    each company's capital.
    Each company c has a 4-feature block: [capital, resilience, esg_score, margin].
    """
    first_agent_key = next(iter(all_obs.keys()))
    obs_mat = all_obs[first_agent_key]  # shape [num_envs, num_agent, obs_dim]

    # all agent observation should be the same, so take the first one
    caps_list = []
    for c in range(num_companies):
        idx = c * 4
        caps_list.append(obs_mat[:, 0, idx])
    return jnp.stack(caps_list, axis=1)  # shape [num_envs, num_companies]

######################################
# 3) Extract market total wealth
######################################
def extract_market_total_wealth_from_obs(all_obs, num_companies=5):
    """
    'all_obs' is a dict: {'obs' -> jnp.array([num_envs, num_agent, obs_dim])}.
    We pick the first agent's array (identical for all), then parse out
    each company's capital.
    Each company c has a 4-feature block: [capital, resilience, esg_score, margin].
    """
    first_agent_key = next(iter(all_obs.keys()))
    obs_mat = all_obs[first_agent_key]  # shape [num_envs, num_agent, obs_dim]

    # all agent observation should be the same, so take the first one
    caps_list = []
    for c in range(num_companies):
        idx = c * 4
        caps_list.append(obs_mat[:, 0, idx])
    return jnp.stack(caps_list, axis=1)  # shape [num_envs, num_companies]

######################################
# 4) Overwrite checkpoint parameters only
######################################
def update_resume_runner_state_from_checkpoint(ckpt_dict, resume_runner_state):
    """
    Overwrites just the actor/critic params in 'resume_runner_state'
    from the loaded 'ckpt_dict'. Ignores optimizer states.
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

######################################
# 5) Plotting
######################################
def plot_and_save(x, defector_means, defector_stds, cooperator_means, cooperator_stds,
                  xlabel, ylabel, title, filename):
    """
    Plots error bars for defector and cooperator metrics and saves the figure.

    Parameters:
      x: Array-like, x-axis values.
      defector_means: Array-like, mean values for the defector metric.
      defector_stds: Array-like, standard deviations for the defector metric.
      cooperator_means: Array-like, mean values for the cooperator metric.
      cooperator_stds: Array-like, standard deviations for the cooperator metric.
      xlabel: String, label for the x-axis.
      ylabel: String, label for the y-axis.
      title: String, title for the plot.
      filename: String, output file name to save the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x, defector_means, yerr=defector_stds,
        marker='o', label='Defector', capsize=5
    )
    plt.errorbar(
        x, cooperator_means, yerr=cooperator_stds,
        marker='s', label='Cooperator', capsize=5
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

######################################
# 6) Roll out scenario
######################################
def run_scenario(
    controller,
    runner_state,
    num_steps: int,
    forced_indices: list[int],
    forced_action: str = "defect"
):
    """
    forced_indices: list of company indices to force with a specific action.
    forced_action: "defect" or "cooperate"
    All other companies & investors use the loaded policy.
    Returns the final company capitals: shape [num_envs, 5].
    """
    def pick_forced_action():
        if forced_action.lower() == "defect":
            return always_defect_action()
        else:
            return always_cooperate_action()

    env_const = controller.env_fn.get_default_const()
    rng, train_state_lst, agent_state_lst, env_state, obs = runner_state

    for t in range(num_steps):
        all_actions = {}
        for i, (agent_name, agent_fn, train_state, agent_state, agent_roles) in enumerate(
            zip(
                controller.env_fn.agent_lst,
                controller.agent_fn_lst,
                train_state_lst,
                agent_state_lst,
                controller.role_index_lst
            )
        ):
            if not agent_name.startswith("company_"):
                # Investor => loaded policy
                obs_i = select_env_agent(obs, agent_roles)
                rng, rng_action = rng_batch_split(rng, len(agent_roles))
                next_agent_state, action, log_p, val = jax.vmap(
                    agent_fn.rollout_step, in_axes=(0, None, 0, 0)
                )(rng_action, train_state, agent_state, obs_i)
                agent_state_lst[i] = next_agent_state
                all_actions[agent_name] = action
                continue

            company_idx = int(agent_name.split("_")[1])
            if company_idx in forced_indices:
                # Forced action for this company.
                forced_a = pick_forced_action()
                a = jnp.vstack([forced_a for _ in range(controller.num_envs)])
                all_actions[agent_name] = a
            else:
                # Loaded policy action.
                obs_i = select_env_agent(obs, agent_roles)
                rng, rng_action = rng_batch_split(rng, len(agent_roles))
                next_agent_state, action, log_p, val = jax.vmap(
                    agent_fn.rollout_step, in_axes=(0, None, 0, 0)
                )(rng_action, train_state, agent_state, obs_i)
                agent_state_lst[i] = next_agent_state

                action_space = controller.env_fn.get_action_space(agent_name)
                clipped = (action + 1.) / 2. * action_space.high
                clipped = jnp.clip(clipped, action_space.low, action_space.high)
                all_actions[agent_name] = clipped

        rng, rng_env_step = rng_batch_split(rng, controller.num_envs)
        env_state, obs, rew, done, info = jax.vmap(
            controller.env_fn.step, in_axes=(0, None, 0, 0)
        )(rng_env_step, env_const, env_state, all_actions)

    final_company_capitals = extract_company_capitals_from_obs(obs, 5)
    new_runner_state = (rng, train_state_lst, agent_state_lst, env_state, obs)
    return new_runner_state, final_company_capitals

######################################
# 7) Main
######################################
def main():
    parser = get_base_parser()
    args = parser.parse_args()

    # Example dictionary mapping from “friendly” key => checkpoint folder name.
    # You can rename these keys and values as you wish, so long as the folder names
    # match the directories where your checkpoints are stored.
    run_ids_dict = {
        "ppo_42_esg_score_0_obs_True": "ppo_42_esg_score_0_obs_True",
        "ppo_43_esg_score_0_obs_True": "ppo_43_esg_score_0_obs_True",
        "ppo_44_esg_score_0_obs_True": "ppo_44_esg_score_0_obs_True",
        "ppo_45_esg_score_0_obs_True": "ppo_45_esg_score_0_obs_True",
        "ppo_46_esg_score_0_obs_True": "ppo_46_esg_score_0_obs_True",
        "ppo_47_esg_score_0_obs_True": "ppo_47_esg_score_0_obs_True",
        "ppo_48_esg_score_0_obs_True": "ppo_48_esg_score_0_obs_True",
        "ppo_49_esg_score_0_obs_True": "ppo_49_esg_score_0_obs_True",
        "ppo_50_esg_score_0_obs_True": "ppo_50_esg_score_0_obs_True",
        "ppo_51_esg_score_0_obs_True": "ppo_51_esg_score_0_obs_True"
    }

    # Setup configuration: always 5 companies.
    config = generate_parameters(
        domain=args.env_config_name,
        debug=args.debug,
        wandb_project="InvestESG",
        config_from_arg=vars(args)
    )
    config.update({"num_companies": 5}, allow_val_change=True)

    # We'll collect payoff data across seeds in these lists.
    defector_payoffs_all = []  # shape: (num_seeds, len(forced_counts))
    cooperator_payoffs_all = []  # shape: (num_seeds, len(forced_counts))

    defector_mtw_all = []
    cooperator_mtw_all = []

    forced_counts = [5, 4, 3, 2, 1]  # We'll forcibly defect or cooperate for these many companies
    # This is the x-axis (number of "trained" companies that do not get forced).
    trained_companies = [0, 1, 2, 3, 4]  # for the plot
    num_steps = args.episode_length

    # We iterate over each seed/folder in the dictionary:
    for friendly_name, checkpoint_folder in run_ids_dict.items():
        print(f"\n--- Processing run_id: {checkpoint_folder} ({friendly_name}) ---")

        # Create a fresh environment & controller for each run_id.
        env_fn = InvestESGEnv(config=config)
        model_config = {'obs': MLPConfig(hidden_layers=2, hidden_size=256)}
        controller = IPPOController([args], env_fn=env_fn, model_config_lst=[model_config])

        # Build role_index for each agent.
        rng = jax.random.PRNGKey(config.seed)
        agent_roles_lst = []
        for ag_idx in range(env_fn.num_agents):
            arr = [(env_idx, ag_idx) for env_idx in range(args.num_envs)]
            agent_roles_lst.append(arr)

        role_index_lst = [
            RoleIndex(
                jnp.array(r, dtype=int)[:, 0],
                jnp.array(r, dtype=int)[:, 1],
            )
            for r in agent_roles_lst
        ]
        controller.role_index_lst = role_index_lst

        # Initialize the runner state.
        init_runner_state = controller.init_runner_state(rng, env_fn.get_default_const(), role_index_lst)

        # Load checkpoint for the current run_id.
        save_dir = os.path.join(args.save_directory, checkpoint_folder)
        ckpt_files = glob.glob(os.path.join(save_dir, "checkpoint_*"))
        if ckpt_files:
            init_runner_state = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=init_runner_state)
            ckpt_dict = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=None)
            init_runner_state = update_resume_runner_state_from_checkpoint(ckpt_dict, init_runner_state)
            print(f"Loaded checkpoint for run_id={checkpoint_folder}")
        else:
            print(f"No checkpoint found for {checkpoint_folder}; using random init.")

        # For each forced_count, compute defector and cooperator payoffs.
        defector_payoffs_run = []
        cooperator_payoffs_run = []

        defector_mtw = []
        cooperator_mtw = []

        for n in forced_counts:
            # Reset to the initial runner state for each scenario.
            runner_state = init_runner_state

            all_companies = list(range(5))
            forced_indices = all_companies[-n:]  # e.g., n=5 => [0,1,2,3,4]; n=1 => [4]

            # Scenario with forced defect.
            defector_runner_state, final_caps_def = run_scenario(
                controller=controller,
                runner_state=runner_state,
                num_steps=num_steps,
                forced_indices=forced_indices,
                forced_action="defect"
            )
            defector_mtw.append(jnp.mean(defector_runner_state[3]._state.history_market_total_wealth[:, -1]))
            def_caps = final_caps_def[:, forced_indices]
            payoff_def = float(def_caps.mean())
            defector_payoffs_run.append(payoff_def)
            print(f"Run {checkpoint_folder} - Forced defect with n={n}: payoff = {payoff_def:.3f}")

            # Scenario with forced cooperate.
            runner_state = init_runner_state  # reset again
            cooperator_runner_state, final_caps_coop = run_scenario(
                controller=controller,
                runner_state=runner_state,
                num_steps=num_steps,
                forced_indices=forced_indices,
                forced_action="cooperate"
            )
            cooperator_mtw.append(jnp.mean(cooperator_runner_state[3]._state.history_market_total_wealth[:, -1]))
            coop_caps = final_caps_coop[:, forced_indices]
            payoff_coop = float(coop_caps.mean())
            cooperator_payoffs_run.append(payoff_coop)
            print(f"Run {checkpoint_folder} - Forced cooperate with n={n}: payoff = {payoff_coop:.3f}")

        # Store this run's results.
        defector_payoffs_all.append(defector_payoffs_run)
        cooperator_payoffs_all.append(cooperator_payoffs_run)
        defector_mtw_all.append(defector_mtw)
        cooperator_mtw_all.append(cooperator_mtw)
        print(f"Transferred results for run {checkpoint_folder} into defector_payoffs_all and cooperator_payoffs_all.")

    # Print the complete lists before aggregation.
    print("\nAll defector_payoffs_all:")
    print(defector_payoffs_all)
    print("\nAll cooperator_payoffs_all:")
    print(cooperator_payoffs_all)

    # Aggregating results: computing mean and standard deviation across seeds.
    print("\nComputing aggregated means and standard deviations across all seeds...")
    defector_means = np.mean(defector_payoffs_all, axis=0)
    defector_stds = np.std(defector_payoffs_all, axis=0)
    cooperator_means = np.mean(cooperator_payoffs_all, axis=0)
    cooperator_stds = np.std(cooperator_payoffs_all, axis=0)

    defector_mtw_means = np.mean(defector_mtw_all, axis=0)
    defector_mtw_stds = np.std(defector_mtw_all, axis=0)
    cooperator_mtw_means = np.mean(cooperator_mtw_all, axis=0)
    cooperator_mtw_stds = np.std(cooperator_mtw_all, axis=0)

    print("\nFinal aggregated results across all seeds:")
    for i, n in enumerate(forced_counts):
        print(f"[n={n}] Defector => mean={defector_means[i]:.3f}, std={defector_stds[i]:.3f} | "
              f"Cooperator => mean={cooperator_means[i]:.3f}, std={cooperator_stds[i]:.3f}")

    plot_and_save(
        trained_companies,
        defector_means,
        defector_stds,
        cooperator_means,
        cooperator_stds,
        "Number of Trained Companies (i.e., not forced)",
        "Individual Ending Capital (avg)",
        "ESG pref=0 Baseline: Mean ± 1 STD over multiple seeds",
        "evaluation_plot_esg_0.png"
    )

    plot_and_save(
        trained_companies,
        defector_mtw_means,
        defector_mtw_stds,
        cooperator_mtw_means,
        cooperator_mtw_stds,
        "Number of Trained Companies (i.e., not forced)",
        "Market Total Wealth (avg)",
        "MTW: Mean ± 1 STD over multiple seeds",
        "evaluation_plot_mtw.png"
    )

if __name__ == "__main__":
    main()

# command: python scripts/evaluation_v2.py --episode_length 100 --num_env 64 --env_config_name exp_default_1 --debug True 