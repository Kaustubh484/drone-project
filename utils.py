from ppo import PPOAgent
from sac import SACAgent
from ddpg import DDPGAgent

def initialize_agent(args, state_dim, action_dim, max_action):
    if args.algo == "ddpg":
        return DDPGAgent(
            state_dim, action_dim, max_action,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            buffer_size=args.buffer_size, batch_size=args.batch_size
        )

    elif args.algo == "ppo":
        return PPOAgent(
            state_dim, action_dim, max_action,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            n_epochs=args.ppo_epochs, clip_epsilon=args.ppo_clip,
            batch_size=args.batch_size
        )

    elif args.algo == "sac":
        return SACAgent(
            state_dim, action_dim, max_action,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            buffer_size=args.buffer_size, batch_size=args.batch_size
        )

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")