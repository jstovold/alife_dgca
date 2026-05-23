import argparse
import numpy as np
from types import SimpleNamespace

from measure.tasks import narmax, santa_fe

from grow.runner import Runner
from grow.reservoir import get_seed

from evolve.fitness import FixedOutputFitness, TaskFitness, MetricFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA


def run_ga(run_id: int, args: SimpleNamespace) -> None:
    print(f"Starting GA run {run_id}...")

    conditions = {
        "max_size": args.max_size,
        "min_size": args.input_nodes + args.output_nodes + (10 if not args.order else args.order),
        "io_path": True
    }

    w_out = None
    if args.task:
        #fitness_fn = TaskFitness(
        fitness_fn = FixedOutputFitness(
            series=narmax if args.task == "narma" else santa_fe,
            conditions=conditions,
            verbose=False,
            order=args.order,
            fixed_series=True,
        )

        w_out = np.ones((1, args.output_nodes))
        weights = np.linspace(0,1, args.output_nodes + 1)[1:]    # evenly spaced
        w_out = w_out * weights

    elif args.metric:
        fitness_fn = MetricFitness(
            metric=args.metric,
            conditions=conditions,
            verbose=False,
        )
    else:
        raise ValueError("Either args.task or args.metric must be set.")

    reservoir = get_seed(args.input_nodes, args.output_nodes, args.n_states, fixed_out = True, w_out = w_out)
    model = EvolvableDGCA(n_states=reservoir.n_states, hidden_size=64, noise=args.noise)
    runner = Runner(max_steps=100, max_size=300)

    mga = ChromosomalMGA(
        popsize=args.pop_size,
        seed_graph=reservoir,
        model=model,
        runner=runner,
        fitness_fn=fitness_fn,
        mutate_rate=args.mutate_rate,
        cross_rate=args.cross_rate,
        cross_style=args.cross_style,
        run_id=run_id,
        n_trials=args.n_trials,
        output_dir=args.output_dir,          
        heavy_log=args.heavy_log,
        n_reps_if_noisy=args.n_reps_if_noisy
    )

    mga.run(progress=False)
    print(f"Completed GA run {run_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True, help="ID of the GA run")
    args_cli = parser.parse_args()

    args_dict = {
        "pop_size": 10,
        "mutate_rate": 0.02,
        "cross_rate": 0.5,
        "cross_style": "cols",
        "n_trials": 100,
        "input_nodes": 20,
        "output_nodes": 20,
        "noise": 0.0, #.05,
        "order": 10,
        "task": "narma",
        "max_size": 200,
        "metric": None,
        "n_states": 3,
        "output_dir": "results",
        "heavy_log": True,
        "n_reps_if_noisy": 5
    }

    args = SimpleNamespace(**args_dict, run_id=args_cli.run_id)
    run_ga(args.run_id, args)
