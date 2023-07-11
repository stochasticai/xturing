import argparse
import logging
import shlex
import sys
from functools import cached_property
from typing import Any, Mapping, Optional

import openai

from .base import EvalSpec, RunSpec
from .record import DummyRecorder, LocalRecorder, Recorder
from .registry import Registry, registry

logger = logging.getLogger(__name__)


def _purple(str):
    return f"\033[1;35m{str}\033[0m"


def run_evaluation(args):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    model = args.model

    run_config = {
        "model": model,
        "eval": args.eval_spec,
        "seed": args.seed,
    }

    model_name = model.config_name
    eval_name = args.eval_spec.key

    run_spec = RunSpec(
        model_name=model_name,
        eval_name=eval_name,
        base_eval=eval_name.split(".")[0],
        split=eval_name.split(".")[1],
        run_config=run_config,
        created_by=args.user,
        run_id="something",
    )
    if args.record_path is None:
        record_path = f"/tmp/evallogs/{run_spec.run_id}_{args.model}_{args.eval}.jsonl"
    else:
        record_path = args.record_path

    # Recording progress
    if args.dry_run:
        recorder = DummyRecorder(run_spec=run_spec, log=args.dry_run_logging)
    elif args.local_run:
        recorder = LocalRecorder(record_path, run_spec=run_spec)
    else:
        recorder = Recorder(record_path, run_spec=run_spec)

    api_extra_options = {}
    if not args.cache:
        api_extra_options["cache_level"] = 0

    run_url = f"{run_spec.run_id}"
    logger.info(_purple(f"Run started: {run_url}"))

    def parse_extra_eval_params(param_str: Optional[str]) -> Mapping[str, Any]:
        """Parse a string of the form "key1=value1,key2=value2" into a dict."""
        if not param_str:
            return {}

        def to_number(x):
            try:
                return int(x)
            except:
                pass
            try:
                return float(x)
            except:
                pass
            return x

        str_dict = dict(kv.split("=") for kv in param_str.split(","))
        return {k: to_number(v) for k, v in str_dict.items()}

    extra_eval_params = parse_extra_eval_params(args.extra_eval_params)

    eval_class = registry.get_class(args.eval_spec)
    eval = eval_class(
        model_specs=model,
        seed=args.seed,
        name=eval_name,
        registry=registry,
        **extra_eval_params,
    )
    result = eval.run(recorder)
    recorder.record_final_report(result)

    if not (args.dry_run or args.local_run):
        logger.info(_purple(f"Run completed: {run_url}"))

    logger.info("Final report:")
    for key, value in result.items():
        logger.info(f"{key}: {value}")
    return run_spec.run_id


def evaluate(
    model: str,
    eval: str,
    embedding_model: str = "",
    ranking_model: str = "",
    extra_eval_params: str = "",
    max_samples: Optional[int] = None,
    cache: bool = True,
    visible: Optional[bool] = None,
    seed: int = 20220722,
    user: str = "",
    record_path: Optional[str] = None,
    log_to_file: Optional[str] = None,
    debug: bool = False,
    local_run: bool = True,
    dry_run: bool = False,
    dry_run_logging: bool = True,
) -> Any:
    parser = argparse.ArgumentParser(description="Run evals through the API")
    parser.add_argument("model", type=str, help="Name of a completion model.")
    parser.add_argument("eval", type=str, help="Name of an eval. See registry.")
    parser.add_argument("--embedding_model", type=str, default="")
    parser.add_argument("--ranking_model", type=str, default="")
    parser.add_argument("--extra_eval_params", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--visible", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument("--user", type=str, default="")
    parser.add_argument("--record_path", type=str, default=None)
    parser.add_argument(
        "--log_to_file", type=str, default=None, help="Log to a file instead of stdout"
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--local-run", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--dry-run-logging", action=argparse.BooleanOptionalAction, default=True
    )

    args = argparse.Namespace(
        model=model,
        eval=eval,
        embedding_model=embedding_model,
        ranking_model=ranking_model,
        extra_eval_params=extra_eval_params,
        max_samples=max_samples,
        cache=cache,
        visible=visible,
        seed=seed,
        user=user,
        record_path=record_path,
        log_to_file=log_to_file,
        debug=debug,
        local_run=local_run,
        dry_run=dry_run,
        dry_run_logging=dry_run_logging,
    )

    # args_parsed = parser.parse_args()

    # Running evaluation code
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        filename=args.log_to_file if args.log_to_file else None,
    )

    logging.getLogger("openai").setLevel(logging.WARN)
    if hasattr(openai.error, "set_display_cause"):
        openai.error.set_display_cause()

    run_evaluation(args)


####################################
# EXAMPLE USAGE:

# evaluate(
#     model_name="davinci",
#     eval="test",
#     embedding_model="",
#     ranking_model="",
#     extra_eval_params="",
#     max_samples=None,
# )
