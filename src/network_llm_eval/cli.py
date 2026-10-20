"""CLI for the versioned networking-domain LLM benchmark."""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .contracts import load_manifest, sha256, validate_manifest
from .control import (start as start_worker, status as worker_status, stop as stop_worker,
                      start_judge_all, judge_all_status, stop_judge_all,
                      start_pipeline, pipeline_status, stop_pipeline)
from .env import load_judge_env
from .generation import run, run_openai_compatible
from .judging import judge
from .plan import load_run_config, resolve_target
from .reporting import build_comparison_report
from .scoring import aggregate

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "list-routes"):
        item = commands.add_parser(name); item.add_argument("manifest", type=Path)
    item = commands.add_parser("run")
    item.add_argument("--manifest", type=Path, required=True); item.add_argument("--prompts", type=Path, required=True); item.add_argument("--model", required=True); item.add_argument("--model-revision", help="immutable model revision or local artifact SHA-256"); item.add_argument("--adapter"); item.add_argument("--adapter-revision", help="adapter artifact SHA-256 or immutable revision"); item.add_argument("--device", default="cuda:0"); item.add_argument("--output", type=Path, required=True); item.add_argument("--max-new-tokens", type=int); item.add_argument("--limit-per-route", type=int); item.add_argument("--routes", help="comma-separated subset of manifest routes"); item.add_argument("--resume", action="store_true"); item.add_argument("--local-files-only", action="store_true"); item.add_argument("--trust-remote-code", action="store_true")
    item = commands.add_parser("run-config", help="run a named target from a versioned evaluation config")
    item.add_argument("--config", type=Path, required=True); item.add_argument("--target", required=True)
    item.add_argument("--limit-per-route", type=int); item.add_argument("--output-suffix", help="append a child directory below the configured target output")
    item.add_argument("--resume", action="store_true"); item.add_argument("--dry-run", action="store_true")
    item.add_argument("--device", help=argparse.SUPPRESS)  # coordinator-owned runtime placement
    # Internal coordinator entrypoint.  Users invoke the stable ``start-all``
    # command; it launches this process in the background.
    item = commands.add_parser("pipeline-run", help=argparse.SUPPRESS)
    item.add_argument("--config", type=Path, required=True); item.add_argument("--targets", required=True)
    item.add_argument("--limit-per-route", type=int); item.add_argument("--resume", action="store_true")
    for name in ("start", "status", "stop"):
        item = commands.add_parser(name, help=f"{name} a managed config-driven evaluation worker")
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--target", required=True)
        if name == "start":
            item.add_argument("--limit-per-route", type=int)
            item.add_argument("--resume", action="store_true")
    for name in ("start-all", "status-all", "stop-all"):
        item = commands.add_parser(name, help=f"{name} configured managed evaluation workers")
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--targets", help="comma-separated target subset; defaults to every configured target")
        if name == "start-all":
            item.add_argument("--limit-per-route", type=int)
            item.add_argument("--resume", action="store_true")
    item = commands.add_parser("tui", help="read-only terminal monitor for config-driven evaluation evidence")
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--targets", help="comma-separated targets; defaults to every configured target")
    item.add_argument("--refresh-seconds", type=float, default=2.0)
    item = commands.add_parser("judge", help="run the configured LLM judge over frozen generation evidence")
    item.add_argument("--config", type=Path, required=True); item.add_argument("--target", required=True)
    item.add_argument("--env-file", type=Path, help="local .env; defaults to config directory/.env")
    item.add_argument("--batch-size", type=int, default=5); item.add_argument("--retries", type=int, default=3)
    item.add_argument("--resume", action="store_true"); item.add_argument("--allow-incomplete", action="store_true")
    item.add_argument("--max-items", type=int); item.add_argument("--dry-run", action="store_true"); item.add_argument("--routes", help="comma-separated completed route subset")
    item = commands.add_parser("report", help="write CPT/MOPD/routed-teacher comparison reports after judging")
    item.add_argument("--config", type=Path, required=True)
    item = commands.add_parser("judge-all", help="judge every configured target under one frozen protocol")
    item.add_argument("--config", type=Path, required=True); item.add_argument("--targets", help="comma-separated target subset")
    item.add_argument("--env-file", type=Path, help="local .env; defaults to config directory/.env")
    item.add_argument("--batch-size", type=int, default=5); item.add_argument("--retries", type=int, default=3)
    item.add_argument("--parallelism", type=int, default=4, help="maximum independent target judge workers (default: 4)")
    item.add_argument("--resume", action="store_true"); item.add_argument("--allow-incomplete", action="store_true")
    item.add_argument("--max-items", type=int); item.add_argument("--dry-run", action="store_true")
    for name in ("judge-all-start", "judge-all-status", "judge-all-stop"):
        item = commands.add_parser(name, help=f"{name} managed all-target LLM judging")
        item.add_argument("--config", type=Path, required=True)
        if name == "judge-all-start":
            item.add_argument("--batch-size", type=int, default=5); item.add_argument("--retries", type=int, default=3)
            item.add_argument("--parallelism", type=int, default=4, help="maximum independent target judge workers (default: 4)")
            item.add_argument("--resume", action="store_true")
    item = commands.add_parser("score"); item.add_argument("--generations", type=Path, required=True); item.add_argument("--judgments", type=Path, required=True); item.add_argument("--output", type=Path, required=True); item.add_argument("--judge-protocol", type=Path, help="versioned rubric/judge-protocol document used to create judgments"); item.add_argument("--judge-prompt", type=Path, help="exact full judge prompt template, if separate from --judge-protocol"); item.add_argument("--env-file", type=Path, help="untracked .env containing judge identity metadata"); item.add_argument("--judge-provider"); item.add_argument("--judge-model"); item.add_argument("--judge-model-revision", help="immutable provider/model revision when available")
    args = parser.parse_args()
    if args.command == "validate":
        summary = validate_manifest(args.manifest); print(f"valid: {summary['routes']} routes, {summary['prompts']} prompts, manifest={sha256(args.manifest)}")
    elif args.command == "list-routes":
        for entry in load_manifest(args.manifest)["routes"]: print(f"{entry['route']}\t{entry['records']}\t{entry['path']}")
    elif args.command == "run":
        routes = args.routes.split(",") if args.routes else None
        print(json.dumps(run(manifest=args.manifest.resolve(), prompts=args.prompts.resolve(), model_path=args.model, adapter=args.adapter, model_revision=args.model_revision, adapter_revision=args.adapter_revision, device=args.device, output=args.output.resolve(), max_new_tokens=args.max_new_tokens, limit_per_route=args.limit_per_route, resume=args.resume, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code, routes=routes), sort_keys=True))
    elif args.command == "run-config":
        config = load_run_config(args.config.resolve())
        plan = resolve_target(config, args.config.resolve(), args.target)
        limit_per_route = args.limit_per_route
        if limit_per_route is None:
            configured_limit = config.get("benchmark", {}).get("limit_per_route")
            limit_per_route = int(configured_limit) if configured_limit is not None else None
        if args.device:
            plan["device"] = args.device
        if args.output_suffix:
            plan["output"] = plan["output"] / args.output_suffix
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        else:
            runner = run_openai_compatible if plan.get("backend") == "openai_compatible" else run
            print(json.dumps(runner(**plan, limit_per_route=limit_per_route, resume=args.resume), sort_keys=True))
    elif args.command == "pipeline-run":
        from .pipeline import run_pipeline
        config_path = args.config.resolve()
        config = load_run_config(config_path)
        targets = args.targets.split(",")
        unknown = sorted(set(targets) - set(config["targets"]))
        if unknown:
            parser.error(f"unknown pipeline target(s): {', '.join(unknown)}")
        print(json.dumps(run_pipeline(config, config_path, targets=targets, limit_per_route=args.limit_per_route, resume=args.resume), indent=2, sort_keys=True))
    elif args.command in {"start", "status", "stop", "start-all", "status-all", "stop-all"}:
        config_path = args.config.resolve()
        config = load_run_config(config_path)
        all_targets = args.command.endswith("-all")
        targets = args.targets.split(",") if all_targets and args.targets else ([args.target] if not all_targets else list(config["targets"]))
        unknown = sorted(set(targets) - set(config["targets"]))
        if unknown:
            parser.error(f"unknown target(s): {', '.join(unknown)}")
        if args.command == "start-all":
            # Validate every requested artifact before launching any worker.
            plans = {target: resolve_target(config, config_path, target) for target in targets}
            # Remote endpoint generations consume no local GPU.  Do not send
            # them through the GPU-admission pipeline, which would wait for
            # unrelated training jobs to release accelerator memory.
            if all(plan.get("backend") == "openai_compatible" for plan in plans.values()):
                limit = args.limit_per_route
                if limit is None:
                    configured = config.get("benchmark", {}).get("limit_per_route")
                    limit = int(configured) if configured is not None else None
                results = {target: start_worker(config, config_path, target, limit_per_route=limit, resume=args.resume) for target in targets}
                print(json.dumps(results, indent=2, sort_keys=True, default=str))
                return
            print(json.dumps(start_pipeline(config, config_path, targets=targets, limit_per_route=args.limit_per_route, resume=args.resume), indent=2, sort_keys=True, default=str))
            return
        if args.command == "status-all":
            result = {
                "pipeline": pipeline_status(config, config_path),
                "targets": {target: worker_status(config, config_path, target) for target in targets},
                "judge": judge_all_status(config, config_path),
            }
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return
        if args.command == "stop-all":
            result = stop_pipeline(config, config_path)
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return
        results = {}
        limit_per_route = args.limit_per_route
        if limit_per_route is None:
            configured_limit = config.get("benchmark", {}).get("limit_per_route")
            limit_per_route = int(configured_limit) if configured_limit is not None else None
        for target in targets:
            if args.command in {"start", "start-all"}:
                results[target] = start_worker(config, config_path, target, limit_per_route=limit_per_route, resume=args.resume)
            elif args.command in {"stop", "stop-all"}:
                results[target] = stop_worker(config, config_path, target)
            else:
                results[target] = worker_status(config, config_path, target)
        print(json.dumps(results if all_targets else results[targets[0]], indent=2, sort_keys=True, default=str))
    elif args.command == "tui":
        from .tui import monitor
        config_path = args.config.resolve()
        config = load_run_config(config_path)
        targets = args.targets.split(",") if args.targets else list(config["targets"])
        unknown = sorted(set(targets) - set(config["targets"]))
        if unknown:
            parser.error(f"unknown TUI target(s): {', '.join(unknown)}")
        if args.refresh_seconds <= 0:
            parser.error("--refresh-seconds must be positive")
        monitor(config, config_path, targets, args.refresh_seconds)
    elif args.command in {"judge", "judge-all"}:
        config_path = args.config.resolve()
        config = load_run_config(config_path)
        protocol_path = Path(str(config["scoring"]["judge_protocol"]))
        if not protocol_path.is_absolute():
            protocol_path = (config_path.parent / protocol_path).resolve()
        rubric_path = Path(str(config["scoring"].get("rubric_config")))
        if rubric_path and not rubric_path.is_absolute():
            rubric_path = (config_path.parent / rubric_path).resolve()
        targets = [args.target] if args.command == "judge" else (args.targets.split(",") if args.targets else list(config["targets"]))
        unknown = sorted(set(targets) - set(config["targets"]))
        if unknown:
            parser.error(f"unknown judge target(s): {', '.join(unknown)}")
        plans = {target: resolve_target(config, config_path, target) for target in targets}
        # A judge-all run is all-or-nothing at preflight time. This prevents a
        # partial paper comparison caused by discovering an unfinished target
        # after provider requests have already started for earlier targets.
        if args.command == "judge-all" and not args.allow_incomplete:
            incomplete = []
            for target, plan in plans.items():
                coverage_path = plan["output"] / "coverage.json"
                coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
                if coverage.get("missing", 1) or coverage.get("errors", 0):
                    incomplete.append(target)
            if incomplete:
                parser.error(f"refusing judge-all: incomplete or failed generation target(s): {', '.join(incomplete)}")
        results = {}
        def run_target(target: str, plan: dict[str, object]) -> dict[str, object]:
            return judge(
                generations=plan["output"] / "generations.jsonl", output=plan["output"],
                protocol_path=protocol_path, rubric_path=rubric_path,
                env_file=args.env_file or config_path.parent / ".env",
                batch_size=args.batch_size, retries=args.retries, resume=args.resume,
                allow_incomplete=args.allow_incomplete, max_items=args.max_items, dry_run=args.dry_run,
                routes=args.routes.split(",") if getattr(args, "routes", None) else None,
            )
        if args.command == "judge":
            target = targets[0]
            results[target] = run_target(target, plans[target])
        else:
            if args.parallelism < 1:
                parser.error("--parallelism must be positive")
            # Targets own disjoint evidence directories.  Parallel target
            # workers therefore preserve resumability and atomic append paths,
            # while the bound protects the judge gateway from an unbounded fanout.
            with ThreadPoolExecutor(max_workers=min(args.parallelism, len(targets)), thread_name_prefix="network-llm-judge") as pool:
                futures = {pool.submit(run_target, target, plans[target]): target for target in targets}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
        print(json.dumps(results if args.command == "judge-all" else results[targets[0]], indent=2, sort_keys=True))
    elif args.command in {"judge-all-start", "judge-all-status", "judge-all-stop"}:
        config_path = args.config.resolve()
        config = load_run_config(config_path)
        if args.command == "judge-all-start":
            result = start_judge_all(config, config_path, resume=args.resume, batch_size=args.batch_size, retries=args.retries, parallelism=args.parallelism)
        elif args.command == "judge-all-stop":
            result = stop_judge_all(config, config_path)
        else:
            result = judge_all_status(config, config_path)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    elif args.command == "report":
        config_path = args.config.resolve()
        config = load_run_config(config_path)
        outputs = {target: resolve_target(config, config_path, target)["output"] for target in config["targets"]}
        output_root = next(iter(outputs.values())).parents[0]
        print(json.dumps(build_comparison_report(outputs=outputs, output=output_root), indent=2, sort_keys=True))
    else:
        judge_env = load_judge_env(args.env_file)
        result = aggregate(args.generations, args.judgments)
        evaluator = Path(__file__).with_name("scoring.py")
        result["provenance"] = {
            "generations_sha256": sha256(args.generations),
            "judgments_sha256": sha256(args.judgments),
            "scoring_implementation": "network_llm_eval.scoring:aggregate",
            "scoring_implementation_sha256": sha256(evaluator),
            "judge_protocol_path": str(args.judge_protocol) if args.judge_protocol else None,
            "judge_protocol_sha256": sha256(args.judge_protocol) if args.judge_protocol else None,
            "judge_prompt_path": str(args.judge_prompt) if args.judge_prompt else None,
            "judge_prompt_sha256": sha256(args.judge_prompt) if args.judge_prompt else None,
            "judge_provider": args.judge_provider or judge_env["provider"],
            "judge_model": args.judge_model or judge_env["model"],
            "judge_model_revision": args.judge_model_revision or judge_env["model_revision"],
            "judge_environment_loaded": judge_env["env_file"] is not None,
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
