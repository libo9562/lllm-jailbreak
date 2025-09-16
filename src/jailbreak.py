"""
Unified Jailbreak Interface

This module provides a unified interface for different jailbreaking techniques:
- Deepinception: Hypnotize Large Language Model to Be Jailbreaker
- PAIR: Jailbreaking Black Box Large Language Models in Twenty Queries

Original implementations are preserved in their respective directories with citations.
"""

import argparse
import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain_core.output_parsers import StrOutputParser


class JailbreakTechnique(Enum):
    """Available jailbreaking techniques."""
    DEEPINCEPTION = "deepinception"
    PAIR = "pair"


class JailbreakResult:
    """Container for jailbreak results."""

    def __init__(self, technique: str, success: bool, responses: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        self.technique = technique
        self.success = success
        self.responses = responses
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technique": self.technique,
            "success": self.success,
            "responses": self.responses,
            "metadata": self.metadata
        }


class BaseJailbreaker(ABC):
    """Base class for jailbreaking techniques."""

    def __init__(self, target_model: str, **kwargs):
        self.target_model = target_model
        self.config = kwargs

    @abstractmethod
    def execute(self, goal: str, **kwargs) -> JailbreakResult:
        """Execute the jailbreaking technique."""
        pass

    @abstractmethod
    def get_technique_name(self) -> str:
        """Get the name of the technique."""
        pass


class DeepinceptionJailbreaker(BaseJailbreaker):
    """Deepinception jailbreaking technique using LangChain."""

    def __init__(self, target_model: str = "llama3.2:1b", **kwargs):
        super().__init__(target_model, **kwargs)
        from .Deepinception.conversers import load_models

        self.load_models = load_models
        self.max_tokens = kwargs.get("target_max_n_tokens", 300)

    def get_technique_name(self) -> str:
        return "Deepinception"

    def execute(self, goal: str = None, exp_name: str = "test", defense: str = "none", **kwargs) -> JailbreakResult:
        """
        Execute Deepinception technique.

        Args:
            goal: Not used in Deepinception (uses predefined data files)
            exp_name: Experiment name (test, main, abl_c, abl_layer, multi_scene, abl_fig6_4, further_q)
            defense: Defense mechanism (none, sr, ic)
        """
        try:
            # Load experiment data
            data_path = os.path.join(
                os.path.dirname(__file__),
                "Deepinception",
                "res",
                f"data_{exp_name}.json"
            )

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Experiment data file not found: {data_path}")

            with open(data_path, "r") as f:
                data = json.load(f)

            # Initialize model
            target_lm = self.load_models(
                "Ollama",
                self.target_model,
                {"target_max_n_tokens": self.max_tokens}
            )
            chain = target_lm | StrOutputParser()

            # Process each data item
            results = []
            for idx, item in enumerate(data):
                if exp_name in ["test", "main", "further_q"]:
                    questions = [item["inception_attack"]] + item["questions"]
                else:
                    questions = item["questions"]

                item_result = {
                    "topic": item["topic"],
                    "qA_pairs": []
                }

                for question in questions:
                    response = chain.invoke(question)
                    item_result["qA_pairs"].append({"Q": question, "A": response})

                results.append(item_result)

            # Determine success (basic heuristic)
            success = len(results) > 0 and any(
                len(item["qA_pairs"]) > 0 for item in results
            )

            return JailbreakResult(
                technique=self.get_technique_name(),
                success=success,
                responses=results,
                metadata={
                    "exp_name": exp_name,
                    "defense": defense,
                    "target_model": self.target_model,
                    "max_tokens": self.max_tokens
                }
            )

        except Exception as e:
            return JailbreakResult(
                technique=self.get_technique_name(),
                success=False,
                responses=[],
                metadata={"error": str(e)}
            )


class PAIRJailbreaker(BaseJailbreaker):
    """PAIR jailbreaking technique using LangChain."""

    def __init__(self, target_model: str = "gpt-3.5-turbo", attack_model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(target_model, **kwargs)
        self.attack_model = attack_model

        # Default PAIR parameters
        self.n_streams = kwargs.get("n_streams", 3)
        self.n_iterations = kwargs.get("n_iterations", 3)
        self.attack_max_n_tokens = kwargs.get("attack_max_n_tokens", 500)
        self.target_max_n_tokens = kwargs.get("target_max_n_tokens", 150)
        self.judge_model = kwargs.get("judge_model", "simple")

    def get_technique_name(self) -> str:
        return "PAIR"

    def execute(self, goal: str, target_str: str = None, **kwargs) -> JailbreakResult:
        """
        Execute PAIR technique using LangChain.

        Args:
            goal: The jailbreaking goal
            target_str: Target response string
        """
        try:
            from .JailbreakingLLMs.pair_langchain import run_langchain_pair

            if target_str is None:
                target_str = f"Sure, here is {goal}"

            # Run PAIR attack
            pair_result = run_langchain_pair(
                goal=goal,
                target_str=target_str,
                attack_model=self.attack_model,
                target_model=self.target_model,
                judge_model=self.judge_model,
                n_streams=self.n_streams,
                n_iterations=self.n_iterations,
                attack_max_tokens=self.attack_max_n_tokens,
                target_max_tokens=self.target_max_n_tokens
            )

            # Convert PAIR result to JailbreakResult
            responses = [{
                "best_prompt": pair_result["best_prompt"],
                "best_response": pair_result["best_response"],
                "best_score": pair_result["best_score"],
                "all_iterations": pair_result["all_iterations"]
            }]

            return JailbreakResult(
                technique=self.get_technique_name(),
                success=pair_result["success"],
                responses=responses,
                metadata={
                    "goal": goal,
                    "target_str": target_str,
                    "attack_model": self.attack_model,
                    "target_model": self.target_model,
                    "n_streams": self.n_streams,
                    "n_iterations": pair_result["n_iterations"],
                    "final_scores": pair_result["final_scores"],
                    "best_score": pair_result["best_score"]
                }
            )

        except Exception as e:
            return JailbreakResult(
                technique=self.get_technique_name(),
                success=False,
                responses=[],
                metadata={
                    "error": str(e),
                    "goal": goal,
                    "target_str": target_str,
                    "attack_model": self.attack_model,
                    "target_model": self.target_model
                }
            )


class UnifiedJailbreaker:
    """Unified interface for all jailbreaking techniques."""

    def __init__(self):
        self.techniques = {
            JailbreakTechnique.DEEPINCEPTION: DeepinceptionJailbreaker,
            JailbreakTechnique.PAIR: PAIRJailbreaker
        }

    def create_jailbreaker(self, technique: Union[str, JailbreakTechnique], **kwargs) -> BaseJailbreaker:
        """Create a jailbreaker instance for the specified technique."""
        if isinstance(technique, str):
            technique = JailbreakTechnique(technique)

        if technique not in self.techniques:
            raise ValueError(f"Unknown technique: {technique}")

        return self.techniques[technique](**kwargs)

    def execute(self, technique: Union[str, JailbreakTechnique], goal: str = None, **kwargs) -> JailbreakResult:
        """Execute a jailbreaking technique."""
        jailbreaker = self.create_jailbreaker(technique, **kwargs)
        return jailbreaker.execute(goal, **kwargs)

    def list_techniques(self) -> List[str]:
        """List available jailbreaking techniques."""
        return [technique.value for technique in JailbreakTechnique]


def main():
    """Command-line interface for the unified jailbreaker."""
    parser = argparse.ArgumentParser(description="Unified Jailbreak Interface")

    # Technique selection
    parser.add_argument(
        "--technique",
        choices=["deepinception", "pair"],
        required=True,
        help="Jailbreaking technique to use"
    )

    # Common parameters
    parser.add_argument(
        "--target-model",
        default="llama3.2:1b",
        help="Target model name"
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="Jailbreaking goal (required for PAIR, optional for Deepinception)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (JSON format)"
    )

    # Deepinception specific
    parser.add_argument(
        "--exp-name",
        default="test",
        choices=["test", "main", "abl_c", "abl_layer", "multi_scene", "abl_fig6_4", "further_q"],
        help="Deepinception experiment name"
    )
    parser.add_argument(
        "--defense",
        default="none",
        choices=["none", "sr", "ic"],
        help="Defense mechanism"
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type=int,
        default=300,
        help="Maximum tokens for target model"
    )

    # PAIR specific
    parser.add_argument(
        "--attack-model",
        default="vicuna-13b-v1.5",
        help="Attack model for PAIR"
    )
    parser.add_argument(
        "--target-str",
        type=str,
        help="Target response string for PAIR"
    )
    parser.add_argument(
        "--n-streams",
        type=int,
        default=3,
        help="Number of concurrent streams for PAIR"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=3,
        help="Number of iterations for PAIR"
    )

    args = parser.parse_args()

    # Create unified jailbreaker
    jailbreaker = UnifiedJailbreaker()

    # Prepare kwargs based on technique
    kwargs = {
        "target_model": args.target_model,
        "target_max_n_tokens": args.target_max_n_tokens
    }

    if args.technique == "deepinception":
        kwargs.update({
            "exp_name": args.exp_name,
            "defense": args.defense
        })
    elif args.technique == "pair":
        if not args.goal:
            parser.error("--goal is required for PAIR technique")
        kwargs.update({
            "attack_model": args.attack_model,
            "target_str": args.target_str or f"Sure, here is {args.goal}",
            "n_streams": args.n_streams,
            "n_iterations": args.n_iterations
        })

    # Execute jailbreak
    result = jailbreaker.execute(args.technique, args.goal, **kwargs)

    # Output results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result.to_dict(), indent=2))

    # Print summary
    print(f"\nTechnique: {result.technique}")
    print(f"Success: {result.success}")
    print(f"Number of responses: {len(result.responses)}")


if __name__ == "__main__":
    main()