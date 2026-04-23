import argparse
import os
from time import time
import numpy as np
from dotenv import load_dotenv
import yaml
from vec_mem import VecMem, VecMemConfig
import concurrent.futures
import json
import threading
from collections import defaultdict
from typing import List, Dict

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm
from dotenv import load_dotenv
from src.pipeline import VecMemPipeline
from src.conv_loader import ConvLoader, Conversation, Question
from embed_manager import EmbedManager
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


load_dotenv()


def run_experiment(args, vecmem: VecMem, data_path: str, output_file_path: str):
    print(f"Iterative anwser: {vecmem.enable_iter_anwser}")

    # Load convs and questions from dataset
    convs, questions = ConvLoader.load_locomo(data_path)
    if vecmem.conv_limit is not None:
        convs = convs[: vecmem.conv_limit]
        questions = questions[: vecmem.conv_limit]
    FINAL_RESULTS = {}

    # Create token stats json file
    stats_file_path = output_file_path.replace(".json", "_token_stats.json")
    os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)

    for conv_idx in tqdm(range(len(convs)), desc="Processing conversations"):
        chat_history: Conversation = convs[conv_idx]
        conv_questions: List[Question] = questions[conv_idx]
        conv_embedding: np.ndarray = np.load(
            os.path.join(os.getenv("LOCOMO_EMBEDDING_PATH"), f"conv_{conv_idx}.npy")
        )
        question_embedding: np.ndarray = np.load(
            os.path.join(
                os.getenv("LOCOMO_EMBEDDING_PATH"), f"question_{conv_idx}.npy"
            )
        )
        conv_mapping: Dict[int, str] = EmbedManager.construct_mapping(chat_history)

        # Init to an empty state
        vecmem.vec_store.reset()
        vecmem.aug_mem.reset()
        vecmem._payload_mapping = {}
        vecmem.id_assigner = 0

        # Initialize conversation results
        FINAL_RESULTS[str(conv_idx)] = []

        if not hasattr(vecmem, "_lock"):
            vecmem._lock = threading.Lock()

        # Adding memory phase
        for session_idx in tqdm(range(len(conv_embedding)), desc="Adding memories", leave=False):
            raw_text = conv_mapping[session_idx]
            emb = conv_embedding[session_idx]
            with vecmem._lock:
                vecmem.add_memory(emb, raw_text)

        FINAL_RESULTS[str(conv_idx)].append(vecmem.vec_store)
        FINAL_RESULTS[str(conv_idx)].append(vecmem.aug_mem)
        if vecmem.enable_semantic_memory:
            FINAL_RESULTS[str(conv_idx)].append(vecmem.semantic_memory)

        # Retrieving memory phase
        for q_idx, item in tqdm(
            enumerate(conv_questions), desc="Answering questions", leave=False
        ):
            question: str = item.content
            answer: str = item.anwser
            category: int = item.category
            response: str = vecmem.anwser_question(
                question, question_embedding[q_idx]
            )
            FINAL_RESULTS[str(conv_idx)].append(
                {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "response": response,
                }
            )
        if vecmem.token_monitor is not None:
            vecmem.token_monitor.end_conversation(
                len(conv_questions), 0, len(vecmem._payload_mapping)
            )

    # Write all results at once in the correct format
    with open(output_file_path, "w") as f:
        json.dump(FINAL_RESULTS, f, indent=4)
    if vecmem.token_monitor is not None:
        vecmem.token_monitor.save_stats(stats_file_path)


def add_arguments():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top memories to retrieve"
    )
    parser.add_argument(
        "--min_aug_count",
        type=int,
        default=3,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--min_relevant_score",
        type=float,
        default=0.7,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--retrieve_raw_topk",
        type=int,
        default=5,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--retrieve_aug_topk",
        type=int,
        default=5,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/tang/workspace/HMEMS/src/VecMem/results/Locomo/scores/debug.json",
        help="Output file name",
    )
    parser.add_argument(
        "--enable_iter_anwser",
        action="store_true",
        help="Whether to enable iterative anwser",
    )
    parser.add_argument(
        "--itr_raw_topk",
        type=int,
        default=3,
        help="Number of top memories to retrieve during iterative anwser",
    )
    parser.add_argument(
        "--itr_aug_topk",
        type=int,
        default=3,
        help="Number of top memories to retrieve during iterative anwser",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use",
    )
    parser.add_argument(
        "--merge_with_aug_thresh",
        type=float,
        default=0.85,
        help="Threshold to merge new memory with augmented memory",
    )
    parser.add_argument(
        "--enable_stat",
        action="store_true",
        help="Whether to enable token monitoring and saving ratio counting",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Whether to evaluate only",
    )
    parser.add_argument(
        "--init_env",
        action="store_true",
        help="Whether to initialize the environment",
    )
    parser.add_argument(
        "--conv_limit",
        type=int,
        default=None,
        help="Number of conversations to sample for testing",
    )
    parser.add_argument(
        "--enable_semantic_memory",
        action="store_true",
        help="Whether to enable semantic memory",
    )
    parser.add_argument(
        "--semantic_memory_topk",
        type=int,
        default=10,
        help="Number of top memories to retrieve from semantic memory",
    )
    parser.add_argument(
        "--semantic_memory_threshold",
        type=float,
        default=0.5,
        help="Threshold to consider a memory as semantic memory",
    )
    return parser.parse_args()


def process_item(item_data):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        # Skip category 5
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
            }
        )

    return local_results


def eval_experiment(input_file: str, output_file: str):
    # Use gpt-4o-mini to evaluate the experiment
    os.environ["MODEL"] = "gpt-4o-mini"
    with open(input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_item, item_data) for item_data in data.items()
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

    summary_path = output_file[:-5] + "_summary"

    VecMemPipeline.generate_eval_summary(output_file, summary_path)


def environment_setup():
    if not os.path.exists(os.getenv("LOCOMO_PATH")):
        raise ValueError(f"Locomo dataset not found at {os.getenv('LOCOMO_PATH')}")
    if not os.path.exists(os.getenv("LOCOMO_RES_PATH")):
        os.makedirs(os.getenv("LOCOMO_RES_PATH"))
    if not os.path.exists(os.getenv("LOCOMO_SCORE_PATH")):
        os.makedirs(os.getenv("LOCOMO_SCORE_PATH"))
    if not os.path.exists(os.getenv("LOCOMO_EMBEDDING_PATH")):
        os.makedirs(os.getenv("LOCOMO_EMBEDDING_PATH"))
    if not os.path.exists(os.getenv("LOCOMO_INDEX_PATH")):
        os.makedirs(os.getenv("LOCOMO_INDEX_PATH"))
    pipeline = VecMemPipeline()
    pipeline.generate_embeddings_and_save(
        os.getenv("LOCOMO_PATH"), os.getenv("LOCOMO_EMBEDDING_PATH")
    )
    pipeline.construct_index_and_save(
        os.getenv("LOCOMO_EMBEDDING_PATH"), os.getenv("LOCOMO_INDEX_PATH")
    )


def main():
    args = add_arguments()
    if args.init_env:
        environment_setup()
        return
    score_output_dir = os.getenv("LOCOMO_SCORE_PATH")

    if args.eval_only:
        anwser_file_name = args.output_file.split("/")[-1]
        score_file_name = score_output_dir + "/" + anwser_file_name
        logger.info(
            f"Evaluating experiment. Source file: {args.output_file}, Score file: {score_file_name}"
        )
        eval_experiment(args.output_file, score_file_name)
        return

    print(f"Using model: {os.getenv('MODEL1')}")

    cfg = VecMemConfig(
        min_aug_count=args.min_aug_count,
        min_relevant_score=args.min_relevant_score,
        retrieve_raw_topk=args.retrieve_raw_topk,
        retrieve_aug_topk=args.retrieve_aug_topk,
        enable_iter_anwser=args.enable_iter_anwser,
        iterative_raw_topk=args.itr_raw_topk,
        iterative_aug_topk=args.itr_aug_topk,
        merge_with_aug_thresh=args.merge_with_aug_thresh,
        conv_limit=args.conv_limit,
        enable_semantic_memory=args.enable_semantic_memory,
        semantic_memory_topk=args.semantic_memory_topk,
        semantic_memory_threshold=args.semantic_memory_threshold,
    )
    print(f"Enable iterative anwser: {cfg.enable_iter_anwser}")
    vec_mem = VecMem(cfg)
    print(f"Enable token monitoring: {args.enable_stat}")
    if args.enable_stat:
        vec_mem.enable_token_monitoring()

    run_experiment(args, vec_mem, os.getenv("LOCOMO_PATH"), args.output_file)

    # Eval section
    anwser_file_name = args.output_file.split("/")[-1]
    score_file_name = score_output_dir + "/" + anwser_file_name
    logger.info(
        f"Evaluating experiment. Source file: {args.output_file}, Score file: {score_file_name}"
    )
    eval_experiment(args.output_file, score_file_name)


if __name__ == "__main__":
    main()
