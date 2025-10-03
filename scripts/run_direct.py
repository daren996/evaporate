import json
import argparse
import datetime
import numpy as np
import os
from collections import Counter

from evaporate.main import EvaporateData
from evaporate.evaluate_synthetic import main as evaluate_synthetic_main
from evaporate.evaluate_synthetic_utils import get_file_attribute
 

class PerDocumentSaver:
    def __init__(self, base_data_dir, dataset_name):
        if not os.path.exists(base_data_dir):
            os.makedirs(base_data_dir)
        self.extraction_file = os.path.join(base_data_dir, "extractions.json")
        self.extraction_data = {}
    
    def save_document_result(self, attribute, file_path, extracted_value):
        if attribute not in self.extraction_data:
            self.extraction_data[attribute] = {}
        self.extraction_data[attribute][file_path] = extracted_value
        
        with open(self.extraction_file, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_data, f, indent=2, ensure_ascii=False)


class DocumentSavingEvaporateData(EvaporateData):
    def __init__(self, profiler_args, document_saver=None):
        super().__init__(profiler_args)
        self.document_saver = document_saver
        
    def direct_extract(self, use_retrieval_model=True, is_getting_sample=False, gold=""):
        if self.attributes == []:
            print("Please run get_attribute first")
            return
            
        files = list(self.data_dict["file2chunks"].keys())
        if is_getting_sample:
            files = self.data_dict["sample_files"]
            
        time_begin = datetime.datetime.now()
        token_used = 0
        
        for attribute in self.attributes:
            if attribute in self.direct_result:
                continue
                
            new_file_chunk_dict = self.data_dict["file2chunks"]
                
            from evaporate.profiler import get_model_extractions
            extractions, num_toks, errored_out = get_model_extractions(
                new_file_chunk_dict, files, attribute,
                self.manifest_sessions[self.GOLD_MODEL], self.GOLD_MODEL,
                overwrite_cache=self.profiler_args.overwrite_cache, collecting_preds=True,
            )
            
            token_used += num_toks
            self.direct_result[attribute] = {}
            
            for file in extractions:
                golds = []
                for tmp in extractions[file]:
                    golds.append("- " + "\n- ".join(tmp))
                golds = "- " + "\n- ".join(golds)
                
                from evaporate.evaluate_profiler import pick_a_gold_label
                result_value = pick_a_gold_label(golds, attribute, self.manifest_sessions[self.GOLD_MODEL])
                self.direct_result[attribute][file] = result_value
                
                # Save after each document
                if self.document_saver:
                    self.document_saver.save_document_result(attribute, file, result_value)
                
            print(f"Finished {attribute}")
        
        time_end = datetime.datetime.now()
        self.runtime["direct_extract"] = (time_end - time_begin).total_seconds()
        self.token_used["direct_extract"] = token_used
        
        return self.direct_result, self.evaluate(self.direct_result)


def compute_detailed_metrics(preds, golds, attribute=""):
    """
    Compute detailed precision, recall, and F1 metrics for predictions vs gold labels.
    Returns both individual scores and aggregated statistics.
    """
    if len(preds) != len(golds):
        raise ValueError("Predictions and gold labels must have the same length")
    
    if len(preds) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0, 
            "f1": 0.0,
            "precision_scores": [],
            "recall_scores": [],
            "f1_scores": [],
            "num_samples": 0
        }
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for pred, gold in zip(preds, golds):
        # Convert to string if needed
        if isinstance(pred, list):
            pred = ' '.join(str(p) for p in pred)
        if isinstance(gold, list):
            gold = ' '.join(str(g) for g in gold)
        
        # Convert to string and split into tokens
        pred_str = str(pred) if pred is not None else ""
        gold_str = str(gold) if gold is not None else ""
        
        pred_toks = pred_str.split()
        gold_toks = gold_str.split()
        
        # Calculate overlap
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 and len(pred_toks) == 0:
            # Both empty - perfect match
            precision, recall, f1 = 1.0, 1.0, 1.0
        elif len(gold_toks) == 0:
            # Gold is empty but pred is not - precision 0, recall undefined (set to 1)
            precision, recall, f1 = 0.0, 1.0, 0.0
        elif len(pred_toks) == 0:
            # Pred is empty but gold is not - precision undefined (set to 1), recall 0
            precision, recall, f1 = 1.0, 0.0, 0.0
        elif num_same == 0:
            # No overlap
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            # Normal case with overlap
            precision = num_same / len(pred_toks)
            recall = num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Calculate aggregated metrics
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores, 
        "f1_scores": f1_scores,
        "num_samples": len(preds),
        "precision_std": np.std(precision_scores),
        "recall_std": np.std(recall_scores),
        "f1_std": np.std(f1_scores)
    }


def evaluate_detailed(evaporate_instance, result):
    """
    Compute detailed evaluation metrics for all attributes.
    """
    detailed_metrics = {}
    all_precision_scores = []
    all_recall_scores = []
    all_f1_scores = []
    
    for attribute in evaporate_instance.attributes:
        preds = []
        golds = []
        
        # Collect predictions and gold labels for this attribute
        for file in result[attribute]:
            if file in evaporate_instance.gold_extractions.keys():
                preds.append(result[attribute][file])
                golds.append(evaporate_instance.gold_extractions[file][attribute])
        
        if preds and golds:
            # Compute detailed metrics for this attribute
            metrics = compute_detailed_metrics(preds, golds, attribute)
            detailed_metrics[attribute] = metrics
            
            # Collect individual scores for overall statistics
            all_precision_scores.extend(metrics["precision_scores"])
            all_recall_scores.extend(metrics["recall_scores"])
            all_f1_scores.extend(metrics["f1_scores"])
        else:
            # No valid data for this attribute
            detailed_metrics[attribute] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_samples": 0
            }
    
    # Calculate overall metrics
    overall_metrics = {
        "overall_precision": np.mean(all_precision_scores) if all_precision_scores else 0.0,
        "overall_recall": np.mean(all_recall_scores) if all_recall_scores else 0.0,
        "overall_f1": np.mean(all_f1_scores) if all_f1_scores else 0.0,
        "overall_precision_std": np.std(all_precision_scores) if all_precision_scores else 0.0,
        "overall_recall_std": np.std(all_recall_scores) if all_recall_scores else 0.0,
        "overall_f1_std": np.std(all_f1_scores) if all_f1_scores else 0.0,
        "total_samples": len(all_f1_scores)
    }
    
    return detailed_metrics, overall_metrics


def get_dataset_config(dataset_type):
    """Get dataset configuration based on dataset type"""
    
    if dataset_type == "fda":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/fda_510ks/data/evaporate/fda-ai-pmas/510k",
            "GOLD_PATH": "./data/fda_510ks/table.json",
            "BASE_DATA_DIR": "./data/fda_510ks/",
            "data_lake": "fda_510ks",
            "topic": "fda 510k"
        }
    elif dataset_type == "spider_store_invoices":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_store_1_invoices/data/evaporate/spider/store_1-invoices",
            "GOLD_PATH": "./data/spider_store_1_invoices/table.json",
            "BASE_DATA_DIR": "./data/spider_store_1_invoices/",
            "data_lake": "spider_store_1_invoices",
            "topic": "store"
        }
    elif dataset_type == "spider_store_albums":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_store_1_albums/data/evaporate/spider/store_1-albums",
            "GOLD_PATH": "./data/spider_store_1_albums/table.json",
            "BASE_DATA_DIR": "./data/spider_store_1_albums/",
            "data_lake": "spider_store_1_albums",
            "topic": "store"
        }
    elif dataset_type == "spider_store_tracks":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_store_1_tracks/data/evaporate/spider/store_1-tracks",
            "GOLD_PATH": "./data/spider_store_1_tracks/table.json",
            "BASE_DATA_DIR": "./data/spider_store_1_tracks/",
            "data_lake": "spider_store_1_tracks",
            "topic": "store"
        }
    elif dataset_type == "spider_store_customers":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_store_1_customers/data/evaporate/spider/store_1-customers",
            "GOLD_PATH": "./data/spider_store_1_customers/table.json",
            "BASE_DATA_DIR": "./data/spider_store_1_customers/",
            "data_lake": "spider_store_1_customers",
            "topic": "store"
        }
    elif dataset_type == "spider_wine_wine":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_wine_1_wine/data/evaporate/spider/wine_1-wine",
            "GOLD_PATH": "./data/spider_wine_1_wine/table.json",
            "BASE_DATA_DIR": "./data/spider_wine_1_wine/",
            "data_lake": "spider_wine_1_wine",
            "topic": "wine"
        }
    elif dataset_type == "spider_soccer_player":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_soccer_1_Player/data/evaporate/spider/soccer_1-Player",
            "GOLD_PATH": "./data/spider_soccer_1_Player/table.json",
            "BASE_DATA_DIR": "./data/spider_soccer_1_Player/",
            "data_lake": "spider_soccer_1_Player",
            "topic": "soccer"
        }
    elif dataset_type == "spider_soccer_player_attributes":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_soccer_1_Player_attributes/data/evaporate/spider/soccer_1-Player_attributes",
            "GOLD_PATH": "./data/spider_soccer_1_Player_attributes/table.json",
            "BASE_DATA_DIR": "./data/spider_soccer_1_Player_attributes/",
            "data_lake": "spider_soccer_1_Player_attributes",
            "topic": "soccer"
        }
    elif dataset_type == "spider_soccer_team_attributes":
        return {
            "MANIFEST_URL": "http://127.0.0.1:5000",
            "DATA_DIR": "./data/spider_soccer_1_Team_Attributes/data/evaporate/spider/soccer_1-Player",
            "GOLD_PATH": "./data/spider_soccer_1_Team_Attributes/table.json",
            "BASE_DATA_DIR": "./data/spider_soccer_1_Team_Attributes/",
            "data_lake": "spider_soccer_1_Team_Attributes",
            "topic": "soccer"
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}.")


def main():
    parser = argparse.ArgumentParser(description="Run direct extraction on different datasets")
    parser.add_argument("--dataset", 
                       choices=["fda", "spider_store_invoices", "spider_store_albums", "spider_store_tracks", "spider_store_customers",
                                "spider_soccer_player", "spider_soccer_player_attributes", "spider_soccer_team_attributes",
                                "spider_wine_wine"],
                       default="fda",
                       help="Dataset to use for extraction")
    parser.add_argument("--model", 
                       default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                       help="Model to use for extraction")
    
    args = parser.parse_args()
    
    # Get dataset configuration
    config = get_dataset_config(args.dataset)
    
    print(f"Running extraction on {args.dataset} dataset...")
    print(f"Data directory: {config['DATA_DIR']}")
    print(f"Gold file: {config['GOLD_PATH']}")
    
    # Set up profiler arguments
    profiler_args = {
        "data_lake": config["data_lake"], 
        "combiner_mode": "mv", 
        "do_end_to_end": False,
        "KEYS": [""], 
        "MODELS": [args.model],
        "EXTRACTION_MODELS": [args.model],
        "GOLD_KEY": args.model,
        "MODEL2URL": {},
        "data_dir": config["DATA_DIR"], 
        "chunk_size": 3000,
        "base_data_dir": config["BASE_DATA_DIR"], 
        "gold_extractions_file": config["GOLD_PATH"],
        "direct_extract_model": args.model,
        "topic": config["topic"]
    }

    # Initialize per-document saver
    document_saver = PerDocumentSaver(config["BASE_DATA_DIR"], args.dataset)
    evaporate = DocumentSavingEvaporateData(profiler_args, document_saver)

    # Get attributes
    attributes = evaporate.get_attribute(do_end_to_end=False)
    print(f"Found attributes: {attributes}")

    # Load gold data
    with open(config["GOLD_PATH"], "r") as f:
        gold = json.load(f)
    
    # Perform direct extraction
    print("Performing direct extraction...")
    direct_attribute, direct_eval = evaporate.direct_extract(is_getting_sample=False, use_retrieval_model=False)
    
    # print(f"\nExtracted attributes: {direct_attribute}")
    print(f"\nBasic evaluation results: {direct_eval}")
    
    # Compute detailed evaluation metrics
    detailed_metrics, overall_metrics = evaluate_detailed(evaporate, direct_attribute)
    
    # Display detailed evaluation results
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\nOVERALL PERFORMANCE:")
    print("-" * 50)
    print(f"  {'Overall Precision':30s}: {overall_metrics['overall_precision']:.4f} ± {overall_metrics['overall_precision_std']:.4f}")
    print(f"  {'Overall Recall':30s}: {overall_metrics['overall_recall']:.4f} ± {overall_metrics['overall_recall_std']:.4f}")
    print(f"  {'Overall F1-Score':30s}: {overall_metrics['overall_f1']:.4f} ± {overall_metrics['overall_f1_std']:.4f}")
    print(f"  {'Total Samples Evaluated':30s}: {overall_metrics['total_samples']}")
    
    # Per-attribute metrics
    print("\nPER-ATTRIBUTE PERFORMANCE:")
    print("-" * 50)
    print(f"{'Attribute':25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Samples':>8s}")
    print("-" * 70)
    
    for attribute, metrics in detailed_metrics.items():
        if metrics['num_samples'] > 0:
            print(f"{attribute[:24]:25s} {metrics['precision']:10.4f} {metrics['recall']:10.4f} {metrics['f1']:10.4f} {metrics['num_samples']:8d}")
        else:
            print(f"{attribute[:24]:25s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {metrics['num_samples']:8d}")
    
    print("="*80)
    
    # Display comprehensive token usage and performance statistics
    print("\n" + "="*70)
    print("PROFILER PERFORMANCE STATISTICS")
    print("="*70)
    
    # Token usage breakdown
    print("\nTOKEN USAGE BY STAGE:")
    print("-" * 40)
    total_tokens = 0
    for stage, tokens in evaporate.token_used.items():
        print(f"  {stage:28s}: {tokens:>10,} tokens")
        total_tokens += tokens
    
    print("-" * 40)
    print(f"  {'TOTAL TOKENS USED':28s}: {total_tokens:>10,} tokens")
    
    # Runtime breakdown
    print("\nRUNTIME BY STAGE:")
    print("-" * 40)
    total_runtime = 0
    for stage, runtime in evaporate.runtime.items():
        print(f"  {stage:28s}: {runtime:>10.2f} seconds")
        total_runtime += runtime
    
    print("-" * 40)
    print(f"  {'TOTAL RUNTIME':28s}: {total_runtime:>10.2f} seconds")
    
    # Cost estimation (rough approximation for common models)
    print("\nESTIMATED COST (approximate):")
    print("-" * 40)
    # Assume average cost of $0.002 per 1K tokens (varies by model)
    estimated_cost = (total_tokens / 1000) * 0.002
    print(f"  {'Estimated cost (USD)':28s}: ${estimated_cost:>10.4f}")
    
    # Efficiency metrics
    tokens_per_second = 0
    if total_runtime > 0:
        tokens_per_second = total_tokens / total_runtime
        print(f"  {'Tokens per second':28s}: {tokens_per_second:>10.1f}")
    
    print("="*70)
    
    # Save comprehensive statistics to JSON file
    stats = {
        "dataset": args.dataset,
        "model": args.model,
        "token_usage": evaporate.token_used,
        "runtime": evaporate.runtime,
        "total_tokens": total_tokens,
        "total_runtime": total_runtime,
        "estimated_cost_usd": estimated_cost,
        "tokens_per_second": tokens_per_second,
        "attributes_extracted": list(direct_attribute.keys()) if direct_attribute else [],
        "basic_evaluation_results": direct_eval,
        "detailed_metrics": {
            "overall": overall_metrics,
            "per_attribute": detailed_metrics
        }
    }
    
    # Save to fixed filename
    stats_filename = config['BASE_DATA_DIR'] + f"token_stats_{args.dataset}.json"
    
    with open(stats_filename, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nToken usage statistics saved to: {stats_filename}")
    
    print(f"Extraction data saved to: {document_saver.extraction_file}")


if __name__ == "__main__":
    main()

