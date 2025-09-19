import json
from evaporate.configs import get_experiment_args
from evaporate.run_profiler import prerun_profiler, identify_attributes

def main():
    """
    This script runs only the schema identification part of the Evaporate pipeline.
    It identifies potential attributes from the documents and saves them to a file.
    """
    # 1. Get experiment arguments
    profiler_args = get_experiment_args()

    # Make sure we run in Open IE mode for schema identification
    profiler_args.do_end_to_end = True

    # # 2. Prepare data (chunking, loading gold standards, etc.)
    # print("Preparing data...")
    # data_dict = prerun_profiler(profiler_args)
    # print("Data preparation complete.")

    # # 3. Run schema identification
    # print("Running schema identification...")
    # attributes, time_taken, num_toks, evaluation_result = identify_attributes(
    #     profiler_args, 
    #     data_dict, 
    #     evaluation=False # Set to True if you want to evaluate against gold schema
    # )
    # print("Schema identification complete.")

    # # 4. Print results
    # print("\n--- Schema Identification Results ---")
    # print(f"Time taken: {time_taken:.2f} seconds")
    # print(f"Total tokens prompted: {num_toks}")
    
    # # The results are saved to a file, let's print the file path
    # output_file = f"{profiler_args.generative_index_path}/{profiler_args.run_string}_identified_schema.json"
    # print(f"\nIdentified schema saved to: {output_file}")
    
    # print("\nIdentified Attributes (sorted by relevance):")
    # for i, attr in enumerate(attributes):
    #     print(f"{i+1}. {attr}")

    # if evaluation_result:
    #     print("\n--- Evaluation Results ---")
    #     print(json.dumps(evaluation_result, indent=2))


if __name__ == "__main__":
    main()
