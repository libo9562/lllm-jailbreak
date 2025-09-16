import argparse
import json
import os

from conversers import load_models
from langchain_core.output_parsers import StrOutputParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default = "llama3.2:1b",
        choices=["llama3.2:1b"],
        help = "Name of target OpenAI-compatible chat model.",
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type = int,
        default = 300,
        help = "Maximum number of generated tokens for the target."
    )
    parser.add_argument(
        "--exp_name",
        type = str,
        default = "test",
        choices=['test', 'main', 'abl_c', 'abl_layer', 'multi_scene', 'abl_fig6_4', 'further_q'],
        help = "Experiment file name"
    )

    parser.add_argument(
        "--defense",
        type = str,
        default = "none",
        choices=['none', 'sr', 'ic'],
        help = "LLM defense: None, Self-Reminder, In-Context"
    )
    ##################################################
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), 'res', f'data_{args.exp_name}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    results = [{} for _ in range(len(data))]
    targetLM = load_models("Ollama", args.target_model, {"target_max_n_tokens": args.target_max_n_tokens})
    chain = targetLM | StrOutputParser()
    for idx, data in enumerate(data):
        if args.exp_name in ['test', 'main', 'further_q']:
            questions = [data['inception_attack']] + data['questions']
        else:
            questions = data['questions']
    
        results[idx]['topic'] = data['topic']
        # Get target responses
        results[idx]['qA_pairs'] = []
        for question in questions:
            print(f"Processing {data['topic']} - {question} ...")
            target_response_list = chain.invoke(question)
            results[idx]['qA_pairs'].append({'Q': question, 'A': target_response_list})

    
    results_dumped = json.dumps(results)
    os.makedirs('results', exist_ok=True)
    with open(f'./results/{args.target_model}_{args.exp_name}_{args.defense}_results.json', 'w+') as f:
        f.write(results_dumped)
    f.close()
