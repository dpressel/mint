import logging
import argparse
import os
import torch
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from mint.opt import OPTCreator
from transformers import GPT2Tokenizer

logger = logging.getLogger(__file__)

"""An example program where you can provide your GPT model with a priming sequence and have it complete
"""


def main():
    parser = argparse.ArgumentParser(description="An interactive shell with OPT")
    parser.add_argument("--model", type=str, required=True, help="Start from a model")

    parser.add_argument(
        "--query",
        type=str,
        help="Optional query.  If you pass this we wont use the repl",
    )
    parser.add_argument("--history_file", type=str, default=".gpt_history")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    model = OPTCreator.lm_from_pretrained(args.model).eval()
    model.to(args.device)

    def complete(query, sampling, temperature):
        logger.info("Query: %s", query)
        inputs = tokenizer.encode(query)
        print(inputs)
        print(tokenizer.convert_ids_to_tokens(inputs))
        outputs = []
        with torch.no_grad():

            for i in range(args.max_len):

                ids = torch.tensor(inputs, device=args.device)
                response = model(ids.unsqueeze(0)).squeeze(0)
                response = response[len(inputs) - 1]
                if sampling:
                    sample_dist = torch.softmax(response / temperature, -1)
                    output = torch.multinomial(sample_dist, num_samples=1)
                    response = output.squeeze().item()
                else:
                    response = response.argmax(-1).item()

                inputs.append(response)
                outputs.append(response)
            #outputs = ' '.join(tokenizer.convert_ids_to_tokens(outputs))
            outputs = tokenizer.decode(outputs)
            return outputs

    if args.query:
        print(complete(args.query, args.sample, args.temperature))
        return

    prompt_name = f"OPT{args.version}>> "
    history = FileHistory(args.history_file)
    while True:
        query = prompt(prompt_name, history=history)
        query = query.strip()
        if query == ":quit" or query == "quit":
            break
        if query == ":sample":
            args.sample = True
            print("Turn sampling mode on")
            continue
        if query == ":max":
            args.sample = False
            print("Turn sampling mode off")
            continue
        print(complete(query, args.sample, args.temperature))


if __name__ == "__main__":
    main()
