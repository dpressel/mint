import logging
import argparse
import torch
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from tfs.gpt import GPTCreator, GPT2Creator
from tokenizers import ByteLevelBPETokenizer

logger = logging.getLogger(__file__)

"""An example program where you can provide your GPT model with a priming sequence and have it complete
"""


def main():
    parser = argparse.ArgumentParser(description='An interactive shell with BERT')
    parser.add_argument("--model", type=str, required=True, help="Start from a model")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--merges_file", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--query", type=str, help="Optional query.  If you pass this we wont use the repl")
    parser.add_argument("--history_file", type=str, default=".gpt_history")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--version", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = ByteLevelBPETokenizer(args.vocab_file, args.merges_file)#, add_prefix_space=True)
    Creator = GPT2Creator if args.version == 2 else GPTCreator
    model = Creator.lm_from_pretrained(args.model).eval()
    model.to(args.device)

    def complete(query, sampling):
        logger.info("Query: %s", query)
        tokenized_input = tokenizer.encode(query)  # .split(), is_pretokenized=True, add_prefix_space=True)
        logger.info("Priming Sequence: %s", ' '.join(tokenized_input.tokens))
        inputs = tokenized_input.ids
        outputs = []
        with torch.no_grad():

            for i in range(args.max_len):

                ids = torch.tensor(inputs, device=args.device)
                response = model(ids.unsqueeze(0)).squeeze(0)
                response = response[len(inputs) - 1]
                if sampling:
                    sample_dist = response.exp()
                    output = torch.multinomial(sample_dist, num_samples=1)
                    response = output.squeeze().item()
                else:
                    response = response.argmax(-1).item()

                inputs.append(response)
                outputs.append(response)
            outputs = tokenizer.decode(outputs)
            return outputs.replace('</w>', ' ')

    if args.query:
        print(complete(args.query, args.sample))
        return

    prompt_name = f'GPT{args.version}>> '
    history = FileHistory(args.history_file)
    while True:
        query = prompt(prompt_name, history=history)
        query = query.strip()
        if query == ':quit' or query == 'quit':
            break
        if query == ':sample':
            args.sample = True
            print("Turn sampling mode on")
            continue
        if query == ':max':
            args.sample = False
            print("Turn sampling mode off")
            continue
        print(complete(query, args.sample))


if __name__ == "__main__":
    main()
