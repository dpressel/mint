import logging
import argparse
import torch
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from tfs.bert import BertCreator
from tokenizers import BertWordPieceTokenizer

logger = logging.getLogger(__file__)

"""An example program where you can provide your BERT model with masked tokens and have it unmask them
"""


def main():
    parser = argparse.ArgumentParser(description='An interactive shell with BERT')
    parser.add_argument("--model", type=str, required=True, help="Start from a model")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--query", type=str, help="Optional query.  If you pass this we wont use the repl")
    parser.add_argument("--lowercase", action="store_true", help="Vocab is lower case")
    parser.add_argument("--history_file", type=str, default=".bert_history")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertWordPieceTokenizer(args.vocab_file, lowercase=args.lowercase)
    model = BertCreator.mlm_from_pretrained(args.model).eval()
    model.to(args.device)

    def complete(query, sampling):
        with torch.no_grad():
            tokenized_input = tokenizer.encode(query)
            masked_offsets = [i for i, t in enumerate(tokenized_input.tokens) if t == '[MASK]']
            tokens = tokenized_input.tokens
            logger.debug("Masked: %s", ' '.join(tokens))
            ids = torch.tensor(tokenized_input.ids, device=args.device).unsqueeze(0)
            response = model(ids).squeeze(0)
            if sampling:
                sample_dist = response.exp()
                output = torch.multinomial(sample_dist, num_samples=1)
                response = output.squeeze().tolist()
            else:
                response = response.argmax(-1).tolist()
            for off in masked_offsets:
                tokens[off] = tokenizer.id_to_token(response[off])
            return ' '.join(tokens[1:-1]).replace(' ##', '')

    if args.query:
        print(complete(args.query, args.sample))
        return

    prompt_name = 'BERT>> '
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
