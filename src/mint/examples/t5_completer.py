"""An example program where you can provide your T5 model with a priming sequence and have it complete
"""
import logging
import argparse
import os
import torch
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from tokenizers import Tokenizer
from t5 import T5Creator

logger = logging.getLogger(__file__)
DECODER_START_TOKEN = 0


def main():
    parser = argparse.ArgumentParser(description="An interactive shell with T5")
    parser.add_argument("--model", type=str, required=True, help="Start from a model")
    parser.add_argument(
        "--tok_file", type=str, required=True, help="Path to tokenizer.json file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional query.  If you pass this we wont use the repl",
    )
    parser.add_argument("--history_file", type=str, default=".t5_history")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--num_encoder_layers", default=12, type=int)
    parser.add_argument("--num_decoder_layers", default=12, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
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
    if os.path.isdir(args.tok_file):
        args.tok_file = os.path.join(args.tok_file, "tokenizer.json")
    tokenizer = Tokenizer.from_file(args.tok_file)

    model = T5Creator.from_pretrained(args.model, **vars(args)).eval()
    model.to(args.device)

    EOS_ID = tokenizer.get_vocab().get("</s>")

    def complete(query, sampling, temperature, decode_as_text=True):
        logger.info("Query: %s", query)
        tokenized_input = tokenizer.encode(query)
        logger.info("Input Sequence: %s", " ".join(tokenized_input.tokens))
        input_ids = torch.tensor(tokenized_input.ids, device=args.device).unsqueeze(0)
        input_enc = model.encode(input_ids)
        outputs = [DECODER_START_TOKEN]
        with torch.no_grad():

            for i in range(args.max_len):

                decode_ids = torch.tensor(outputs, device=args.device)
                # signature is encoder, decoder (up till now), encoder_mask, decoder_mask
                response = model.decode(input_enc, decode_ids.unsqueeze(0)).squeeze(0)
                response = response[len(decode_ids) - 1]
                if sampling:
                    sample_dist = torch.softmax(response / temperature, -1)
                    output = torch.multinomial(sample_dist, num_samples=1)
                    response = output.squeeze().item()
                else:
                    response = response.argmax(-1).item()

                if response == EOS_ID:
                    break
                outputs.append(response)
            outputs = tokenizer.decode(outputs[1:]) if decode_as_text else outputs
            return outputs

    if args.query:
        print(complete(args.query, args.sample, args.temperature))
        return

    prompt_name = "T5>> "
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
        print(complete(query, args.sample))


if __name__ == "__main__":
    main()
