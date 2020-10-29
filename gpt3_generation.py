import json
import logging
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_class, tokenizer_class = MODEL_CLASSES["gpt2"]

tokenizer = tokenizer_class.from_pretrained("/home/dserdyuk/work/models")
model = model_class.from_pretrained("/home/dserdyuk/work/models")
model.to(device)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def generate_from_prompt(request):
    prompt_text = request["prompt"]
    length = adjust_length_to_model(request["length"], max_sequence_length=model.config.max_position_embeddings)
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length + len(encoded_prompt[0]),
        temperature=request["temperature"],
        top_k=request["top_k"],
        top_p=request["top_p"],
        repetition_penalty=request["repetition_penalty"],
        do_sample=True,
        num_return_sequences=request["num_return_sequences"],
    )

    generated_sequences = []
    stop_token = "</s>"
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # print(text)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
        )

        generated_sequences.append(total_sequence)

    return json.dumps(generated_sequences, ensure_ascii=False)
