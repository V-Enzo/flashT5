from transformers import (
    AutoTokenizer,
)


def get_tokenizer(args):
    if hasattr(args.data, 'tokenizer_path'):
        tokenizerName = args.data.tokenizer_path
    else:
        tokenizerName = "/data2/charles/Tokenizer/NetT5WordPiece65536"
    print(f"Using tokenizer from {tokenizerName}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizerName,
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    return tokenizer


def hex_string_to_str(hex_string_ori):
    hex_string = hex_string_ori.replace(' ', '').replace('</s>', '')
    if hex_string.endswith('00'):
        hex_string = hex_string[:-2]
    try:
        s = bytes.fromhex(hex_string).decode('utf-8')
    except:
        # can not convert back as the hex string is not valid
        s = hex_string_ori.replace('</s>', '')
    return s



def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    modified by charles Li.
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1 # real input length
        #XXX Modified by Charles
        _output_length = num_noise_tokens + num_noise_spans + num_noise_spans-1 + 1 # real output length
        # It is possible that the truncation is sparse (short span) and needs more special token to mask, which will increase the span. 
        # It probably becomes more obvious when only truncating over length > 5.
        return _input_length, _output_length

    tokens_length = inputs_length 

    # if this is smaller than the max input length, we may continue to maximize the real input maximun size
    # until the processed input size reach to the max input length
    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length
