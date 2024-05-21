# Per layer (x32)
per_layer_params = {
    "sa_norm.scale": 4096,
    "attn.q_proj": 16777216,
    "attn.k_proj": 4194304,
    "attn.v_proj": 4194304,
    "attn.output_proj": 16777216,
    "mlp_norm.scale": 4096,
    "mlp.w1": 58720256,
    "mlp.w2": 58720256,
    "mlp.w3": 58720256,
}

# Other layers
other_params = {
    "tok_embeddings": 525336576,
    "norm_scale": 4096,
    "output": 525336576,
}

# Total params
total_params_per_layer = sum(per_layer_params.values())
total_params = total_params_per_layer * 32 + sum(other_params.values())

def params_to_GB(num_params: int, n_bit: int) -> int:
    assert n_bit <= 16
    multiplier = 1.0 / 1024 / 1024 / 1024
    n_byte = n_bit / 8.0
    multiplier *= n_byte
    return num_params * multiplier

def skip_vproj_layer_GB(n_bit: int) -> int:
    assert n_bit < 16
    float_params = per_layer_params["attn.v_proj"]
    quantized_params = total_params_per_layer - float_params
    return params_to_GB(float_params, 16) + params_to_GB(quantized_params, n_bit)

def get_skip_vproj_GB(n_bit: int) -> int:
    return (
        skip_vproj_layer_GB(n_bit) * 32 +
        params_to_GB(sum(other_params.values()), n_bit)
    )

def get_skip_first3_last2_GB(n_bit: int) -> int:
    float_params = total_params_per_layer * 5
    quantized_params = total_params_per_layer * 27
    return (
        params_to_GB(float_params, 16) +
        params_to_GB(quantized_params, n_bit) +
        params_to_GB(sum(other_params.values()), n_bit)
    )

def get_skip_first3_last2_vproj_GB(n_bit: int) -> int:
    first3_last2_GB = params_to_GB(total_params_per_layer * 5, 16)
    other_27_layers_GB = skip_vproj_layer_GB(n_bit) * 27
    other_params_GB = params_to_GB(sum(other_params.values()), n_bit)
    return first3_last2_GB + other_27_layers_GB + other_params_GB

def get_skip_first3_last2_vproj_output_GB(n_bit: int) -> int:
    first3_last2_GB = params_to_GB(total_params_per_layer * 5, 16)
    other_27_layers_GB = skip_vproj_layer_GB(n_bit) * 27
    other_float_params = other_params["output"]
    other_quantized_params = sum(other_params.values()) - other_params["output"]
    other_params_GB = params_to_GB(other_float_params, 16) + params_to_GB(other_quantized_params, n_bit)
    return first3_last2_GB + other_27_layers_GB + other_params_GB

def print_all(n_bit: int) -> int:
    print("====== %s-bit quant =====" % n_bit)
    print("full quantized size: %.3f GB" % params_to_GB(total_params, n_bit))
    print("skip_vproj size: %.3f GB" % get_skip_vproj_GB(n_bit))
    print("skip_first3_last2 size: %.3f GB" % get_skip_first3_last2_GB(n_bit))
    print("skip_first3_last2_vproj size: %.3f GB" % get_skip_first3_last2_vproj_GB(n_bit))
    print("skip_first3_last2_vproj_output size: %.3f GB" % get_skip_first3_last2_vproj_output_GB(n_bit))
    print()

bf16_size = params_to_GB(total_params, 16)

print("Total params per layer (x32):", total_params_per_layer)
print("Total params:", total_params)
print("bf16 size: %.3f GB" % bf16_size)
print()

print_all(4)
print_all(3)
print_all(2)
