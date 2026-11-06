import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import bitsandbytes as bnb

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
).eval()

layers = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, bnb.nn.Linear8bitLt)
]
assert layers, "No Linear8bitLt layers were loaded"
name, layer = layers[0]

def report(module, args, output):
    idx = module.state.idx
    print({
        "layer": name,
        "input_shape": tuple(args[0].shape),
        "input_dtype": str(args[0].dtype),
        "weight_dtype": str(module.weight.dtype),
        "output_dtype": str(output.dtype),
        "outlier_columns": None if idx is None else idx.numel(),
    })

handle = layer.register_forward_hook(report)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain CPU caches briefly."}],
    add_generation_prompt=True,
    return_tensors="pt",
)

with torch.inference_mode():
    result = model.generate(
        inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

handle.remove()
print(tokenizer.decode(result[0, inputs.shape[-1]:], skip_special_tokens=True))
