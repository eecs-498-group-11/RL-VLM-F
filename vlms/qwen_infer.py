from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

model = None
processor = None

def init_qwen():
    global model, processor
    if model is None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model, processor

def qwen_query_2(prompts, summary_prompt):
    """Two-stage query similar to gemini_query_2"""
    model, processor = init_qwen()
    
    # Format messages for analysis stage
    messages = [{"role": "user", "content": []}]
    for item in prompts:
        if isinstance(item, Image.Image):
            messages[0]["content"].append({
                "type": "image",
                "image": item
            })
        else:
            messages[0]["content"].append({
                "type": "text", 
                "text": item
            })

    # First stage - analysis
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    analysis = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Second stage - get final label
    summary_messages = [{"role": "user", "content": summary_prompt.format(analysis)}]
    
    summary_inputs = processor.apply_chat_template(
        summary_messages,
        tokenize=True, 
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    summary_inputs = summary_inputs.to(model.device)
    
    summary_ids = model.generate(**summary_inputs, max_new_tokens=128)
    summary_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(summary_inputs.input_ids, summary_ids)
    ]
    summary = processor.batch_decode(
        summary_ids_trimmed,
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    return summary.strip()

def qwen_score(prompts, summary_prompt):
    """Score-based query similar to gemini score"""
    model, processor = init_qwen()
    
    messages = [{"role": "user", "content": []}]
    for item in prompts:
        if isinstance(item, Image.Image):
            messages[0]["content"].append({
                "type": "image",
                "image": item
            })
        else:
            messages[0]["content"].append({
                "type": "text",
                "text": item
            })

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    analysis = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Get numerical score
    summary_messages = [{"role": "user", "content": summary_prompt.format(analysis)}]
    
    summary_inputs = processor.apply_chat_template(
        summary_messages, 
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    summary_inputs = summary_inputs.to(model.device)
    
    summary_ids = model.generate(**summary_inputs, max_new_tokens=128)
    summary_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(summary_inputs.input_ids, summary_ids)
    ]
    score = processor.batch_decode(
        summary_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return score.strip()
    