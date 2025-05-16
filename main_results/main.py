from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    LlavaOnevisionProcessor,
    LlavaOnevisionForConditionalGeneration,
)
import torch
import torch.nn.functional as F
from PIL import Image
import json
from tqdm import trange
import random
import click
import clip
from torchvision import transforms
from torchvision.models import vit_b_16
import os
import numpy as np

# Comment out only if using the EVA-CLIP model
########################################################################################
# import sys
# sys.path.append("./VlmClassifier/EVA/EVA-CLIP/rei")
# from eva_clip import create_model_and_transforms, get_tokenizer
########################################################################################

# === TDA Parameters & State ===
TDA_CFG = {
    "cap_pos": 3, "cap_neg": 2,
    "alpha_pos": 2.0, "beta_pos": 5.0,
    "alpha_neg": 2, "beta_neg": 5.0,
    "tau_low": 0.2, "tau_high": 0.5,
    "pl": 0.03,
}
tda_positive_cache = {}  # {class_idx: [(feature, entropy)]}
tda_negative_cache = {}  # {class_idx: [(feature, entropy, neg_mask)]}

def tda_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-6)).sum(dim=-1)

def update_tda_cache(cache, class_idx, feature, entropy, cap, extra=None):
    if class_idx not in cache:
        cache[class_idx] = []
    entry = (feature, entropy) if extra is None else (feature, entropy, extra)
    if len(cache[class_idx]) < cap:
        cache[class_idx].append(entry)
    else:
        worst = max(cache[class_idx], key=lambda x: x[1])
        if entropy < worst[1]:
            cache[class_idx].remove(worst)
            cache[class_idx].append(entry)
    cache[class_idx].sort(key=lambda x: x[1])

def compute_tda_logits(query, cache, alpha, beta, num_classes, is_negative=False):
    if not cache:
        return torch.zeros((1, num_classes), device=query.device)
    keys, vals = [], []
    for cls, items in cache.items():
        for item in items:
            keys.append(item[0].float())
            if is_negative:
                vals.append(item[2].float())
            else:
                v = torch.zeros(num_classes, device=query.device)
                v[cls] = 1.0
                vals.append(v)
    K = torch.stack(keys).T  # (D, N)
    V = torch.stack(vals)    # (N, C)
    query = query.float()    # (1, D)
    affinity = query @ K     # (1, N)
    weights = alpha * torch.exp(-beta * (1 - affinity))
    return weights @ V       # (1, C)

@torch.no_grad()
def extract_image_features(model, processor, pixel_values):
    with torch.no_grad():
        vision_outputs = model.vision_tower(pixel_values)
        image_features = F.normalize(vision_outputs.last_hidden_state.mean(dim=1), dim=-1)

        # Step 2: Project to language space
        projected_feats = model.multi_modal_projector(image_features)  # [B, 4096]
    return projected_feats



def main(
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    chain_of_thought,
    batch_size,
    fixed_order,
    init_prompt=None,
):
    if "llava-v1.6" in model_id:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    elif "blip2" in model_id:
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    elif "instructblip" in model_id:
        processor = InstructBlipProcessor.from_pretrained(model_id)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    elif "llava-onevision" in model_id:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, device_map="cuda", torch_dtype=torch.bfloat16
        )
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    data = [json.loads(line) for line in open(data_path)]
    data = [item for item in data if item["split"] == split]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    if os.path.exists(output_path):
        outputs = [json.loads(line) for line in open(output_path)]
        data = data[len(outputs) :]

    if init_prompt is None:
        init_prompt = "What type of object is in this photo?"

    # === Precompute W_c ===
    # Generate class text embeddings
    prompts = [f"A photo of a {c}" for c in classes]
    input_ids = processor.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to("cuda")
    with torch.no_grad():
        text_embed = model.language_model.model.embed_tokens(input_ids).mean(dim=1)  # [N, D]
        W_c = F.normalize(text_embed, dim=-1)

    with open(output_path, "a") as f:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            images = [Image.open(item["image"]) for item in batch]
            image_paths = [item["image"] for item in batch] # for llava-onevision

            if including_label:
                choices = []
                for item in batch:
                    if fixed_order:
                        assert n_labels == len(classes)
                        choices.append(classes)
                    else:
                        label = item["label"]
                        sample_seed = hash(f"{seed}_{item['image']}") % (2**32)
                        random.seed(sample_seed)  # Set seed specific to this sample
                        other_choices = random.sample(
                            sorted(list(set(classes) - set([label]))), n_labels - 1
                        )
                        shuffled_choices = [label] + other_choices

                        random.shuffle(shuffled_choices)
                        choices.append(shuffled_choices)
                questions = [
                    f"{init_prompt} Choose one from \"{', '.join(choice)}\"."
                    for choice in choices
                ]
                #questions = [
                #    f"{init_prompt} ### Question\n*Question*: Given an image, what is the most likely answer among [{', '.join(choice)}]?\n\n*Answer*:"
                #    for choice in choices
                #]
            else:
                questions = [init_prompt for _ in batch]

            if "llava-v1.6-mistral" in model_id:
                assert not chain_of_thought
                prompts = [
                    f"[INST] <image>\n{question}[/INST]" for question in questions
                ]
            elif "llava-v1.6-vicuna" in model_id:
                assert not chain_of_thought
                prompts = [
                    f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
                    for question in questions
                ]
            elif "blip" in model_id:
                if not chain_of_thought:
                    prompts = [
                        f"Question: {question} Answer:" for question in questions
                    ]
                else:
                    prompts = [
                        f"Question: {question} Let's think step by step. Answer:"
                        for question in questions
                    ]
            elif "llava-onevision" in model_id:
                assert not chain_of_thought # disregard cot for now

                conversations = []
                for question, image_path in zip(questions, image_paths):
                    conv = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},
                            {"type": "text", "text": question}
                        ],
                    }]
                    conversations.append(conv)

                processor.tokenizer.padding_side = "left" # need for proper inference
                inputs = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to("cuda")
            else:
                if not chain_of_thought:
                    prompts = [
                        f"USER: <image>\n{question}\nASSISTANT:"
                        for question in questions
                    ]
                else:
                    prompts = [
                        f"USER: <image>\n{question}\nASSISTANT: Let's think step by step."
                        for question in questions
                    ]
            if not "llava-onevision" in model_id:
                inputs = processor(
                    text=prompts, images=images, padding=True, return_tensors="pt"
                ).to("cuda")
            
            output = model.generate(
                **inputs, max_new_tokens=64 if not chain_of_thought else 512
            )
            generated_text = processor.batch_decode(output, skip_special_tokens=True)

            # TDA start
            pixel_values = inputs["pixel_values"]
            image_features = extract_image_features(model, processor, pixel_values)  # (B, D)
            logits = image_features @ W_c.T  # (B, C)
            probs = F.softmax(logits, dim=-1)
            entropy_vals = tda_entropy(logits)
            final_logits = logits.clone()

            # print(image_features.shape, W_c.T.shape, probs.shape)
            # exit(0)

            clip_style_preds, tda_preds = [], []
            for j, (feat, prob, ent) in enumerate(zip(image_features, probs, entropy_vals)):
                cls_idx = prob.argmax().item()
                clip_style_preds.append(classes[cls_idx])

                update_tda_cache(tda_positive_cache, cls_idx, feat, ent.item(), TDA_CFG["cap_pos"])

                if TDA_CFG["tau_low"] < ent < TDA_CFG["tau_high"]:
                    neg_mask = (prob < TDA_CFG["pl"]).float() * -1
                    update_tda_cache(tda_negative_cache, cls_idx, feat, ent.item(), TDA_CFG["cap_neg"], neg_mask)

                pos_logit = compute_tda_logits(feat.unsqueeze(0), tda_positive_cache, TDA_CFG["alpha_pos"], TDA_CFG["beta_pos"], len(classes))
                neg_logit = compute_tda_logits(feat.unsqueeze(0), tda_negative_cache, TDA_CFG["alpha_neg"], TDA_CFG["beta_neg"], len(classes), is_negative=True)

                final_logits[j] += pos_logit[0] - neg_logit[0]

                tda_preds.append(classes[final_logits[j].argmax().item()])
            # TDA end

            for item, text, clip_style_pred, tda_pred in zip(batch, generated_text, clip_style_preds, tda_preds):
                item["output"] = text
                if "mistral" in model_id:
                    item["pred"] = text.split("[/INST]")[-1].strip()
                elif "blip" in model_id:
                    item["pred"] = text.split("Answer:")[-1].strip()
                elif "llava-onevision" in model_id:
                    item["pred"] = text.split("assistant")[-1].strip()
                else:
                    item["pred"] = text.split("ASSISTANT:")[-1].strip()
                item["pred_clip_style"] = clip_style_pred
                item["pred_tda"] = tda_pred
                f.write(json.dumps(item) + "\n")
                f.flush()


def main_clip(
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    batch_size,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "eva" in model_id.lower():
        pretrained = "eva_clip"
        model, _, preprocess = create_model_and_transforms(
            model_id, pretrained, force_custom_clip=True
        )
        tokenizer = get_tokenizer(model_id)
        model = model.to(device)
    else:
        model, preprocess = clip.load(model_id, device=device)
        tokenizer = clip.tokenize

    data = [json.loads(line) for line in open(data_path)]
    data = [item for item in data if item["split"] == split]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    with open(output_path, "w") as f:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            images = [
                preprocess(Image.open(item["image"]).convert("RGB")).unsqueeze(0)
                for item in batch
            ]
            image_tensor = torch.cat(images).to(device)

            text_descriptions = [f"A photo of a {cls}" for cls in classes]
            text_tokens = tokenizer(text_descriptions).to(device)

            if including_label:
                choices = []
                for item in batch:
                    label = item["label"]
                    sample_seed = hash(f"{seed}_{item['image']}") % (2**32)
                    random.seed(sample_seed)  # Set seed specific to this sample
                    other_choices = random.sample(
                        sorted(list(set(classes) - set([label]))), n_labels - 1
                    )
                    shuffled_choices = [label] + other_choices

                    random.shuffle(shuffled_choices)
                    indices = [classes.index(choice) for choice in shuffled_choices]
                    choices.append(indices)
            else:
                choices = None

            with torch.no_grad():
                image_features = F.normalize(model.encode_image(image_tensor), dim=-1)
                text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

                similarities = image_features @ text_features.T

            for idx in range(len(batch)):
                item = batch[idx]
                if including_label:
                    choice = choices[idx]
                    pred = similarities[idx][choice].argmax()
                    predicted_class = classes[choice[pred.item()]]
                else:
                    predicted_class = classes[similarities[idx].argmax().item()]

                item["choices"] = (
                    [classes[class_idx] for class_idx in choices[idx]]
                    if including_label
                    else None
                )
                item["pred"] = predicted_class
                f.write(json.dumps(item) + "\n")
                f.flush()


def main_supervised(
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    batch_size,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vit_b_16(pretrained=True).to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = [json.loads(line) for line in open(data_path)]
    data = [item for item in data if item["split"] == split]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    with open(output_path, "w") as f:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            images = [
                preprocess(Image.open(item["image"]).convert("RGB")).unsqueeze(0)
                for item in batch
            ]
            image_tensor = torch.cat(images).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                _, preds = torch.max(outputs, 1)

            for idx in range(len(batch)):
                item = batch[idx]
                predicted_class = classes[preds[idx].item()]

                if including_label:
                    label = item["label"]
                    sample_seed = hash(f"{seed}_{item['image']}") % (2**32)
                    random.seed(sample_seed)  # Set seed specific to this sample
                    other_choices = random.sample(
                        sorted(list(set(classes) - set([label]))), n_labels - 1
                    )
                    choices = [label] + other_choices
                    random.shuffle(choices)
                    item["choices"] = choices
                else:
                    item["choices"] = None

                item["pred"] = predicted_class
                f.write(json.dumps(item) + "\n")
                f.flush()


@click.command()
@click.option("--method", default="vlm")
@click.option("--model_id", default="ViT-B/32")
@click.option("--data_path", default="../data/imagenet.jsonl")
@click.option("--class_path", default="../data/imagenet_classes.json")
@click.option("--split", default="valid")
@click.option("--seed", default=1234)
@click.option("--output_path", default="outputs.jsonl")
@click.option("--including_label", default=False)
@click.option("--n_labels", default=1000)
@click.option("--chain_of_thought", default=False)
@click.option("--batch_size", default=8)
@click.option("--fixed_order", default=False)
@click.option("--prompt", default=None)
def entry(
    method,
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    chain_of_thought,
    batch_size,
    fixed_order,
    prompt,
):
    if method == "vlm":
        main(
            model_id,
            data_path,
            class_path,
            split,
            seed,
            output_path,
            including_label,
            n_labels,
            chain_of_thought,
            batch_size,
            fixed_order,
            prompt,
        )
    elif method == "clip":
        main_clip(
            model_id,
            data_path,
            class_path,
            split,
            seed,
            output_path,
            including_label,
            n_labels,
            batch_size,
        )
    elif method == "supervised":
        main_supervised(
            model_id,
            data_path,
            class_path,
            split,
            seed,
            output_path,
            including_label,
            n_labels,
            batch_size,
        )
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    entry()
