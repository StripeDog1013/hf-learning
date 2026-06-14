import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from peft import LoraConfig
from peft import get_peft_model_state_dict
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from config import (
    MODEL_NAME,
    DATA_DIR,
    IMAGE_SIZE,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    LORA_RANK,
    LORA_ALPHA,
    LORA_OUTPUT_DIR,
    LOG_DIR,
    SEED,
)

from device import (
    get_device,
    print_device_info,
)

from dataset import load_dataset

from load_pipeline import prepare_train_components

from utils import (
    create_directory,
    initialize_seed,
    print_train_settings,
)


def collate_fn(batch):
    pixel_values = torch.stack(
        [
            item["pixel_values"]
            for item in batch
        ]
    )

    prompts = [
        item["prompt"]
        for item in batch
    ]

    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
    }


def add_lora_to_unet(unet):
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
    )

    unet.add_adapter(lora_config)

    return unet

def get_lora_parameters(unet):
    params = []

    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            params.append(param)
        else:
            param.requires_grad = False

    return params


def encode_prompts(
    tokenizer,
    text_encoder,
    prompts,
    device,
):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            input_ids
        )[0]

    return encoder_hidden_states


def train_one_epoch(
    epoch,
    dataloader,
    components,
    optimizer,
    device,
):
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    unet = components["unet"]
    noise_scheduler = components["noise_scheduler"]

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}",
    )

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(
            device=device,
            dtype=vae.dtype,
        )

        prompts = batch["prompts"]

        with torch.no_grad():
            latents = vae.encode(
                pixel_values
            ).latent_dist.sample()

            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)

        batch_size = latents.shape[0]

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()

        noisy_latents = noise_scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        encoder_hidden_states = encode_prompts(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
            device=device,
        )

        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
        ).sample

        loss = F.mse_loss(
            model_pred.float(),
            noise.float(),
            reduction="mean",
        )

        loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        if (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (step + 1) == len(dataloader)
        ):
            torch.nn.utils.clip_grad_norm_(
                get_lora_parameters(unet),
                MAX_GRAD_NORM,
            )

            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.6f}"
            }
        )

    avg_loss = total_loss / len(dataloader)

    return avg_loss


def save_lora_weights(unet):
    create_directory(LORA_OUTPUT_DIR)

    unet_lora_state_dict = get_peft_model_state_dict(
        unet
    )

    StableDiffusionPipeline.save_lora_weights(
        save_directory=LORA_OUTPUT_DIR,
        unet_lora_layers=unet_lora_state_dict,
    )

    print(f"\nLoRA saved to: {LORA_OUTPUT_DIR}")


def main():
    initialize_seed(SEED)

    print_device_info()

    print_train_settings(
        model_name=MODEL_NAME,
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
    )

    create_directory(LOG_DIR)
    create_directory(LORA_OUTPUT_DIR)

    device = get_device()

    dataset = load_dataset()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    components = prepare_train_components()

    unet = components["unet"]

    unet = add_lora_to_unet(unet)

    lora_params = get_lora_parameters(unet)

    print("\n=== LoRA Parameters ===")
    print(
        f"Trainable LoRA params: "
        f"{sum(p.numel() for p in lora_params):,}"
    )

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=LEARNING_RATE,
    )

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(
            epoch=epoch,
            dataloader=dataloader,
            components=components,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"Average Loss: {avg_loss:.6f}"
        )

    save_lora_weights(unet)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()