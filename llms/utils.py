import argparse
from typing import Any
from transformers import AutoModel, MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from io import BytesIO
import base64
from PIL import Image as BaseImage
try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]


def base642img(base64_str):
    imgdata = base64.b64decode(base64_str)
    return BaseImage.open(BytesIO(imgdata))

def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    model: AutoModel = None
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        # assert isinstance(prompt, str)
        # response = generate_from_huggingface_completion(
        #     prompt=prompt,
        #     model_endpoint=lm_config.gen_config["model_endpoint"],
        #     temperature=lm_config.gen_config["temperature"],
        #     top_p=lm_config.gen_config["top_p"],
        #     stop_sequences=lm_config.gen_config["stop_sequences"],
        #     max_new_tokens=lm_config.gen_config["max_new_tokens"],
        # )
        if "Llama-3.2" in lm_config.model:
            processor = AutoProcessor.from_pretrained(lm_config.model)
            
            prompt_content = []
            imgs = []
            for i in range(len(prompt[0]["content"])):
                if prompt[0]["content"][i]["type"] == "text":
                    prompt_content.append(prompt[0]["content"][i])
                elif prompt[0]["content"][i]["type"] == "image_url":
                    prompt_content.append({"type": "image"})
                    img_base64 = prompt[0]["content"][i]["image_url"]["url"].split(",")[1]
                    imgs.append(base642img(img_base64))
            new_prompt = [{"role": "user", "content": prompt_content}]
            input_text = processor.apply_chat_template(new_prompt, add_generation_prompt=True)
            inputs = processor(imgs if len(imgs) > 0 else None, input_text, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            outputs = model.generate(**inputs, max_new_tokens=2500, temperature=0.0, do_sample=False, top_p=1.0)
            response = processor.decode(outputs[0][len(inputs["input_ids"][0]):])
            return response
            
            
        else:
            raise RuntimeError()
            
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
