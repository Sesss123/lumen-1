import argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../checkpoints/lumen_mistral_finetuned")
    args = parser.parse_args()

    print(f"🚀 Starting vLLM Engine with model {args.model}")
    
    # Initialize vLLM Engine
    # Note: For PeftModels, it's recommended to merge the LoRA weights into the base model first 
    # using export_model.py before serving with vLLM for best performance.
    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=500)
    
    prompts = [
        "ඔබ ගැන කෙටි හැඳින්වීමක් කරන්න.",
        "සීගිරිය ගැන විස්තරයක් දෙන්න."
    ]
    
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}\nOutput: {generated_text}")

if __name__ == "__main__":
    main()
