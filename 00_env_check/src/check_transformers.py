import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print("=== Transformers Check ===")
print(f"Transformers version: {transformers.__version__}")

print("\nImports OK")
print("AutoTokenizer       : OK")
print("AutoModelForCausalLM: OK")
print("pipeline            : OK")