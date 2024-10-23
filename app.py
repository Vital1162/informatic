import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from IPython.utils import io
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_model_and_tokenizer(model_name):
    # Load the model and tokenizer dynamically based on the input model_name
    with io.capture_output() as captured:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True
        )
    return tokenizer, model

def generate_mcq(model_name, context, max_new_tokens, temperature, frequency_penalty):
    tokenizer, model = load_model_and_tokenizer(model_name)
    
    alpaca_prompt = """### Đoạn văn bản
{}

### Phản hổi
{}"""
    
    prompt = alpaca_prompt.format(context, '')
    inputs = tokenizer(prompt, return_tensors="pt")
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=frequency_penalty,
        do_sample=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "")
    
    return answer

iface = gr.Interface(
    fn=generate_mcq,
    inputs=[
        gr.Textbox(label="Tên mô hình (Model Name)", placeholder="Nhập tên mô hình ở đây...", lines=1, value="beyoru/informatic_merged_full_training_dpo"),
        gr.Textbox(label="Ngữ cảnh (Context)", placeholder="Nhập ngữ cảnh ở đây...", lines=3),
        gr.Slider(label="Số lượng từ mới tối đa (Max New Tokens)", minimum=1, maximum=512, value=255, step=1),
        gr.Slider(label="Nhiệt độ (Temperature)", minimum=0.0, maximum=1.0, value=0.7, step=0.1),
        gr.Slider(label="Frequency Penalty", minimum=0.0, maximum=1.0, value=0.5, step=0.1),
    ],
    outputs=gr.Textbox(label="Câu hỏi trắc nghiệm (MCQ)", lines=10),
    title="Informatic",
    description="Nhập tên mô hình và ngữ cảnh để tạo một câu hỏi trắc nghiệm",
)

iface.launch(share=True, debug=True)
