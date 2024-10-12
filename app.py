import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from IPython.utils import io
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


with io.capture_output() as captured:
    tokenizer = AutoTokenizer.from_pretrained("beyoru/informatic_merged_full_training")
    model = AutoModelForCausalLM.from_pretrained(
        "beyoru/informatic_merged_full_training",
        load_in_4bit = True
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        # low_cpu_mem_usage = True
    )

def generate_mcq(context, max_new_tokens, frequency_penalty):
    alpaca_prompt = """Sau đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với với hướng dẫn và ngữ cảnh. Hãy viêt một phản hồi là một câu hỏi trắc nghiệm và cung cấp 4 lựa chọn đáp án khác nhau. Hãy chắc chắn rằng mỗi đáp án đều khác biệt, và xác định rõ đáp án đúng.
    
    ### Ngữ cảnh
    {}
    
    ### Phản hồi
    {}"""

    prompt = alpaca_prompt.format(context, '')
    inputs = tokenizer(prompt, return_tensors="pt")
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        # temperature=temperature,
        # repetition_penalty=frequency_penalty,
        # do_sample=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "")

    return answer

iface = gr.Interface(
    fn=generate_mcq,
    inputs=[
        gr.Textbox(label="Ngữ cảnh (Context)", placeholder="Nhập ngữ cảnh ở đây...", lines=3),
        gr.Slider(label="Số lượng từ mới tối đa (Max New Tokens)", minimum=1, maximum=512, value=255, step=1),
        # gr.Slider(label="Nhiệt độ (Temperature)", minimum=0.0, maximum=1.0, value=0.7, step=0.1),
        # gr.Slider(label="Top-p (Nucleus Sampling)", minimum=0.0, maximum=1.0, value=0.9, step=0.01),
        # gr.Slider(label="Top-k", minimum=1, maximum=100, value=50, step=1),
        # gr.Slider(label="Frequency Penalty", minimum=0.0, maximum=2.0, value=0.5, step=0.1),
        # gr.Slider(label="Presence Penalty", minimum=0.0, maximum=2.0, value=0.0, step=0.1),
    ],
    outputs=gr.Textbox(label="Câu hỏi trắc nghiệm (MCQ)", lines=10),
    title="Informatic",
    description="Nhập ngữ cảnh để tạo một câu hỏi trắc nghiệm",
)

iface.launch(share=True, debug=True)
