import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.utils import io
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
with io.capture_output() as captured:
    tokenizer = AutoTokenizer.from_pretrained("beyoru/informatic_merged_full_training")
    model = AutoModelForCausalLM.from_pretrained(
        "beyoru/informatic_merged_full_training",
        load_in_4bit=True,
    )

# Function to generate a multiple-choice question
def generate_mcq(context, max_new_tokens, temperature):
    # Create the prompt with the context only (remove the second placeholder)
    alpaca_prompt = """Sau đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với hướng dẫn và ngữ cảnh. Hãy viết một phản hồi là một câu hỏi trắc nghiệm và cung cấp 4 lựa chọn đáp án khác nhau. Hãy chắc chắn rằng mỗi đáp án đều khác biệt, và xác định rõ đáp án đúng.
    ### Ngữ cảnh
    {}
    ### Phản hồi
    """.format(context)

    # Tokenize the prompt
    inputs = tokenizer(alpaca_prompt, return_tensors="pt")

    prompt = alpaca_prompt.format(context)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=255)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "")

    return answer

# Gradio interface
iface = gr.Interface(
    fn=generate_mcq,
    inputs=[
        gr.Textbox(label="Ngữ cảnh (Context)", placeholder="Nhập ngữ cảnh ở đây...", lines=3),
        gr.Slider(label="Số lượng từ mới tối đa (Max New Tokens)", minimum=1, maximum=300, value=150, step=1),
        gr.Slider(label="Nhiệt độ (Temperature)", minimum=0.0, maximum=1.0, value=1.0, step=0.1),
    ],
    outputs=gr.Textbox(label="Câu hỏi trắc nghiệm (MCQ)", lines=5),
    title="Gradio Chatbot for MCQ Generation",
    description="Nhập ngữ cảnh để tạo một câu hỏi trắc nghiệm",
)

# Launch the Gradio interface with sharing enabled
iface.launch(share=True, debug=True)
