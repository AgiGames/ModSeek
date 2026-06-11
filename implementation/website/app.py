import gradio as gr
import torch

class ModSeek(torch.nn.Module):
    def __init__(self, num_qs: int, num_sols: int):
        super().__init__()
        
        self.num_qs = num_qs
        self.num_sols = num_sols
        self.W_yes = torch.nn.Parameter(torch.randn(1, num_qs, num_sols))
        self.W_no = torch.nn.Parameter(torch.randn(1, num_qs, num_sols))
        self.bias = torch.nn.Parameter(torch.randn(self.num_sols))
    
    def forward(self, answered_qs):        
        answered_qs_dimmed = answered_qs.unsqueeze(-1)
        
        # (B, nq, ns) = (B, nq, 1) * (1, nq, ns) + (B, nq, 1) * (1, nq, ns)
        z_iter = answered_qs_dimmed * self.W_yes + (1 - answered_qs_dimmed) * self.W_no
        z = self.bias + z_iter.sum(dim=1)
        return z

questions = [
    "Is cold start a problem in your system?",
    "Is high computational cost an issue?",
    "Is there a lack of annotated datasets?",
    "Do you face language diversity challenges?",
    "Is limited labeled data a concern?",
    "Do traditional models suffer from poor scalability?",
    "Is your system sample inefficient?",
    "Are there scalability issues in your approach?",
    "Does your model suffer from unstable training?",
    "Is urban traffic congestion a relevant problem in your context?"
]
solutions = [
    "CNN-Based Feature Extraction",
    "Data Augmentation Techniques",
    "Distributed Training Framework",
    "Graph Neural Network Modeling",
    "Matrix Factorization",
    "Multilingual Embeddings",
    "Policy Gradient Optimization",
    "Simulation-Based Training",
    "Spatio-Temporal Learning",
    "Transfer Learning Approach"
]

model = torch.load('mod_seek.pt')
model.eval()

def start():
    return (
        ([], 0),
        gr.update(value=questions[0]),  # question_box
        gr.update(value="", visible=False),  # output_box
        gr.update(visible=True),    # yes_btn
        gr.update(visible=True),    # no_btn
        gr.update(visible=False),   # start_btn
    )

def next_question(state, answer):
    answers, idx = state
    answers = answers + [1 if answer == "Yes" else 0]
    idx += 1

    if idx < len(questions):
        return (
            (answers, idx),
            gr.update(value=questions[idx]),
            gr.update(visible=False),
            gr.update(visible=True),    # yes_btn
            gr.update(visible=True),    # no_btn
            gr.update(visible=False),   # start_btn
        )
    else:
        x = torch.tensor(answers, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
        probs = torch.sigmoid(output.squeeze()).tolist()
        paired = sorted(zip(solutions, probs), key=lambda x: x[1], reverse=True)
        result = "### Recommended Solutions\n\n"
        for sol, p in paired:
            result += f"- **{sol}** → {p:.4f}\n"

        return (
            (answers, idx),
            gr.update(value=result),
            gr.update(visible=False),   # output_box (unused now)
            gr.update(visible=False),   # yes_btn
            gr.update(visible=False),   # no_btn
            gr.update(visible=True),    # start_btn
        )

with gr.Blocks() as demo:
    gr.Markdown("# ModSeek Questionnaire")
    state = gr.State(([], 0))
    question_box = gr.Markdown("")
    output_box = gr.Markdown(visible=False)  # can remove entirely
    start_btn = gr.Button("Start", visible=True)
    yes_btn = gr.Button("Yes", visible=False)
    no_btn = gr.Button("No", visible=False)

    start_btn.click(
        fn=start,
        outputs=[state, question_box, output_box, yes_btn, no_btn, start_btn]
    )
    yes_btn.click(
        fn=lambda s: next_question(s, "Yes"),
        inputs=state,
        outputs=[state, question_box, output_box, yes_btn, no_btn, start_btn]
    )
    no_btn.click(
        fn=lambda s: next_question(s, "No"),
        inputs=state,
        outputs=[state, question_box, output_box, yes_btn, no_btn, start_btn]
    )

demo.launch()