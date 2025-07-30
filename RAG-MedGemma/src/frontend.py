import gradio as gr

def gradio_ui(consult) -> None:
    '''
    Create a Gradio UI for the MedGamma model.
    Args:
        consult (function): The function to call for consulting the MedGamma model.
    Returns:
        None
    '''
    theme = gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
        radius_size="sm",
    )

    with gr.Blocks(theme=theme) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("""
                ### Upload Image and Files
                Upload an image or files to provide context for the AI assistant.
                Supported file types: .txt, .pdf, .docx, .csv, .pptx.
                            """)
                image_input = gr.Image(
                    type="pil", 
                    label="Upload Image (optional)", 
                    show_label=True, 
                    sources=['upload'], 
                    height=200
                )
                file_input = gr.File(
                    label="Upload File(s) (optional)", 
                    file_count="multiple",
                    show_label=True, 
                    file_types=[".txt", ".pdf", ".docx", ".csv", ".pptx"], 
                    height=200
                )
                gr.Markdown("""
                ### Settings
                Adjust the settings below to customize the response generation.
                            """)
                system_prompt = gr.Textbox(
                    label="System Prompt", 
                    value="You are an expert general practitioner. Be concise and empathetic in all your responses."
                    )
                max_tokens = gr.Slider(
                    label="Max New Tokens", 
                    minimum=100, maximum=2000, 
                    value=1000
                    )
                top_k = gr.Slider(
                    label="Top K", 
                    minimum=1, maximum=100, 
                    value=50,
                    step=1
                    )
                temperature = gr.Slider(
                    label="Temperature", 
                    minimum=0.0, maximum=1.0, 
                    value=0.7, 
                    step=0.01
                )
            with gr.Column(scale=3):
                chat = gr.ChatInterface(
                    fn=consult,
                    additional_inputs=[
                        system_prompt, 
                        max_tokens, 
                        top_k, 
                        temperature, 
                        image_input, 
                        file_input
                        ],
                    type="messages",
                    fill_height=True,
                    analytics_enabled=False,
                    title = "AI Medical Assistant"
                )

        demo.launch(debug=True)
