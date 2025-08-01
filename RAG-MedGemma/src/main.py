
# Standard library imports
import os
from threading import Thread
from typing import Optional, List, Dict

# Third-party imports
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig, TextIteratorStreamer
from yaml import safe_load

# Local application imports
import RAG
from frontend import gradio_ui

class MedGemma:
    '''
    A class to interact with the MedGamma model for image-text-to-text tasks.
    This class initializes the model, loads it, 
    and provides a method to consult the model with a message and optional history.
    
    Attributes:
        model_id (str): The identifier for the MedGamma model.
        use_quantization (bool): Whether to use quantization for the model.
        pipe (pipeline): The Hugging Face pipeline for image-text-to-text tasks.
    '''
    def __init__(self, 
                 model_variant: str=None, 
                 use_quantization: bool=False
                 ):
        '''
        Initialize the MedGamma model with the specified variant and quantization option.
        Args:
            model_variant (str): The variant of the MedGamma model to use.
            use_quantization (bool): Whether to use quantization for the model.
        '''
        # Load environment variables and login to Hugging Face
        load_dotenv()
        try:
            login(token=os.getenv("HUGGINGFACE_TOKEN"))
        except Exception as e:
            print(f"Error logging in to Hugging Face: {e}")

        self.model_id = f"google/medgemma-{model_variant}"
        self.use_quantization = use_quantization
        self.pipe = self.load_pipeline()

    def load_pipeline(self):
        '''
        Load the MedGamma model pipeline with optional quantization.
        
        Returns:
            pipeline: A Hugging Face pipeline for image-text-to-text tasks.
        '''
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

        if self.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=True,
            )
            
        pipe = pipeline(
        "image-text-to-text",
        model=self.model_id,
        model_kwargs=model_kwargs
            )
        return pipe

    def consult(
        self,
        message: str, 
        history: List[Dict[str, str]], 
        system_prompt: str = None,
        max_new_tokens: int = None,
        top_k: int = None,
        temperature: float = None,
        image: Optional[Image.Image] = None, 
        file: Optional[List] = None
        ):
        '''
        Consult the MedGamma model with a message and optional history, system prompt, image, and file attachments.
        
        Args:
            message (str): The user's message.
            history (List[Dict[str, str]]): Conversation history.
            system_prompt (str, optional): System prompt for the model.
            max_new_tokens (int, optional): Maximum number of new tokens to generate.
            top_k (int, optional): Top-k sampling parameter.
            temperature (float, optional): Temperature for sampling.
            image (Optional[Image.Image], optional): Image to include in the query.
            file (Optional[List], optional): List of files to process.
            
        Returns:
            Generator: Yields the model's response as it streams in.
            '''
        try:
            formatted_history = [
                {
                    "role": turn["role"],
                    "content": [{"type": "text", "text": turn["content"]}]
                }
                for turn in history
            ]

            messages = [
                {"role": "system", "content": [{
                    "type": "text", 
                    "text": system_prompt
                }]},
                *formatted_history
            ]

            # implement RAG for files (pdf, txt, docx, csv, pptx)
            if file:
                for f in file:
                    if f.name.endswith('.pdf'):
                        documents = RAG.extract_text_from_pdf(f.name)
                    elif f.name.endswith('.txt'):
                        with open(f.name, 'r') as txt_file:
                            documents = txt_file.read()
                    elif f.name.endswith('.docx'):
                        documents = RAG.load_docx(f.name)
                    elif f.name.endswith('.csv'):
                        documents = RAG.load_csv(f.name)
                    elif f.name.endswith('.pptx'):
                        documents = RAG.extract_text_from_pptx(f.name)
                    
                    chunks = RAG.split_documents(documents)
                    vectorstore = RAG.embed_chunks(chunks)
                    query = message
                    results = RAG.search_similar_chunks(query, vectorstore)

                    retrieved_texts = [result.page_content for result in results]
                    combined_context = "\n\n".join(retrieved_texts)
                    message = f"Use the following context to answer the question:\n\n{combined_context}\n\n" + message

            user_message = {"role": "user", "content": [{"type": "text", "text": message}]}
            # implement RAG for images
            if image:
                user_message["content"].append({"type": "image", "image": image})
            messages.append(user_message)

            # Set up streaming
            streamer = TextIteratorStreamer(self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)
            input_ids = self.pipe.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
                ).to(self.pipe.model.device)
            
            self.pipe.model.generation_config.do_sample = True
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "streamer": streamer,
                "do_sample": True,
                "temperature": temperature,
                "top_k": top_k
                }

            Thread(target=self.pipe.model.generate, kwargs=generation_kwargs).start()

            output = ""
            # Yield tokens as they stream in
            for token in streamer:
                output += token
                yield output

        except Exception as e:
            yield f"_Error: {str(e)}_"


def main():
    # load configuration from config.yaml
    try:
        with open("config.yaml", "r") as file:
            config = safe_load(file)
            model_variant = config.get("model_variant")
            use_quantization = config.get("use_quantization")
    except ImportError as e:
        print(f"Error loading config.yaml: {e}. Using default settings.")
        model_variant = "4b-it"
        use_quantization = True
        
    # Initialize MedGamma with the specified model variant and quantization option
    medgemma = MedGemma(
        model_variant=model_variant, 
        use_quantization=use_quantization
        )
    # Start the Gradio UI for the MedGamma model
    gradio_ui(medgemma.consult)
    
if __name__ == "__main__":
    main()
