from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

class PegasusParaphraser:
    """
    A utility class for generating paraphrased versions of a given sentence
    using the PEGASUS paraphrasing model (tuner007/pegasus_paraphrase).
    """

    def __init__(self, 
                 model_name: str = "tuner007/pegasus_paraphrase",
                 device: str = None):
        """
        Initialize the PEGASUS paraphraser.

        Args:
            model_name (str): HuggingFace model name for PEGASUS paraphrasing.
            device (str): Optional. "cuda" or "cpu". If None, select automatically.
        """
        # Automatically choose GPU if available, otherwise use CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model into the selected device
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def paraphrase(self, 
                   sentence: str, 
                   num_return_sequences: int = 5, 
                   num_beams: int = 10,
                   max_length: int = 60,
                   temperature: float = 1.5):
        """
        Generate paraphrased versions of a given sentence.

        Args:
            sentence (str): The input sentence to be paraphrased.
            num_return_sequences (int): Number of paraphrased outputs to generate.
            num_beams (int): Beam search width for diverse generation.
            max_length (int): Maximum length of generated sentences.
            temperature (float): Sampling temperature to control creativity.
        
        Returns:
            List[str]: A list of paraphrased sentences.
        """

        # Encode the input sentence into model-ready tensors
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=max_length
        ).to(self.device)

        # Generate paraphrases using the PEGASUS model
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature
        )

        # Decode model outputs back to natural language text
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    

    
    def __call__(self, sentence: str, **kwargs):
        """
        Allows the instance to be called like a function.
        Equivalent to calling paraphrase() directly.

        Example:
            paraphraser("This is a sentence.")
        """
        return self.paraphrase(sentence, **kwargs)
