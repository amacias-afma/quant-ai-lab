# src/models/alpha_rag/llm_client.py
import os
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

class LLMClient:
    def __init__(self, provider="vertex_ai"):
        self.provider = provider
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
        self.location = os.environ.get("GCP_REGION", "us-central1") # Default region
        self.model = None

        if self.provider == "vertex_ai":
            try:
                print(f"☁️ Initializing Vertex AI (Gemini) in {self.project_id}...")
                vertexai.init(project=self.project_id, location=self.location)
                
                # 'gemini-1.5-flash' is faster and cheaper for high-frequency analysis
                # Use 'gemini-1.5-pro' for complex reasoning
                self.model = GenerativeModel("gemini-1.5-flash")
                print("✅ Vertex AI Client Ready.")
            except Exception as e:
                print(f"⚠️ Vertex AI Init Failed: {e}")
                print("   -> Falling back to 'mock' provider.")
                self.provider = "mock"
    
    def analyze_context(self, context_text, question):
        """
        Sends relevant news (context) to the LLM to get a trading signal.
        """
        prompt = f"""
        You are a Senior Financial Analyst for a Quant Fund.
        Your job is to evaluate market sentiment based STRICTLY on the provided news context.

        --- CONTEXT (NEWS CLIPPINGS) ---
        {context_text}
        --------------------------------

        USER QUESTION: {question}

        INSTRUCTIONS:
        1. Analyze the news for concrete financial impact (revenue, regulations, product launches).
        2. Ignore vague marketing fluff.
        3. Determine a Trading Signal: BULLISH, BEARISH, or NEUTRAL.
        4. Provide a 1-sentence reasoning.

        OUTPUT FORMAT:
        SIGNAL: [BULLISH/BEARISH/NEUTRAL]
        REASON: [Your concise analysis]
        """
        
        if self.provider == "mock":
            return self._mock_response(context_text)
        
        return self._vertex_response(prompt)

    def _vertex_response(self, prompt):
        """Calls Google Gemini API"""
        try:
            # Safety settings to prevent the model from blocking financial "violence" (e.g., "market crash")
            safety_config = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }

            response = self.model.generate_content(
                prompt,
                safety_settings=safety_config,
                generation_config={"temperature": 0.2, "max_output_tokens": 150} # Low temp for determinism
            )
            return response.text.strip()
            
        except Exception as e:
            return f"Error calling Vertex AI: {str(e)}"

    def _mock_response(self, text):
        """Mock response for local testing or fallback"""
        return f"[MOCK LLM] I analyzed {len(text.split())} words. The signal appears NEUTRAL based on text volatility."