import logging
import numpy as np
from typing import List, Tuple, Callable
from llama_cpp import Llama
import nltk
import ssl
import string

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelLoader:
    def __init__(self, model_path: str):
        try:
            self.model = Llama(model_path=model_path)
            logging.debug(f"Successfully loaded Qwen model from: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load the model from {model_path}: {str(e)}")
            raise e

class SolutionGenerator:
    def __init__(self, model: Llama):
        self.model = model

    def generate_output(self, prompt: str, max_length: int = 1024, temperature: float = 0.1, top_p: float = 0.8) -> str:
        chat_prompt = f"system\nUser\n{prompt}\nassistant\n"
        
        try:
            outputs = self.model(chat_prompt, max_tokens=max_length, temperature=temperature, top_p=top_p)
            response = outputs.get("choices", [{}])[0].get("text", "")
            
            if not response.strip():
                logging.warning("Empty response received from the model.")
                return "No valid response generated."
                
            return response
        except Exception as e:
            logging.error(f"Error generating output: {str(e)}")
            return "An error occurred during generation."

class AlgorithmOfThoughts:
    def __init__(self, generator: SolutionGenerator):
        self.generator = generator

    def generate_output_with_refinement(self, prompt: str, max_attempts: int = 6) -> Tuple[str, List[str]]:
        solutions = []

        for i, fixed_temperature in enumerate(np.linspace(0.01, 0.1, 10)):
            logging.debug(f"Initial Attempt {i+1} with temperature {fixed_temperature}")
            solution = self.generator.generate_output(prompt, temperature=fixed_temperature, top_p=0.95)
            logging.debug(f"Proposed solution (Initial Attempt {i+1}): {solution[:1000]}...")
            if solution and solution != "No valid response generated." and solution != "An error occurred during generation.":
                solutions.append(solution)

        for attempt in range(max_attempts):
            temperature = np.random.uniform(0.4, 0.95)
            top_p = np.random.uniform(0.8, 0.95)

            logging.debug(f"Refinement Attempt {attempt + 1} with temperature {temperature} and top_p {top_p}")

            prompt_variation = self.get_prompt_variation(prompt, solutions)
            solution = self.generator.generate_output(prompt_variation, temperature=temperature, top_p=top_p)

            logging.debug(f"Proposed solution (Refinement Attempt {attempt + 1}): {solution[:1000]}...")
            if solution and solution != "No valid response generated." and solution != "An error occurred during generation.":
                solutions.append(solution)

        if not solutions:
            logging.warning("No valid solutions were generated.")
            return "No valid solution generated.", []

        final_solution = self.combine_solutions(solutions)

        if not final_solution.strip():
            logging.warning("Failed to generate a valid final solution.")
            return "No valid solution generated.", solutions

        logging.debug(f"Final mixed solution: {final_solution[:1000]}...")
        return final_solution, solutions

    def get_prompt_variation(self, prompt: str, solutions: List[str]) -> str:
        if not solutions:
            return prompt

        previous_solution = solutions[-1]
        feedback = self.generate_feedback(previous_solution)

        refined_prompt = f"{prompt}\nPrevious solution:\n{previous_solution[:200]}\nFeedback:\n{feedback}\nPlease refine the solution."
        
        return refined_prompt

    def generate_feedback(self, solution: str) -> str:
        feedback = []

        if "introduction" not in solution.lower():
            feedback.append("Missing introduction.")
        if "conclusion" not in solution.lower():
            feedback.append("Missing conclusion.")
        if "step-by-step approach" not in solution.lower():
            feedback.append("Lacks a step-by-step approach.")
        if "examples" not in solution.lower():
            feedback.append("Lacks practical examples.")
        if "real-world applications" not in solution.lower():
            feedback.append("Does not discuss real-world applications.")

        if "definition" not in solution.lower():
            feedback.append("Missing definitions for key terms.")
        if "assumptions" not in solution.lower():
            feedback.append("Lacks assumptions and limitations.")
        if "validation" not in solution.lower():
            feedback.append("Lacks validation or testing steps.")
        if "consistency" not in solution.lower():
            feedback.append("Inconsistent terminology or logic.")

        if "jargon" in solution.lower():
            feedback.append("Uses unnecessary jargon.")
        if "sentence structure" not in solution.lower():
            feedback.append("Improper sentence structure.")
        if "paragraph flow" not in solution.lower():
            feedback.append("Poor paragraph flow.")
        if "clarity" not in solution.lower():
            feedback.append("Could be more clear and concise.")

        if "depth" not in solution.lower():
            feedback.append("Lacks depth in analysis or explanation.")
        if "breadth" not in solution.lower():
            feedback.append("Lacks breadth in coverage or examples.")

        return " ".join(feedback) if feedback else "The solution is mostly correct, just refine for clarity."

    def combine_solutions(self, solutions: List[str]) -> str:
        """Intelligently combine the best parts of multiple solutions."""
        if not solutions:
            return "No solutions were generated."

        # Preprocess solutions
        preprocessed_solutions = [preprocess(sol) for sol in solutions]
        
        # Find the most similar pair of solutions
        max_similarity = 0
        best_pair_index = (0, 1)  # Default to first two solutions if no better pair is found
        for i in range(len(preprocessed_solutions)):
            for j in range(i + 1, len(preprocessed_solutions)):
                similarity = tfidf_similarity(preprocessed_solutions[i], preprocessed_solutions[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pair_index = (i, j)

        # Combine the best pair of solutions
        base_solution = solutions[best_pair_index[0]]
        additional_solution = solutions[best_pair_index[1]]
        
        # Split solutions into sections and merge the best parts
        sections = [
            "introduction",
            "identify financial needs",
            "create a budget",
            "investment strategies",
            "risk management",
            "conclusion"
        ]
        
        best_solution_parts = {section: "" for section in sections}
        
        for sol in [base_solution, additional_solution]:
            sol_lower = sol.lower()
            
            for i, section in enumerate(sections):
                start_index = sol_lower.find(section)
                if start_index != -1:
                    end_index = len(sol)
                    for next_section in sections[i+1:]:
                        next_index = sol_lower.find(next_section)
                        if next_index != -1:
                            end_index = next_index
                            break
                    section_content = sol[start_index:end_index].strip()
                    if len(section_content) > len(best_solution_parts[section]):
                        best_solution_parts[section] = section_content

        # Combine the best parts into a coherent final solution
        final_solution = "\n\n".join(best_solution_parts[section] for section in sections if best_solution_parts[section])

        # If no structured sections were found, return the longest solution
        if not final_solution.strip():
            return max(solutions, key=len)

        return final_solution.strip()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() or word in string.punctuation or word in stop_words]
    return ' '.join(filtered_tokens)

def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_similarities = tfidf_matrix.dot(tfidf_matrix.T).toarray()
    return cosine_similarities[0][1]

class QwenAoTSolver:
    def __init__(self, model_path: str, feedback_fn: Callable[[str], str] = None, prompt_variation_fn: Callable[[str, List[str]], str] = None):
        self.model_loader = ModelLoader(model_path)
        self.solution_generator = SolutionGenerator(self.model_loader.model)
        self.aot = AlgorithmOfThoughts(self.solution_generator)
        self.feedback_fn = feedback_fn or self.default_feedback_fn
        self.prompt_variation_fn = prompt_variation_fn or self.default_prompt_variation_fn

    def generate_output(self, prompt: str, max_length: int = 1024, temperature: float = 0.1, top_p: float = 0.8) -> str:
        return self.solution_generator.generate_output(prompt, max_length, temperature, top_p)

    def algorithm_of_thoughts(self, problem: str, max_attempts: int = 6) -> Tuple[str, List[str]]:
        return self.aot.generate_output_with_refinement(problem, max_attempts)

    @staticmethod
    def default_feedback_fn(solution: str) -> str:
        feedback = []
        
        if "introduction" not in solution.lower():
            feedback.append("Missing introduction or context setting.")
        
        if "main concepts" not in solution.lower():
            feedback.append("Missing explanation of main concepts.")
        
        if "strategies" not in solution.lower():
            feedback.append("Missing specific strategies or methods.")
        
        if "risks" not in solution.lower() and "considerations" not in solution.lower():
            feedback.append("Missing discussion of risks or important considerations.")
        
        if "conclusion" not in solution.lower():
            feedback.append("Missing conclusion or summary.")
        
        if len(solution.split()) < 100:
            feedback.append("Solution seems too brief. Please elaborate more.")
        
        if len(solution.split()) > 500:
            feedback.append("Solution might be too verbose. Consider being more concise.")
        
        return " ".join(feedback) if feedback else "The solution is comprehensive, just refine for clarity if needed."

    @staticmethod
    def default_prompt_variation_fn(prompt: str, solutions: List[str]) -> str:
        if not solutions:
            return prompt

        previous_solution = solutions[-1]
        feedback = QwenAoTSolver.default_feedback_fn(previous_solution)

        refined_prompt = f"{prompt}\nPrevious solution:\n{previous_solution[:200]}\nFeedback:\n{feedback}\nPlease refine the solution.\nContext: Provide a detailed explanation of each step."
        
        return refined_prompt

def main():
    model_path = "/Users/davidblum/Downloads/predfoot002/qwen1_5-0_5b-chat-q2_k.gguf"
    
    def truncate_prompt(prompt, max_tokens):
        tokens = word_tokenize(prompt)
        truncated_tokens = tokens[:max_tokens]
        return ' '.join(truncated_tokens)

    try:
        solver = QwenAoTSolver(model_path)
    except Exception as e:
        logging.error(f"Exiting due to failure in model initialization: {str(e)}")
        return

    problem = "how to raise a dog in a small apartment from which I am absent all day ?"
    #problem = "How to use crypto to become financially independent through strategic investments and risk management?"
    max_tokens = 1024
    truncated_problem = truncate_prompt(problem, max_tokens)
    
    logging.info(f"Starting Algorithm of Thoughts with problem: {truncated_problem}")
    
    try:
        best_solution, all_solutions = solver.algorithm_of_thoughts(truncated_problem, max_attempts=6)
    except Exception as e:
        logging.error(f"Error during Algorithm of Thoughts execution: {str(e)}")
        return

    if not all_solutions:
        logging.warning("No solutions were generated.")
    else:
        for i, solution in enumerate(all_solutions):
            logging.info(f"Solution {i+1}: {solution[:500]}...")  # Print first 500 characters of each solution

    logging.info(f"Final solution: {best_solution[:1000]}...")  # Print first 1000 characters of the final solution
    logging.info(f"Generated {len(all_solutions)} solutions in total.")

    if best_solution == "No valid solution generated.":
        logging.error("Failed to generate a valid solution. Please check the logs for more details.")
    else:
        logging.info("Successfully generated a valid solution.")

if __name__ == "__main__":
    main()
