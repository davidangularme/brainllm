from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import threading
import gc
import logging
import numpy as np
from typing import List, Tuple
import requests
import re
from bs4 import BeautifulSoup
import html
import string
from collections import Counter

class AlgorithmOfThoughts:
    def __init__(self, model, tokenizer, max_attempts=3, temperature_range=(0.3, 0.7)):
        self.model = model
        self.tokenizer = tokenizer
        self.max_attempts = max_attempts
        self.temp_range = temperature_range

    def generate_solution(self, question: str, max_length: int = 256) -> Tuple[str, List[str]]:
        """
        Main AoT implementation with three phases:
        1. Initial generation
        2. Exploration
        3. Refinement
        """
        solutions = []
        
        # Phase 1: Initial Generation
        initial_solution = self._generate_initial_solution(question, max_length)
        solutions.append(initial_solution)
        
        # Phase 2: Exploration
        explored_solutions = self._explore_solutions(question, max_length)
        solutions.extend(explored_solutions)
        
        # Phase 3: Refinement
        refined_solution = self._refine_solutions(solutions, question, max_length)
        solutions.append(refined_solution)
        
        # Select best solution
        final_solution = self._select_best_solution(solutions)
        return final_solution, solutions

    def _score_solution(self, solution: str) -> float:
        """
        Sophisticated scoring system based on multiple metrics
        """
        if not solution or len(solution.strip()) == 0:
            return 0.0

        # 1. Lexical Analysis (25% of total score)
        def calculate_lexical_score():
            translator = str.maketrans('', '', string.punctuation)
            normalized_text = solution.translate(translator).lower()
            words = normalized_text.split()
            word_count = len(words)
            unique_words = len(set(words))
            ttr = unique_words / word_count if word_count > 0 else 0
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            sophisticated_words = sum(1 for word in words if len(word) > 6)
            vocab_sophistication = sophisticated_words / word_count if word_count > 0 else 0
            return (ttr * 0.4 + vocab_sophistication * 0.4 + (avg_word_length/10) * 0.2) * 25

        # 2. Structural Analysis (25% of total score)
        def calculate_structural_score():
            sentences = re.split(r'[.!?]+', solution)
            sentences = [s.strip() for s in sentences if s.strip()]
            paragraphs = [p.strip() for p in solution.split('\n\n') if p.strip()]
            avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
            sentence_length_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
            para_length_variation = np.std([len(p.split()) for p in paragraphs]) if paragraphs else 0
            para_count_score = min(len(paragraphs) / 3, 1.0)
            has_structure = bool(re.search(r'^\s*[\d\-\*\•]\s+', solution, re.MULTILINE))
            return (sentence_length_score * 0.4 + para_count_score * 0.3 + 
                    (1.0 - para_length_variation/100) * 0.2 + 
                    (has_structure * 0.1)) * 25

        # 3. Content Relevance (20% of total score)
        def calculate_relevance_score():
            key_phrases = ["law", "legal", "regulation", "section", "ruling", "court"]
            phrase_count = sum(1 for phrase in key_phrases if phrase in solution.lower())
            has_citations = bool(re.search(r'\(\d{4}\)|\d+/\d+', solution))
            has_legal_refs = bool(re.search(r'section \d+|regulation \d+|law .{3,50}', solution))
            return (min(phrase_count/len(key_phrases), 1.0) * 0.4 + 
                    has_citations * 0.3 + 
                    has_legal_refs * 0.3) * 20

        # 4. Coherence Analysis (15% of total score)
        def calculate_coherence_score():
            sentences = re.split(r'[.!?]+', solution)
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) < 2:
                return 0.0
            transition_words = ["therefore", "thus", "additionally", "however", "conversely", "accordingly"]
            transition_count = sum(1 for word in transition_words if word in solution.lower())
            pronouns = ["it", "they", "these", "this", "those", "which"]
            pronoun_count = sum(1 for pronoun in pronouns if pronoun in solution.lower())
            flow_score = transition_count / (len(sentences) - 1)
            return (min(flow_score, 1.0) * 0.6 + 
                    min(pronoun_count/len(sentences), 1.0) * 0.4) * 15

        # 5. Technical Quality (15% of total score)
        def calculate_technical_score():
            spelling_errors = len(re.findall(r'\b\w+[A-Za-z]\w+\b', solution))
            repeated_words = sum(count > 2 for count in Counter(solution.split()).values())
            consistent_spacing = 1.0 - len(re.findall(r'\s{2,}', solution)) / len(solution)
            consistent_punctuation = 1.0 - len(re.findall(r'[.!?]{2,}', solution)) / len(solution)
            return (max(0, 1.0 - spelling_errors/50) * 0.4 + 
                    max(0, 1.0 - repeated_words/10) * 0.3 +
                    consistent_spacing * 0.15 +
                    consistent_punctuation * 0.15) * 15

        try:
            total_score = (calculate_lexical_score() + 
                         calculate_structural_score() + 
                         calculate_relevance_score() + 
                         calculate_coherence_score() + 
                         calculate_technical_score())
            
            normalized_score = total_score / 100.0
            return normalized_score
            
        except Exception as e:
            logging.error(f"Error in score calculation: {str(e)}")
            return 0.0

    def _generate_initial_solution(self, question: str, max_length: int) -> str:
        return self._generate_text(question, max_length, temperature=0.3, top_p=0.9)

    def _explore_solutions(self, question: str, max_length: int) -> List[str]:
        solutions = []
        temperatures = np.linspace(self.temp_range[0], self.temp_range[1], self.max_attempts)
        
        for temp in temperatures:
            solution = self._generate_text(
                question,
                max_length,
                temperature=temp,
                top_p=np.random.uniform(0.8, 0.95)
            )
            solutions.append(solution)
        
        return solutions

    def _refine_solutions(self, previous_solutions: List[str], question: str, max_length: int) -> str:
        best_solution = self._select_best_solution(previous_solutions)
        refined_prompt = self._create_refinement_prompt(question, best_solution)
        
        return self._generate_text(
            refined_prompt,
            max_length,
            temperature=0.4,
            top_p=0.9
        )

    def _generate_text(self, prompt: str, max_length: int, temperature: float, top_p: float) -> str:
        combined_snippets = get_combined_snippets(prompt, 10)
        full_prompt = PROMPT_TEMPLATE.format(snippets=combined_snippets, question=prompt)
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(full_prompt):].strip()

    def _create_refinement_prompt(self, question: str, previous_solution: str) -> str:
        return f"""Based on the previous answer:
{previous_solution}

Please provide an improved answer to the question:
{question}

Focus on:
1. Accuracy and precision
2. Clarity of explanation
3. Completeness of response
"""

    def _select_best_solution(self, solutions: List[str]) -> str:
        scores = [self._score_solution(solution) for solution in solutions]
        return solutions[np.argmax(scores)]

# Constants and Templates
PROMPT_TEMPLATE = """
Use the following guidelines to answer the question, considering the information gathered from search results:

Relevant search results:
{snippets}

1. Provide a concise and focused answer.
2. Address the question directly.
3. If there is relevant legal information from search results, briefly mention it.
4. If uncertain, state so.
5. No need to provide a list of sources or bibliography at the end.

Question: {question}

Answer:
"""

# Flask application setup and routes remain the same as in your original code
