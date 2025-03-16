import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sympy import sympify, simplify
from pylatexenc import latex2text

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
EQUALITY_TEMPLATE = """
Look at the following two expressions and judge whether they are equivalent as response to math problems. 
Answer with only "Yes" or "No".

Expression 1: \"{expr1}\"\n
Expression 2: \"{expr2}\"\n

Answer:
""".strip()

def extract_answer(response_text: str) -> str: # THIS DOES NOT WORK ON QWEN model
    """Extracts the answer using regex from model response."""
    match = re.search(ANSWER_PATTERN, response_text)
    return match.group(1).strip() if match else None

# TODO: Implement a real answer extractor

def extract_final_answer(pred_answer):
    """
    Extracts the final mathematical answer from the model's response, handling different formats.
    Always returns the LAST match found.
    """
    def _parse_latex(expr: str) -> str:
        """Attempts to parse latex to an expression sympy can read."""
        expr = expr.replace("\\tfrac", "\\frac")
        expr = expr.replace("\\dfrac", "\\frac")
        expr = expr.replace("\\frac", " \\frac")
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)

        # Replace the specific characters that this parser uses.
        expr = expr.replace("√", "sqrt")
        expr = expr.replace("π", "pi")
        expr = expr.replace("∞", "inf")
        expr = expr.replace("∪", "U")
        expr = expr.replace("·", "*")
        expr = expr.replace("×", "*")
        expr = expr.replace("^\\circ", "")

        return expr.strip()

    pred_answer = pred_answer.strip()

    # # Step 1: Extract LAST boxed LaTeX answer (handling nested braces)
    # boxed_matches = re.findall(r"\\boxed\s*{((?:[^{}]+|{[^{}]*})*)}", pred_answer, re.DOTALL)
    # if boxed_matches:
    #     extracted = boxed_matches[-1].strip()  # Always return the last boxed match
    #     return _parse_latex(extracted)  # Convert LaTeX to readable math format

    pred_answer = pred_answer.strip()

    # --- Step 1: Manually find and extract the LAST \boxed{...} content ---
    # We'll do our own brace matching to handle nested braces properly.
    all_boxed = []
    start_index = 0
    while True:
        # Find the next occurrence of '\boxed{'
        pos = pred_answer.find(r'\boxed{', start_index)
        if pos == -1:
            break  # No more occurrences
        pos_open_brace = pos + len(r'\boxed{')
        
        brace_count = 1
        i = pos_open_brace
        
        # Walk forward until we've matched all braces
        while i < len(pred_answer) and brace_count > 0:
            if pred_answer[i] == '{':
                brace_count += 1
            elif pred_answer[i] == '}':
                brace_count -= 1
            i += 1
        
        # If brace_count returned to 0, i-1 is the position of the matching '}'.
        if brace_count == 0:
            content = pred_answer[pos_open_brace : i - 1]
            all_boxed.append(content)
            # Move start_index forward to search for any later \boxed{}
            start_index = i
        else:
            # Mismatched braces; stop
            break

    if all_boxed:
        # Return the LAST one found
        extracted = all_boxed[-1].strip()
        return _parse_latex(extracted)
    
    # Step 2: Look for LAST output block (```output ... ```)
    output_matches = re.findall(r"```output\n([\s\S]+?)\n```", pred_answer)
    if output_matches:
        return output_matches[-1].strip()  # Return the last output block found
    
    # Step 3: Try extracting LAST math expression after "Answer is:" (case insensitive)
    answer_matches = []
    ANSWER_PATTERNS = [
        r"(?i)answer\s*(is|:)\s*([^\n]+)",  # Matches "Answer is:" or "Answer:"
        r"(?i)Thus, the final answer is:\s*([^\n]+)",  # Matches variations of final answer
        r"(?i)Final answer:\s*([^\n]+)"  # Matches "Final answer:"
    ]
    
    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, pred_answer)
        if matches:
            answer_matches.extend(matches)

    if answer_matches:
        extracted = answer_matches[-1][-1].strip()  # Extract the last found answer
        return extracted

    # Step 4: Extract LAST math expression inside \[ \] or $$ $$ for LaTeX block equations
    math_expr_matches = re.findall(r"\$\$([\s\S]+?)\$\$|\[(.*?)\]", pred_answer, re.DOTALL)
    if math_expr_matches:
        last_match = math_expr_matches[-1]
        extracted_expr = last_match[0] if last_match[0] else last_match[1]
        return extracted_expr.strip()  # Return the last math expression found

    return ""  # No valid answer found



def normalize_expression(expression: str) -> str:
    """
    Cleans and normalizes expressions for better matching.
    Removes LaTeX markers like \boxed and spaces.
    """
    if expression is None:
        return ""
    return (
        expression.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
        .replace(" ", "")
    )


class MistralEqualityChecker:
    """Uses `Mistral-7B-Instruct` as a lightweight model to check if two mathematical expressions are equivalent."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def check(self, expr1: str, expr2: str) -> bool:
        """Checks math equivalence using a lightweight model"""
        prompt = EQUALITY_TEMPLATE.format(expr1=expr1, expr2=expr2)


        # make sure both expr1 and expr2 are non-empty strings
        if not expr1 or not expr2:
            print("Both expressions must be non-empty strings.")
            return False

        # apply_chat_template
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "Judge whether the following expressions are equivalent."},
            {"role": "user", "content": prompt}],
            return_tensors="pt",
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"{'='*10} Checking Equivalence {'='*10}")
        print(f"Expression 1: {expr1} vs. Expression 2: {expr2}")
        input_ids = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = self.model.generate(input_ids.input_ids, max_new_tokens=32, pad_token_id=self.tokenizer.eos_token_id)

        # remove input prompt (input_ids) from response
        generated_tokens = output_ids[0][input_ids.input_ids.shape[-1]:]
        # generated_tokens = output_ids[0]

        # Decode the generated tokens
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()


        print(f"Response: {response}")
        return "yes" in response, response

def check_equality(expr1: str, expr2: str, mistral_checker: MistralEqualityChecker) -> bool:
    """Checks if two mathematical expressions are equivalent using both SymPy and Mistral-7B."""
    try:
        expr1_normalized = normalize_expression(expr1)
        expr2_normalized = normalize_expression(expr2)

        if expr1_normalized == expr2_normalized:
            return True  # Direct match

        # Convert to SymPy expressions
        try:
            sympy_expr1 = sympify(expr1_normalized)
            sympy_expr2 = sympify(expr2_normalized)

            if simplify(sympy_expr1 - sympy_expr2) == 0:
                return True  # Exact symbolic match
        except Exception:
            pass  # Ignore SymPy errors and fall back to Mistral

        # Fallback: Use Mistral for fuzzy checking
        return mistral_checker.check(expr1_normalized, expr2_normalized)

    except Exception as e:
        print(f"Error checking equality: {e}")
        return False  # If parsing fails, return False


