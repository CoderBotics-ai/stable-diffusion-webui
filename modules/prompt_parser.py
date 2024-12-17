from __future__ import annotations

import re
from dataclasses import dataclass
from collections import namedtuple
from typing import List, Dict, Tuple, Union, Optional
import torch
import lark

# Grammar for parsing prompt schedules
PROMPT_GRAMMAR = r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
"""

@dataclass
class PromptSchedule:
    """Represents a scheduled prompt with its timing"""
    step: int
    prompt: str

class ScheduledPromptConditioning(namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])):
    """Represents conditioning for a prompt at a specific step"""
    pass

class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    def __init__(self, 
                 prompts: List[str], 
                 is_negative_prompt: bool = False,
                 width: Optional[int] = None, 
                 height: Optional[int] = None,
                 copy_from: Optional[Union[List[str], 'SdConditioning']] = None):
        super().__init__()
        self.extend(prompts)
        
        if copy_from is None:
            copy_from = prompts
            
        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)

@dataclass
class ComposableScheduledPromptConditioning:
    """Represents a scheduled prompt conditioning with an associated weight"""
    schedules: List[ScheduledPromptConditioning]
    weight: float = 1.0

@dataclass 
class MulticondLearnedConditioning:
    """Container for multiple learned conditionings"""
    shape: Tuple[int, ...]  # needed to send this object to DDIM/PLMS
    batch: List[List[ComposableScheduledPromptConditioning]]

class DictWithShape(dict):
    """Dictionary that maintains a shape property for compatibility"""
    def __init__(self, x: Dict, shape: Optional[Tuple] = None):
        super().__init__()
        self.update(x)

    @property
    def shape(self) -> Tuple:
        return self["crossattn"].shape

# Initialize parser with grammar
schedule_parser = lark.Lark(PROMPT_GRAMMAR)

# Compile regex patterns
re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")
re_attention = re.compile(r"""
\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:
""", re.X)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def get_learned_conditioning_prompt_schedules(
    prompts: List[str], 
    base_steps: int,
    hires_steps: Optional[int] = None,
    use_old_scheduling: bool = False
) -> List[List[PromptSchedule]]:
    """
    Convert prompts into step schedules for learned conditioning.
    
    Args:
        prompts: List of prompt strings to process
        base_steps: Number of base sampling steps
        hires_steps: Optional number of high-res sampling steps
        use_old_scheduling: Whether to use legacy scheduling behavior
        
    Returns:
        List of prompt schedules for each input prompt
    """
    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps: int, tree: lark.Tree) -> List[int]:
        """Collect all step numbers from a parse tree"""
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                if use_old_scheduling:
                    v = v * steps if v < 1 else v
                else:
                    if "." in s:
                        v = (v - flt_offset) * steps
                    else:
                        v = (v - int_offset)
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step: int, tree: lark.Tree) -> str:
        """Get the prompt state at a specific step"""
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after
                
            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]
                
            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
                
            def plain(self, args):
                yield args[0].value
                
            def __default__(self, data, children, meta):
                for child in children:
                    yield child
                    
        return AtStep().transform(tree)

    def get_schedule(prompt: str) -> List[PromptSchedule]:
        """Get the complete schedule for a single prompt"""
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            return [PromptSchedule(steps, prompt)]
            
        return [PromptSchedule(t, at_step(t, tree)) for t in collect_steps(steps, tree)]

    # Create schedule dictionary for deduplication
    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    
    # Return schedules in original prompt order
    return [promptdict[prompt] for prompt in prompts]

def get_learned_conditioning(
    model,
    prompts: Union[SdConditioning, List[str]], 
    steps: int,
    hires_steps: Optional[int] = None,
    use_old_scheduling: bool = False
) -> List[List[ScheduledPromptConditioning]]:
    """
    Convert prompts into learned conditioning schedules.
    
    Args:
        model: The model to use for conditioning
        prompts: List of prompts to process
        steps: Number of sampling steps
        hires_steps: Optional number of high-res steps
        use_old_scheduling: Whether to use legacy scheduling
        
    Returns:
        List of conditioning schedules for each prompt
    """
    res = []
    cache = {}

    prompt_schedules = get_learned_conditioning_prompt_schedules(
        prompts, steps, hires_steps, use_old_scheduling
    )

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        # Check cache first
        cached = cache.get(prompt)
        if cached is not None:
            res.append(cached)
            continue

        # Get conditioning for all steps
        texts = SdConditioning([x.prompt for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)

        # Create schedule
        cond_schedule = []
        for i, schedule in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]
            cond_schedule.append(ScheduledPromptConditioning(schedule.step, cond))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res

def get_multicond_prompt_list(prompts: Union[SdConditioning, List[str]]) -> Tuple[List, SdConditioning, Dict]:
    """
    Parse prompts with AND conditions into separate components with weights.
    
    Returns:
        Tuple containing:
        - List of (index, weight) pairs for each subprompt
        - Flattened list of all unique subprompts
        - Dictionary mapping subprompts to their indices
    """
    res_indexes = []
    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        subprompts = re_AND.split(prompt)
        indexes = []
        
        for subprompt in subprompts:
            match = re_weight.search(subprompt)
            text, weight = match.groups() if match else (subprompt, 1.0)
            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes

def get_multicond_learned_conditioning(
    model,
    prompts: List[str],
    steps: int,
    hires_steps: Optional[int] = None,
    use_old_scheduling: bool = False
) -> MulticondLearnedConditioning:
    """
    Get learned conditioning for prompts with multiple weighted conditions.
    
    This implements the method from:
    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """
    res_indexes, prompt_flat_list, _ = get_multicond_prompt_list(prompts)
    
    learned_conditioning = get_learned_conditioning(
        model, prompt_flat_list, steps, hires_steps, use_old_scheduling
    )

    res = []
    for indexes in res_indexes:
        res.append([
            ComposableScheduledPromptConditioning(learned_conditioning[i], weight) 
            for i, weight in indexes
        ])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)

def reconstruct_cond_batch(
    c: List[List[ScheduledPromptConditioning]], 
    current_step: int
) -> Union[torch.Tensor, DictWithShape]:
    """Reconstruct conditioning batch for the current step"""
    param = c[0][0].cond
    is_dict = isinstance(param, dict)

    if is_dict:
        dict_cond = param
        res = {
            k: torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)
            for k, param in dict_cond.items()
        }
        res = DictWithShape(res, (len(c),) + dict_cond['crossattn'].shape)
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)

    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, entry in enumerate(cond_schedule):
            if current_step <= entry.end_at_step:
                target_index = current
                break

        if is_dict:
            for k, param in cond_schedule[target_index].cond.items():
                res[k][i] = param
        else:
            res[i] = cond_schedule[target_index].cond

    return res

def stack_conds(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Stack tensors with padding to handle different lengths"""
    token_count = max(x.shape[0] for x in tensors)
    
    for i, tensor in enumerate(tensors):
        if tensor.shape[0] != token_count:
            last_vector = tensor[-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensor.shape[0], 1])
            tensors[i] = torch.vstack([tensor, last_vector_repeated])

    return torch.stack(tensors)

def reconstruct_multicond_batch(
    c: MulticondLearnedConditioning, 
    current_step: int
) -> Tuple[List, Union[torch.Tensor, DictWithShape]]:
    """Reconstruct multi-conditioning batch for current step"""
    param = c.batch[0][0].schedules[0].cond
    tensors = []
    conds_list = []

    for composable_prompts in c.batch:
        conds_for_batch = []

        for composable_prompt in composable_prompts:
            target_index = 0
            for current, entry in enumerate(composable_prompt.schedules):
                if current_step <= entry.end_at_step:
                    target_index = current
                    break

            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked, stacked['crossattn'].shape)
    else:
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)

    return conds_list, stacked

def parse_prompt_attention(text: str) -> List[Tuple[str, float]]:
    """
    Parse a prompt string with attention weights.
    
    Supports:
    - (abc) - increases attention to abc by 1.1
    - (abc:3.12) - increases attention to abc by 3.12
    - [abc] - decreases attention to abc by 1.1
    - Escaped characters: \( \[ \) \] \\
    
    Returns:
        List of (text, weight) tuples
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float) -> None:
        """Apply multiplier to weights in range"""
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    # Handle unclosed brackets
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    # Merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
else:
    import torch  # doctest faster