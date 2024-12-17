from __future__ import annotations

import re
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import lark
import torch

# Grammar for parsing prompt schedules
SCHEDULE_GRAMMAR = r"""
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

# Parser for prompt schedules
schedule_parser = lark.Lark(SCHEDULE_GRAMMAR)

@dataclass
class PromptChunk:
    """
    Contains token ids, weights and textual inversion embedding info for a chunk of prompt.
    Each chunk contains exactly 77 tokens (75 from prompt + start/end tokens).
    """
    tokens: List[int]
    multipliers: List[float] 
    fixes: List[Any]

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []

# Named tuple for marking textual inversion embedding placement
PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])

@dataclass
class SdConditioning(list):
    """
    List containing prompts for stable diffusion's conditioner model.
    Can specify width and height of created image - required for SDXL.
    """
    is_negative_prompt: bool = False
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(self, prompts: List[str], is_negative_prompt: bool = False, 
                 width: Optional[int] = None, height: Optional[int] = None, 
                 copy_from: Optional[Any] = None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)

def get_learned_conditioning_prompt_schedules(prompts: Union[List[str], SdConditioning], 
                                           base_steps: int,
                                           hires_steps: Optional[int] = None,
                                           use_old_scheduling: bool = False) -> List[List[Tuple[int, str]]]:
    """
    Converts a list of prompts into prompt schedules - each schedule specifies when to switch prompts.
    
    Args:
        prompts: List of prompt strings or SdConditioning object
        base_steps: Number of base sampling steps
        hires_steps: Optional number of high-res sampling steps
        use_old_scheduling: Whether to use old scheduling behavior
        
    Returns:
        List of prompt schedules, where each schedule is a list of (step, prompt) tuples
        
    Example:
        >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
        >>> g("test")
        [[10, 'test']]
        >>> g("a [b:3]")
        [[3, 'a '], [10, 'a b']]
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
        """Collect all step numbers from a parsed prompt tree"""
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
        """Get the prompt text at a specific step"""
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

    def get_schedule(prompt: str) -> List[Tuple[int, str]]:
        """Get the full schedule for a single prompt"""
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            return [[steps, prompt]]
            
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]

def get_multicond_prompt_list(prompts: Union[SdConditioning, List[str]]) -> Tuple[List[List[Tuple[int, float]]], SdConditioning, Dict[str, int]]:
    """
    Parse prompts with AND/weight syntax into prompt indices and weights.
    
    Returns:
        Tuple containing:
        - List of prompt index/weight pairs for each prompt
        - Flattened list of unique prompts
        - Dictionary mapping prompt text to index
    """
    res_indexes = []
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()
    prompt_indexes: Dict[str, int] = {}

    re_AND = re.compile(r"\bAND\b")
    re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?\d*|\.\d+)))?\s*$")

    for prompt in prompts:
        subprompts = re_AND.split(prompt)
        indexes = []
        
        for subprompt in subprompts:
            match = re_weight.search(subprompt)
            text, weight = match.groups() if match else (subprompt, "1.0")
            weight = float(weight) if weight is not None else 1.0

            idx = prompt_indexes.get(text)
            if idx is None:
                idx = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = idx

            indexes.append((idx, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes

@dataclass 
class ComposableScheduledPromptConditioning:
    """Container for composable prompt schedules with weights"""
    schedules: List[Any]  # List[ScheduledPromptConditioning]
    weight: float = 1.0

@dataclass
class MulticondLearnedConditioning:
    """Container for multiple learned conditioning schedules"""
    shape: Tuple[int, ...]
    batch: List[List[ComposableScheduledPromptConditioning]]

def get_multicond_learned_conditioning(model: Any, prompts: Union[List[str], SdConditioning], 
                                     steps: int, hires_steps: Optional[int] = None,
                                     use_old_scheduling: bool = False) -> MulticondLearnedConditioning:
    """
    Get learned conditioning for multiple prompts with weights and schedules.
    Handles AND syntax and prompt weights.
    
    Returns MulticondLearnedConditioning containing the conditioning schedules.
    """
    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, 
                                                 hires_steps, use_old_scheduling)

    res = []
    for indexes in res_indexes:
        res.append([
            ComposableScheduledPromptConditioning(learned_conditioning[i], weight) 
            for i, weight in indexes
        ])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)

class DictWithShape(dict):
    """Dictionary that also stores a shape attribute"""
    def __init__(self, x: Dict, shape: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.update(x)
        self._shape = shape

    @property 
    def shape(self) -> Tuple[int, ...]:
        return self["crossattn"].shape if self._shape is None else self._shape

def reconstruct_cond_batch(c: List[List[Any]], current_step: int) -> Union[torch.Tensor, DictWithShape]:
    """
    Reconstruct conditioning batch from schedules at current step.
    Handles both tensor and dictionary conditioning types.
    """
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
    """Stack tensor conditions, handling different lengths by padding"""
    # If prompts have wildly different lengths above the limit we'll get tensors of different shapes
    # and won't be able to torch.stack them. So this fixes that.
    pass # TODO: Implement tensor stacking with padding