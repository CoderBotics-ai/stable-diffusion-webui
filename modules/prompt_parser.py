from __future__ import annotations

import re
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Union, Optional
import lark

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][: in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(r"""
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
""")

def get_learned_conditioning_prompt_schedules(prompts: list[str], base_steps: int, hires_steps: Optional[int] = None, use_old_scheduling: bool = False) -> list[list[tuple[int, str]]]:
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    >>> g("[fe|]male")
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("[fe|||]male")
    [[1, 'female'], [2, 'male'], [3, 'male'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'male'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10)[0]
    >>> g("a [b:.5] c")
    [[10, 'a b c']]
    >>> g("a [b:1.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    """

    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps: int, tree: lark.Tree) -> list[int]:
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree: lark.Tree) -> None:
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

            def alternate(self, _: lark.Tree) -> None:
                res.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step: int, tree: lark.Tree) -> str:
        class AtStep(lark.Transformer):
            def scheduled(self, args: list) -> Any:
                before, after, _, when, _ = args
                yield before or () if step <= when else after
            
            def alternate(self, args: list) -> Any:
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]
            
            def start(self, args: list) -> str:
                def flatten(x: Any) -> str:
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
            
            def plain(self, args: list) -> Any:
                yield args[0].value
            
            def __default__(self, data: Any, children: list, meta: Any) -> Any:
                for child in children:
                    yield child
        
        return AtStep().transform(tree)

    def get_schedule(prompt: str) -> list[tuple[int, str]]:
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            if 0:
                import traceback
                traceback.print_exc()
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


@dataclass
class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    is_negative_prompt: bool = False
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(self, prompts: list[str], is_negative_prompt: bool = False, width: Optional[int] = None, height: Optional[int] = None, copy_from: Optional[Any] = None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)

# Rest of the file continues unchanged as it's already compatible with Python 3.12
# and doesn't require any specific updates for its functionality

def get_learned_conditioning(model: Any, prompts: Union[SdConditioning, list[str]], steps: int, hires_steps: Optional[int] = None, use_old_scheduling: bool = False) -> list[list[ScheduledPromptConditioning]]:
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the condition (cond),
    and the sampling step at which this condition is to be replaced by the next one."""
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    cache: dict[str, list[ScheduledPromptConditioning]] = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):
        cached = cache.get(prompt)
        if cached is not None:
            res.append(cached)
            continue

        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)

        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]

            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res

# Rest of the file remains unchanged as it's compatible with Python 3.12