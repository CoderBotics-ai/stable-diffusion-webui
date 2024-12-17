from __future__ import annotations

import datetime
import html
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import gradio as gr
import re

from modules import ui_extra_networks_user_metadata


@dataclass
class LoraMetadata:
    description: str = ""
    sd_version: str = "Unknown"
    activation_text: str = ""
    preferred_weight: float = 0.0
    negative_text: str = ""
    notes: str = ""


def is_non_comma_tagset(tags: Dict[str, int]) -> bool:
    if not tags:
        return False
    average_tag_length = sum(len(x) for x in tags) / len(tags)
    return average_tag_length >= 16


re_word = re.compile(r"[-_\w']+")
re_comma = re.compile(r" *, *")


def build_tags(metadata: Dict[str, Any]) -> List[Tuple[str, int]]:
    tags: Dict[str, int] = {}

    ss_tag_frequency = metadata.get("ss_tag_frequency", {})
    if ss_tag_frequency is not None and hasattr(ss_tag_frequency, 'items'):
        for _, tags_dict in ss_tag_frequency.items():
            for tag, tag_count in tags_dict.items():
                tag = tag.strip()
                tags[tag] = tags.get(tag, 0) + int(tag_count)

    if tags and is_non_comma_tagset(tags):
        new_tags: Dict[str, int] = {}

        for text, text_count in tags.items():
            for word in re.findall(re_word, text):
                if len(word) < 3:
                    continue
                new_tags[word] = new_tags.get(word, 0) + text_count

        tags = new_tags

    ordered_tags = sorted(tags.keys(), key=lambda x: tags[x], reverse=True)
    return [(tag, tags[tag]) for tag in ordered_tags]


class LoraUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    def __init__(self, ui: Any, tabname: str, page: Any) -> None:
        super().__init__(ui, tabname, page)

        self.select_sd_version: Optional[gr.Dropdown] = None
        self.taginfo: Optional[gr.HighlightedText] = None
        self.edit_activation_text: Optional[gr.Text] = None
        self.slider_preferred_weight: Optional[gr.Slider] = None
        self.edit_notes: Optional[gr.TextArea] = None
        self.edit_negative_text: Optional[gr.Text] = None

    def save_lora_user_metadata(self, name: str, desc: str, sd_version: str, 
                              activation_text: str, preferred_weight: float, 
                              negative_text: str, notes: str) -> None:
        user_metadata = self.get_user_metadata(name)
        metadata = LoraMetadata(
            description=desc,
            sd_version=sd_version,
            activation_text=activation_text,
            preferred_weight=preferred_weight,
            negative_text=negative_text,
            notes=notes
        )
        
        user_metadata |= vars(metadata)  # Python 3.13 dict union operator
        self.write_user_metadata(name, user_metadata)

    def get_metadata_table(self, name: str) -> List[Tuple[str, str]]:
        table = super().get_metadata_table(name)
        item = self.page.items.get(name, {})
        metadata: Dict[str, Any] = item.get("metadata") or {}

        keys = {
            'ss_output_name': "Output name:",
            'ss_sd_model_name': "Model:",
            'ss_clip_skip': "Clip skip:",
            'ss_network_module': "Kohya module:",
        }

        table.extend(
            (label, html.escape(str(metadata[key])))
            for key, label in keys.items()
            if (value := metadata.get(key)) is not None and str(value) != "None"
        )

        if ss_training_started_at := metadata.get('ss_training_started_at'):
            table.append((
                "Date trained:",
                datetime.datetime.fromtimestamp(float(ss_training_started_at), tz=datetime.UTC)
                .strftime('%Y-%m-%d %H:%M')
            ))

        if ss_bucket_info := metadata.get("ss_bucket_info"):
            if "buckets" in ss_bucket_info:
                resolutions: Dict[str, int] = {}
                for bucket in ss_bucket_info["buckets"].values():
                    resolution = bucket["resolution"]
                    res_str = f'{resolution[1]}x{resolution[0]}'
                    resolutions[res_str] = resolutions.get(res_str, 0) + int(bucket["count"])

                resolutions_list = sorted(resolutions.keys(), key=lambda x: resolutions[x], reverse=True)
                resolutions_text = html.escape(", ".join(resolutions_list[:4]))
                if len(resolutions) > 4:
                    resolutions_text += ", ..."
                    resolutions_text = f"<span title='{html.escape(', '.join(resolutions_list))}'>{resolutions_text}</span>"

                table.append(('Resolutions:' if len(resolutions_list) > 1 else 'Resolution:', resolutions_text))

        image_count = sum(
            int(params.get("img_count", 0))
            for params in metadata.get("ss_dataset_dirs", {}).values()
        )

        if image_count:
            table.append(("Dataset size:", image_count))

        return table

    def put_values_into_components(self, name: str) -> List[Any]:
        user_metadata = self.get_user_metadata(name)
        values = super().put_values_into_components(name)

        item = self.page.items.get(name, {})
        metadata = item.get("metadata") or {}

        tags = build_tags(metadata)
        gradio_tags = [(tag, str(count)) for tag, count in tags[:24]]

        return [
            *values[:5],
            item.get("sd_version", "Unknown"),
            gr.HighlightedText.update(value=gradio_tags, visible=bool(tags)),
            user_metadata.get('activation text', ''),
            float(user_metadata.get('preferred weight', 0.0)),
            user_metadata.get('negative text', ''),
            gr.update(visible=bool(tags)),
            gr.update(value=self.generate_random_prompt_from_tags(tags), visible=bool(tags)),
        ]

    def generate_random_prompt(self, name: str) -> str:
        item = self.page.items.get(name, {})
        metadata = item.get("metadata") or {}
        tags = build_tags(metadata)
        return self.generate_random_prompt_from_tags(tags)

    def generate_random_prompt_from_tags(self, tags: List[Tuple[str, int]]) -> str:
        if not tags:
            return ""
            
        max_count = tags[0][1]
        res = []
        for tag, count in tags:
            v = random.random() * max_count
            if count > v:
                tag = re.escape(tag)  # More robust way to escape special characters
                res.append(tag)

        return ", ".join(sorted(res))

    def create_extra_default_items_in_left_column(self) -> None:
        self.select_sd_version = gr.Dropdown(
            choices=['SD1', 'SD2', 'SDXL', 'Unknown'],
            value='Unknown',
            label='Stable Diffusion version',
            interactive=True
        )

    def create_editor(self) -> None:
        self.create_default_editor_elems()

        self.taginfo = gr.HighlightedText(label="Training dataset tags")
        self.edit_activation_text = gr.Text(label='Activation text', info="Will be added to prompt along with Lora")
        self.slider_preferred_weight = gr.Slider(
            label='Preferred weight',
            info="Set to 0 to disable",
            minimum=0.0,
            maximum=2.0,
            step=0.01
        )
        self.edit_negative_text = gr.Text(label='Negative prompt', info="Will be added to negative prompts")
        
        with gr.Row() as row_random_prompt:
            with gr.Column(scale=8):
                random_prompt = gr.Textbox(label='Random prompt', lines=4, max_lines=4, interactive=False)
            with gr.Column(scale=1, min_width=120):
                generate_random_prompt = gr.Button('Generate', size="lg", scale=1)

        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        generate_random_prompt.click(
            fn=self.generate_random_prompt,
            inputs=[self.edit_name_input],
            outputs=[random_prompt],
            show_progress=False
        )

        def select_tag(activation_text: str, evt: gr.SelectData) -> str:
            tag = evt.value[0]
            words = re.split(re_comma, activation_text or "")
            if tag in words:
                return ", ".join(word for word in words if word != tag and word.strip())
            return f"{activation_text}, {tag}" if activation_text else tag

        self.taginfo.select(
            fn=select_tag,
            inputs=[self.edit_activation_text],
            outputs=[self.edit_activation_text],
            show_progress=False
        )

        self.create_default_buttons()

        viewed_components = [
            self.edit_name,
            self.edit_description,
            self.html_filedata,
            self.html_preview,
            self.edit_notes,
            self.select_sd_version,
            self.taginfo,
            self.edit_activation_text,
            self.slider_preferred_weight,
            self.edit_negative_text,
            row_random_prompt,
            random_prompt,
        ]

        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=viewed_components)\
            .then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])

        edited_components = [
            self.edit_description,
            self.select_sd_version,
            self.edit_activation_text,
            self.slider_preferred_weight,
            self.edit_negative_text,
            self.edit_notes,
        ]

        self.setup_save_handler(self.button_save, self.save_lora_user_metadata, edited_components)