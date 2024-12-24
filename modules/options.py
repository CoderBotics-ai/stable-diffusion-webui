import os
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Callable

import gradio as gr

from modules import errors
from modules.shared_cmd_options import cmd_opts
from modules.paths_internal import script_path


@dataclass
class OptionInfo:
    default: Any = None
    label: str = ""
    component: Optional[Callable] = None
    component_args: Optional[Dict[str, Any]] = None
    onchange: Optional[Callable] = None
    section: Optional[tuple] = None
    refresh: Optional[bool] = None
    comment_before: str = ''
    comment_after: str = ''
    infotext: Optional[str] = None
    restrict_api: bool = False
    category_id: Optional[str] = None
    do_not_save: bool = False

    def link(self, label: str, url: str) -> 'OptionInfo':
        self.comment_before += f"[<a href='{url}' target='_blank'>{label}</a>]"
        return self

    def js(self, label: str, js_func: str) -> 'OptionInfo':
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    def info(self, info: str) -> 'OptionInfo':
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    def html(self, html: str) -> 'OptionInfo':
        self.comment_after += html
        return self

    def needs_restart(self) -> 'OptionInfo':
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self

    def needs_reload_ui(self) -> 'OptionInfo':
        self.comment_after += " <span class='info'>(requires Reload UI)</span>"
        return self


class OptionHTML(OptionInfo):
    def __init__(self, text: str):
        super().__init__(
            default=str(text).strip(),
            label='',
            component=lambda **kwargs: gr.HTML(elem_classes="settings-info", **kwargs)
        )
        self.do_not_save = True


def options_section(
    section_identifier: Union[tuple[str, str], tuple[str, str, str]],
    options_dict: Dict[str, OptionInfo]
) -> Dict[str, OptionInfo]:
    match len(section_identifier):
        case 2:
            for v in options_dict.values():
                v.section = section_identifier
        case 3:
            for v in options_dict.values():
                v.section = section_identifier[0:2]
                v.category_id = section_identifier[2]

    return options_dict


options_builtin_fields: set[str] = {"data_labels", "data", "restricted_opts", "typemap"}


class Options:
    typemap: Dict[type, type] = {int: float}

    def __init__(self, data_labels: Dict[str, OptionInfo], restricted_opts: list[str]):
        self.data_labels = data_labels
        self.data = {k: v.default for k, v in self.data_labels.items() if not v.do_not_save}
        self.restricted_opts = restricted_opts

    def __setattr__(self, key: str, value: Any) -> None:
        if key in options_builtin_fields:
            return super().__setattr__(key, value)

        if hasattr(self, 'data') and self.data is not None:
            if key in self.data or key in self.data_labels:
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                info = self.data_labels.get(key)
                if info and info.do_not_save:
                    return

                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set '{key}' because it is restricted")

                if cmd_opts.freeze_settings_in_sections is not None:
                    frozen_sections = [s.strip() for s in cmd_opts.freeze_settings_in_sections.split(',')]
                    section_key = info.section[0]
                    section_name = info.section[1]
                    assert section_key not in frozen_sections, f"not possible to set '{key}' because settings in section '{section_name}' ({section_key}) are frozen with --freeze-settings-in-sections"

                if cmd_opts.freeze_specific_settings is not None:
                    frozen_keys = [k.strip() for k in cmd_opts.freeze_specific_settings.split(',')]
                    assert key not in frozen_keys, f"not possible to set '{key}' because this setting is frozen with --freeze-specific-settings"

                if cmd_opts.hide_ui_dir_config and key in self.restricted_opts:
                    raise RuntimeError(f"not possible to set '{key}' because it is restricted with --hide_ui_dir_config")

                self.data[key] = value
                return

        return super().__setattr__(key, value)

    def __getattr__(self, item: str) -> Any:
        if item in options_builtin_fields:
            return super().__getattribute__(item)

        if hasattr(self, 'data') and self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super().__getattribute__(item)

    def set(self, key: str, value: Any, is_api: bool = False, run_callbacks: bool = True) -> bool:
        """Sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""
        
        oldval = self.data.get(key)
        if oldval == value:
            return False

        option = self.data_labels[key]
        if option.do_not_save:
            return False

        if is_api and option.restrict_api:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if run_callbacks and option.onchange is not None:
            try:
                option.onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def get_default(self, key: str) -> Any:
        """Returns the default value for the key"""
        data_label = self.data_labels.get(key)
        return None if data_label is None else data_label.default

    def save(self, filename: str) -> None:
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)

    def same_type(self, x: Any, y: Any) -> bool:
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename: str) -> None:
        try:
            with open(filename, "r", encoding="utf8") as file:
                self.data = json.load(file)
        except FileNotFoundError:
            self.data = {}
        except Exception:
            errors.report(
                f'\nCould not load settings\nThe config file "{filename}" is likely corrupted'
                f'\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n',
                exc_info=True
            )
            os.replace(filename, os.path.join(script_path, "tmp", "config.json"))
            self.data = {}

        # Handle migrations and defaults
        if self.data.get('sd_vae_as_default') is not None and self.data.get('sd_vae_overrides_per_model_preferences') is None:
            self.data['sd_vae_overrides_per_model_preferences'] = not self.data.get('sd_vae_as_default')

        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]

        if isinstance(self.data.get('ui_reorder'), str) and self.data.get('ui_reorder') and "ui_reorder_list" not in self.data:
            self.data['ui_reorder_list'] = [i.strip() for i in self.data.get('ui_reorder').split(',')]

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key: str, func: Callable, call: bool = True) -> None:
        item = self.data_labels.get(key)
        if item:
            item.onchange = func
            if call:
                func()

    def dumpjson(self) -> str:
        d = {k: self.data.get(k, v.default) for k, v in self.data_labels.items()}
        d["_comments_before"] = {k: v.comment_before for k, v in self.data_labels.items() if v.comment_before}
        d["_comments_after"] = {k: v.comment_after for k, v in self.data_labels.items() if v.comment_after}

        item_categories: Dict[str, str] = {}
        for item in self.data_labels.values():
            if item.section[0] is None:
                continue

            category = categories.mapping.get(item.category_id)
            category = "Uncategorized" if category is None else category.label
            if category not in item_categories:
                item_categories[category] = item.section[1]

        d["_categories"] = [[v, k] for k, v in item_categories.items()] + [["Defaults", "Other"]]

        return json.dumps(d)

    def add_option(self, key: str, info: OptionInfo) -> None:
        self.data_labels[key] = info
        if key not in self.data and not info.do_not_save:
            self.data[key] = info.default

    def reorder(self) -> None:
        """
        Reorder settings so that:
        - all items related to section always go together
        - all sections belonging to a category go together
        - sections inside a category are ordered alphabetically
        - categories are ordered by creation order
        """
        category_ids: Dict[str, int] = {}
        section_categories: Dict[tuple, Optional[str]] = {}

        settings_items = self.data_labels.items()
        for _, item in settings_items:
            if item.section not in section_categories:
                section_categories[item.section] = item.category_id

        for _, item in settings_items:
            item.category_id = section_categories.get(item.section)

        for category_id in categories.mapping:
            if category_id not in category_ids:
                category_ids[category_id] = len(category_ids)

        def sort_key(x: tuple[str, OptionInfo]) -> tuple[int, str]:
            item: OptionInfo = x[1]
            category_order = category_ids.get(item.category_id, len(category_ids))
            section_order = item.section[1]
            return category_order, section_order

        self.data_labels = dict(sorted(settings_items, key=sort_key))

    def cast_value(self, key: str, value: Any) -> Any:
        """
        Casts an arbitrary value to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """
        if value is None:
            return None

        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None

        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        else:
            value = expected_type(value)

        return value


@dataclass
class OptionsCategory:
    id: str
    label: str


class OptionsCategories:
    def __init__(self):
        self.mapping: Dict[str, OptionsCategory] = {}

    def register_category(self, category_id: str, label: str) -> str:
        if category_id not in self.mapping:
            self.mapping[category_id] = OptionsCategory(category_id, label)
        return category_id


categories = OptionsCategories()