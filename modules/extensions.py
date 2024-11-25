from __future__ import annotations

import configparser
import dataclasses
import os
import threading
import re

from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

extensions: list[Extension] = []
extension_paths: dict[str, Extension] = {}
loaded_extensions: dict[str, Exception] = {}

os.makedirs(extensions_dir, exist_ok=True)

def active() -> list[Extension]:
    if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
        return []
    elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]

@dataclasses.dataclass
class CallbackOrderInfo:
    name: str
    before: list[str]
    after: list[str]

class ExtensionMetadata:
    filename = "metadata.ini"
    config: configparser.ConfigParser
    canonical_name: str
    requires: list[str]

    def __init__(self, path: str, canonical_name: str):
        self.config = configparser.ConfigParser()

        filepath = os.path.join(path, self.filename)
        try:
            self.config.read(filepath)
        except Exception:
            errors.report(f"Error reading {self.filename} for extension {canonical_name}.", exc_info=True)

        self.canonical_name = self.config.get("Extension", "Name", fallback=canonical_name).lower().strip()
        self.requires = []

    def get_script_requirements(self, field: str, section: str, extra_section: str = None) -> list[str]:
        x = self.config.get(section, field, fallback='')

        if extra_section:
            x += ', ' + self.config.get(extra_section, field, fallback='')

        listed_requirements = self.parse_list(x.lower())
        res = []

        for requirement in listed_requirements:
            loaded_requirements = (x for x in requirement.split("|") if x in loaded_extensions)
            relevant_requirement = next(loaded_requirements, requirement)
            res.append(relevant_requirement)

        return res

    def parse_list(self, text: str) -> list[str]:
        if not text:
            return []
        return [x for x in re.split(r"[,\s]+", text.strip()) if x]

    def list_callback_order_instructions(self):
        for section in self.config.sections():
            if not section.startswith("callbacks/"):
                continue

            callback_name = section[10:]

            if not callback_name.startswith(self.canonical_name):
                errors.report(f"Callback order section for extension {self.canonical_name} is referencing the wrong extension: {section}")
                continue

            before = self.parse_list(self.config.get(section, 'Before', fallback=''))
            after = self.parse_list(self.config.get(section, 'After', fallback=''))

            yield CallbackOrderInfo(callback_name, before, after)

class Extension:
    lock = threading.Lock()
    cached_fields = ['remote', 'commit_date', 'branch', 'commit_hash', 'version']
    metadata: ExtensionMetadata

    def __init__(self, name: str, path: str, enabled: bool = True, is_builtin: bool = False, metadata: ExtensionMetadata = None):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False
        self.metadata = metadata if metadata else ExtensionMetadata(self.path, name.lower())
        self.canonical_name = self.metadata.canonical_name

    def to_dict(self) -> dict:
        return {x: getattr(self, x) for x in self.cached_fields}

    def from_dict(self, d: dict):
        for field in self.cached_fields:
            setattr(self, field, d[field])

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        def read_from_repo():
            with self.lock:
                if self.have_info_from_repo:
                    return
                self.do_read_info_from_repo()
                return self.to_dict()

        try:
            d = cache.cached_data_for_file('extensions-git', self.name, os.path.join(self.path, ".git"), read_from_repo)
            self.from_dict(d)
        except FileNotFoundError:
            pass
        self.status = 'unknown' if self.status == '' else self.status

    def do_read_info_from_repo(self):
        repo = None
        try:
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = Repo(self.path)
        except Exception:
            errors.report(f"Error reading github repository info from {self.path}", exc_info=True)

        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                self.remote = next(repo.remote().urls, None)
                commit = repo.head.commit
                self.commit_date = commit.committed_date
                if repo.active_branch:
                    self.branch = repo.active_branch.name
                self.commit_hash = commit.hexsha
                self.version = self.commit_hash[:8]

            except Exception:
                errors.report(f"Failed reading extension data from Git repository ({self.name})", exc_info=True)
                self.remote = None

        self.have_info_from_repo = True

    def list_files(self, subdir: str, extension: str) -> list[scripts.ScriptFile]:
        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        return [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

    def check_updates(self):
        repo = Repo(self.path)
        branch_name = f'{repo.remote().name}/{self.branch}'
        for fetch in repo.remote().fetch(dry_run=True):
            if self.branch and fetch.name != branch_name:
                continue
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "new commits"
                return

        try:
            origin = repo.rev_parse(branch_name)
            if repo.head.commit != origin:
                self.can_update = True
                self.status = "behind HEAD"
                return
        except Exception:
            self.can_update = False
            self.status = "unknown (remote error)"
            return

        self.can_update = False
        self.status = "latest"

    def fetch_and_reset_hard(self, commit: str = None):
        repo = Repo(self.path)
        if commit is None:
            commit = f'{repo.remote().name}/{self.branch}'
        repo.git.fetch(all=True)
        repo.git.reset(commit, hard=True)
        self.have_info_from_repo = False

def list_extensions():
    extensions.clear()
    extension_paths.clear()
    loaded_extensions.clear()

    if shared.cmd_opts.disable_all_extensions:
        print("*** \"--disable-all-extensions\" arg was used, will not load any extensions ***")
    elif shared.opts.disable_all_extensions == "all":
        print("*** \"Disable all extensions\" option was set, will not load any extensions ***")
    elif shared.cmd_opts.disable_extra_extensions:
        print("*** \"--disable-extra-extensions\" arg was used, will only load built-in extensions ***")
    elif shared.opts.disable_all_extensions == "extra":
        print("*** \"Disable all extensions\" option was set, will only load built-in extensions ***")

    for dirname in [extensions_builtin_dir, extensions_dir]:
        if not os.path.isdir(dirname):
            continue

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            canonical_name = extension_dirname
            metadata = ExtensionMetadata(path, canonical_name)

            already_loaded_extension = loaded_extensions.get(metadata.canonical_name)
            if already_loaded_extension is not None:
                errors.report(f'Duplicate canonical name "{canonical_name}" found in extensions "{extension_dirname}" and "{already_loaded_extension.name}". Former will be discarded.', exc_info=False)
                continue

            is_builtin = dirname == extensions_builtin_dir
            extension = Extension(name=extension_dirname, path=path, enabled=extension_dirname not in shared.opts.disabled_extensions, is_builtin=is_builtin, metadata=metadata)
            extensions.append(extension)
            extension_paths[extension.path] = extension
            loaded_extensions[canonical_name] = extension

    for extension in extensions:
        extension.metadata.requires = extension.metadata.get_script_requirements("Requires", "Extension")

    for extension in extensions:
        if not extension.enabled:
            continue

        for req in extension.metadata.requires:
            required_extension = loaded_extensions.get(req)
            if required_extension is None:
                errors.report(f'Extension "{extension.name}" requires "{req}" which is not installed.', exc_info=False)
                continue

            if not required_extension.enabled:
                errors.report(f'Extension "{extension.name}" requires "{required_extension.name}" which is disabled.', exc_info=False)
                continue

def find_extension(filename: str) -> Extension | None:
    parentdir = os.path.dirname(os.path.realpath(filename))

    while parentdir != filename:
        extension = extension_paths.get(parentdir)
        if extension is not None:
            return extension

        filename = parentdir
        parentdir = os.path.dirname(filename)

    return None