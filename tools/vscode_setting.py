import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--nokey", action="store_true")
parser.add_argument("--cs", action="store_true")
args = parser.parse_args()


path = args.path
if path is None:
    if args.cs:
        path = "/root/.local/share/code-server/User"
    else:
        path = "/home/docker/.config/Code/User"

setting_path = os.path.join(path, "settings.json")
if os.path.exists(setting_path):
    with open(setting_path) as f:
        setting = json.load(f)
else:
    print("[WARNING] setting.json not found")
    setting = {}


setting["workbench.colorTheme"] = "Default Dark+"
setting["security.workspace.trust.untrustedFiles"] =  "open"
setting["workbench.editorAssociations"] = {
    "*.ipynb": "jupyter-notebook"
}
setting["notebook.cellToolBarLocation"] = {
    "default": "right",
    "jupyter-notebook": "left"
}
setting["jupyter.askForKernelRestart"] = False
setting["editor.rulers"] = [{"color":"#000000", "column":80}]
setting["python.linting.enabled"] = False
setting["editor.acceptSuggestionOnEnter"] = "off"
setting["editor.hover.enabled"] = False
setting["editor.hover.sticky"] = False
setting["jupyter.askForKernelRestart"] = False
setting["workbench.editor.enablePreview"] = False
setting["notebook.lineNumbers"] = 'on'
setting["workbench.startupEditor"] = "none"
setting["editor.parameterHints.enabled"] = False
# added in 230719
setting["explorer.autoReveal"] = False
with open(setting_path, mode='w') as f:
    json.dump(setting, f, indent=4)

if not args.nokey:
    setting_path = os.path.join(path, "keybindings.json")
    if os.path.exists(setting_path):
        with open(setting_path) as f:
            setting = json.load(f)
    else:
        print("[WARNING] keybindigs.json not found")
        setting =[]
    setting += [{
        "key": "shift+tab",
        "command": "-outdent",
        "when": "editorTextFocus && !editorReadonly && !editorTabMovesFocus"
    },
    {
        "key": "ctrl+tab",
        "command": "outdent",
        "when": "editorTextFocus && !editorReadonly && !editorTabMovesFocus"
    }]
    with open(setting_path, mode='w') as f:
        json.dump(setting, f, indent=4)
    print("Successfully changed setting.")


